"""教学循环测试（p2-teaching，FR-401~FR-416）。

这是项目里最重要的一组测试，重点覆盖三类"做错了很严重"的行为：

- **门禁**：只有判定为 pass 才推进；重试耗尽必须停下等用户选择（FR-403、FR-408）；
- **不变量**：判定失败不得消耗尝试次数、不得清掉待回答问题（否则一次网络抖动就白丢一次机会）；
- **可恢复**：刷新/重启后仍能续上"待回答的检查点"或"待选择的三选一"（FR-410）。
"""

import json

import pytest

from app.contracts.schemas import MasteryLevel, NodeDecision, NodeState, SubjectState, WeakPointReason
from app.infra.settings import AppSettings
from app.main_context import AppContext
from app.services.learning.assessment import AssessmentService
from app.services.learning.planning import PlanningService
from app.services.learning.state import AnswerNotAllowedError
from app.services.learning.teaching import QUESTION_TRANSITION, TeachingService
from tests.learning_harness import LearningHarness, make_harness
from tests.stubs import ScriptedResponse, json_response, text_events

TOPICS_JSON = json.dumps(
    {"topics": ["定义", "公式"], "first_question": "先说说它解决什么问题？"},
    ensure_ascii=False,
)

PROFILE_JSON = json.dumps(
    {
        "levels": [{"topic": "定义", "level": "none", "evidence": "未作答"}],
        "summary": "需要从基础开始。",
        "suggested_start_node": "从定义开始。",
    },
    ensure_ascii=False,
)


def node_payload(title: str, checkpoints: int = 1) -> dict:
    return {
        "title": title,
        "objective": f"能解释 {title}",
        "prerequisites": [],
        "key_points": ["要点一", "要点二", "要点三"],
        "checkpoints": [
            {
                "question": f"{title} 的检查点 {index}？",
                "expected_points": [f"判定要点 {index}"],
                "reference_answer": f"参考答案 {index}",
            }
            for index in range(checkpoints)
        ],
        "is_quick_review": False,
    }


def path_json(node_count: int = 2, checkpoints: int = 1) -> str:
    return json.dumps(
        {
            "rationale": "从基础开始逐点推进。",
            "nodes": [node_payload(f"节点 {index}", checkpoints) for index in range(node_count)],
        },
        ensure_ascii=False,
    )


def verdict_json(verdict: str, *, hit: bool = True) -> str:
    payload: dict = {
        "verdict": verdict,
        "confidence": 0.8 if verdict == "pass" else 0.4,
        "feedback": "反馈内容",
    }
    if verdict == "pass":
        payload["hit_points"] = ["判定要点 0"] if hit else []
        payload["missing_points"] = []
    else:
        payload["hit_points"] = []
        payload["missing_points"] = ["判定要点 0"]
    return json.dumps(payload, ensure_ascii=False)


def build_services(harness: LearningHarness) -> tuple[AssessmentService, PlanningService, TeachingService]:
    assessments = AssessmentService(harness.repos, harness.state, harness.sessions, lambda: harness.settings)
    planning = PlanningService(
        harness.repos,
        harness.state,
        harness.sessions,
        assessments,
        lambda: harness.settings,
        harness.policy,
    )
    teaching = TeachingService(
        harness.repos,
        harness.state,
        harness.sessions,
        planning,
        assessments,
        lambda: harness.settings,
    )
    return assessments, planning, teaching


def create_path(
    harness: LearningHarness,
    subject_id: str,
    *,
    node_count: int,
    checkpoints: int = 1,
) -> str:
    """直接建一条路径。

    教学测试关心的是教学循环，不是路径生成（后者由 test_planning_service.py 覆盖）；
    直接建路径可以避免"先生成路径"带来的额外 LLM 调用与节点数约束干扰。
    """
    path = harness.repos.paths.create(
        subject_id,
        rationale="从基础开始逐点推进。",
        model="deepseek-flash",
        nodes=[node_payload(f"节点 {index}", checkpoints) for index in range(node_count)],
    )
    return path.id


def enter_teaching(
    harness: LearningHarness,
    *,
    node_count: int = 2,
    checkpoints: int = 1,
    confirmed: bool = True,
) -> tuple[AssessmentService, PlanningService, TeachingService, str]:
    """进入教学态：创建主题 → 跳过测评 → 建路径 → （可选）确认路径。"""
    assessments, planning, teaching = build_services(harness)
    subject_id = harness.new_subject()
    harness.state.transition(subject_id, SubjectState.PATH_REVIEW)
    create_path(harness, subject_id, node_count=node_count, checkpoints=checkpoints)
    if confirmed:
        planning.confirm(subject_id)

    harness.client.stream_script = []
    harness.client.blocking_script = []
    harness.client.stream_calls = 0
    harness.client.blocking_calls = 0
    return assessments, planning, teaching, subject_id


async def enter_teaching_after_assessment(
    harness: LearningHarness,
    *,
    node_count: int = 3,
    checkpoints: int = 1,
) -> tuple[AssessmentService, PlanningService, TeachingService, str]:
    """走完整流程（含真实测评）进入教学态，用于验证记忆隔离等跨阶段行为。"""
    harness.client.blocking_script = [
        json_response(TOPICS_JSON),
        json_response(PROFILE_JSON),
        json_response(path_json(node_count, checkpoints)),
    ]
    harness.client.stream_script = [ScriptedResponse(events=text_events("第二方面的问题。"))]

    assessments, planning, teaching = build_services(harness)
    subject_id = harness.new_subject()
    await assessments.start(subject_id)
    await assessments.answer(subject_id, "回答一")
    await assessments.answer(subject_id, "回答二")
    await planning.generate(subject_id)
    planning.confirm(subject_id)

    harness.client.stream_script = []
    harness.client.blocking_script = []
    harness.client.stream_calls = 0
    harness.client.blocking_calls = 0
    return assessments, planning, teaching, subject_id


# --------------------------------------------------------------------------- #
# 讲解 + 提问
# --------------------------------------------------------------------------- #


async def test_start_explains_then_asks_first_checkpoint(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness)
    harness.client.stream_script = [
        ScriptedResponse(events=text_events("这是讲解内容。")),
    ]
    deltas: list[str] = []

    turn = await teaching.start(subject_id, on_delta=lambda event: deltas.append(event.content or ""))

    assert turn.ok
    assert turn.phase == "question"
    assert "这是讲解内容。" in turn.text
    assert "节点 0 的检查点 0？" in turn.text
    assert turn.node is not None and turn.node.title == "节点 0"
    assert turn.node_index == 1 and turn.node_total == 2

    # 讲解与问题都推给了前端（前端只需追加收到的一切）
    assert "这是讲解内容。" in "".join(deltas)
    assert QUESTION_TRANSITION in "".join(deltas)

    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.has_pending_question is True
    assert snapshot.node_taught is True
    assert snapshot.pending_question == "节点 0 的检查点 0？"


async def test_start_does_not_explain_twice(context: AppContext) -> None:
    """AC-12：已经讲解过并抛出问题后，再次进入不应重复讲解。"""
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解一次。"))]
    await teaching.start(subject_id)
    calls_after_first = harness.client.stream_calls

    second = await teaching.start(subject_id)

    assert second.phase == "question"
    assert "讲解一次。" not in second.text
    assert second.text == QUESTION_TRANSITION + "节点 0 的检查点 0？"
    # 没有产生新的模型调用
    assert harness.client.stream_calls == calls_after_first


async def test_start_resumes_pending_question_without_new_calls(context: AppContext) -> None:
    """FR-410：刷新后恢复——重新装配服务后重新进入，直接给出待回答的问题。"""
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解。"))]
    await teaching.start(subject_id)
    calls = harness.client.stream_calls

    # 模拟"重新打开应用"：用同一份数据库重新装配服务
    _, _, fresh_teaching = build_services(harness)
    resumed = await fresh_teaching.start(subject_id)

    assert resumed.phase == "question"
    assert resumed.text == QUESTION_TRANSITION + "节点 0 的检查点 0？"
    assert harness.client.stream_calls == calls  # 没有再次调用模型


async def test_start_returns_options_when_awaiting_decision(context: AppContext) -> None:
    """FR-408 + FR-410：等三选一的状态落库，重启后仍给选项而不是重新讲解。"""
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.settings = AppSettings(node_max_retries=0)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解。"))]
    harness.client.blocking_script = [json_response(verdict_json("fail"))]
    await teaching.start(subject_id)

    await teaching.answer(subject_id, "第一次作答")  # attempts=1 > 0 → 立刻要决策

    assert harness.state.snapshot(subject_id).needs_user_decision is True

    # 换一套服务实例（模拟重启）后仍应给选项
    _, _, fresh = build_services(harness)
    again = await fresh.start(subject_id)

    assert again.phase == "decision"
    assert again.decision_required is True


async def test_start_rejects_when_path_not_confirmed(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2, confirmed=False)

    with pytest.raises(AnswerNotAllowedError, match="学习还没开始"):
        await teaching.start(subject_id)


# --------------------------------------------------------------------------- #
# 通过：推进与小结
# --------------------------------------------------------------------------- #


async def test_answer_pass_masters_node_and_writes_note(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=1)
    harness.client.stream_script = [
        ScriptedResponse(events=text_events("讲解。")),
        ScriptedResponse(events=text_events("小结：这个知识点解决的是……")),
    ]
    harness.client.blocking_script = [json_response(verdict_json("pass"))]
    await teaching.start(subject_id)

    turn = await teaching.answer(subject_id, "我的正确回答")

    assert turn.ok
    assert turn.passed is True
    assert turn.phase == "completed"  # 只有一个节点 → 完成后主题结束
    assert turn.completed is True
    assert "小结：这个知识点解决的是……" in turn.text

    node = harness.repos.paths.first_node(harness.repos.paths.get_active(subject_id).id)
    assert node.state is NodeState.MASTERED
    assert node.summary_note == "小结：这个知识点解决的是……"
    assert harness.repos.notes.get(subject_id, node.id).content == "小结：这个知识点解决的是……"  # type: ignore[union-attr]

    assert harness.state.snapshot(subject_id).state is SubjectState.COMPLETED
    assert harness.state.snapshot(subject_id).has_pending_question is False


async def test_answer_pass_records_answer_with_hit_points(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [
        ScriptedResponse(events=text_events("讲解。")),
        ScriptedResponse(events=text_events("小结。")),
    ]
    harness.client.blocking_script = [json_response(verdict_json("pass"))]
    await teaching.start(subject_id)

    await teaching.answer(subject_id, "我的回答")

    records = harness.repos.answers.list_for_subject(subject_id)
    assert len(records) == 1
    assert records[0].verdict == "pass"
    assert records[0].hit_points == ("判定要点 0",)
    assert records[0].user_answer == "我的回答"
    assert records[0].attempt_index == 1


async def test_pass_with_multiple_checkpoints_asks_the_next_one(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=1, checkpoints=2)
    harness.client.stream_script = [
        ScriptedResponse(events=text_events("讲解。")),
        ScriptedResponse(events=text_events("小结。")),
    ]
    harness.client.blocking_script = [json_response(verdict_json("pass")), json_response(verdict_json("pass"))]
    await teaching.start(subject_id)

    first = await teaching.answer(subject_id, "回答一")

    assert first.phase == "question"
    assert "检查点 1" in first.text
    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.has_pending_question is True
    assert snapshot.pending_question == "节点 0 的检查点 1？"

    second = await teaching.answer(subject_id, "回答二")

    assert second.passed is True
    node = harness.repos.paths.first_node(harness.repos.paths.get_active(subject_id).id)
    assert node.state is NodeState.MASTERED


# --------------------------------------------------------------------------- #
# 未通过：重讲与三选一
# --------------------------------------------------------------------------- #


async def test_failure_reteaches_and_asks_again(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [
        ScriptedResponse(events=text_events("讲解。")),
        ScriptedResponse(events=text_events("换个角度再讲一遍。")),
    ]
    harness.client.blocking_script = [json_response(verdict_json("partial"))]
    await teaching.start(subject_id)

    turn = await teaching.answer(subject_id, "不完整的回答")

    assert turn.ok
    assert turn.phase == "reteach"
    assert turn.decision_required is False
    assert "换个角度再讲一遍。" in turn.text
    assert "节点 0 的检查点 0？" in turn.text  # 重讲后重新提问
    assert turn.attempts == 1

    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.has_pending_question is True
    assert snapshot.needs_user_decision is False


async def test_retry_exhaustion_requires_user_decision(context: AppContext) -> None:
    """FR-408 的核心：超过上限后必须停下等用户选择，绝不自动前进或放弃。"""
    harness = make_harness(context)
    harness.settings = AppSettings(node_max_retries=1)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [
        ScriptedResponse(events=text_events("讲解。")),
        ScriptedResponse(events=text_events("重讲。")),
    ]
    harness.client.blocking_script = [
        json_response(verdict_json("fail")),
        json_response(verdict_json("fail")),
    ]
    await teaching.start(subject_id)

    first = await teaching.answer(subject_id, "错答一")
    assert first.phase == "reteach"

    second = await teaching.answer(subject_id, "错答二")

    assert second.phase == "decision"
    assert second.decision_required is True
    assert second.attempts == 2

    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.needs_user_decision is True
    assert snapshot.has_pending_question is False
    # 节点没有被标为完成
    node = harness.repos.paths.first_node(harness.repos.paths.get_active(subject_id).id)
    assert node.state is NodeState.NOT_STARTED


async def test_answer_rejected_while_awaiting_decision(context: AppContext) -> None:
    """等三选一时提交作答必须被拒绝，否则用户的选项会被顶掉。"""
    harness = make_harness(context)
    harness.settings = AppSettings(node_max_retries=0)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解。"))]
    harness.client.blocking_script = [json_response(verdict_json("fail"))]
    await teaching.start(subject_id)
    await teaching.answer(subject_id, "错答")

    with pytest.raises(AnswerNotAllowedError) as exc:
        await teaching.answer(subject_id, "再来一次")

    assert exc.value.reason == "awaiting_decision"


async def test_judgement_failure_does_not_consume_attempt(context: AppContext) -> None:
    """技术故障不能消耗尝试次数，也不能清掉待回答问题。"""
    harness = make_harness(context)
    harness.settings = AppSettings(node_max_retries=1)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解。"))]
    harness.client.blocking_script = [json_response("坏数据"), json_response("坏数据")]
    await teaching.start(subject_id)
    before = harness.state.snapshot(subject_id).node_attempts

    turn = await teaching.answer(subject_id, "我的回答")

    assert turn.ok is False
    assert turn.retryable is True
    assert harness.state.snapshot(subject_id).node_attempts == before
    assert harness.state.snapshot(subject_id).has_pending_question is True
    assert harness.repos.answers.count_for_node(turn.node.id) == 0  # type: ignore[union-attr]


async def test_retry_exhaustion_respects_configured_limit(context: AppContext) -> None:
    harness = make_harness(context)
    harness.settings = AppSettings(node_max_retries=2)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [
        ScriptedResponse(events=text_events("讲解。")),
        ScriptedResponse(events=text_events("重讲一。")),
        ScriptedResponse(events=text_events("重讲二。")),
    ]
    harness.client.blocking_script = [json_response(verdict_json("fail")) for _ in range(3)]
    await teaching.start(subject_id)

    assert (await teaching.answer(subject_id, "错一")).phase == "reteach"
    assert (await teaching.answer(subject_id, "错二")).phase == "reteach"
    assert (await teaching.answer(subject_id, "错三")).phase == "decision"


# --------------------------------------------------------------------------- #
# 三选一
# --------------------------------------------------------------------------- #


async def drive_to_decision(harness: LearningHarness, teaching: TeachingService, subject_id: str) -> None:
    harness.client.stream_script.extend(
        [
            ScriptedResponse(events=text_events("重讲内容。")),
            ScriptedResponse(events=text_events("参考答案与简化讲解。")),
        ]
    )
    harness.client.blocking_script.extend(
        [json_response(verdict_json("fail")), json_response(verdict_json("fail"))]
    )
    await teaching.answer(subject_id, "错一")
    await teaching.answer(subject_id, "错二")


async def test_decide_reteach_keeps_node_and_asks_again(context: AppContext) -> None:
    harness = make_harness(context)
    harness.settings = AppSettings(node_max_retries=0)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [
        ScriptedResponse(events=text_events("讲解。")),
        ScriptedResponse(events=text_events("按更基础的方向重讲。")),
    ]
    harness.client.blocking_script = [json_response(verdict_json("fail"))]
    await teaching.start(subject_id)
    await teaching.answer(subject_id, "错答")
    assert harness.state.snapshot(subject_id).needs_user_decision is True

    turn = await teaching.decide(subject_id, NodeDecision.RETEACH, direction="simpler")

    assert turn.ok
    assert turn.phase == "reteach"
    assert "按更基础的方向重讲。" in turn.text
    assert "检查点 0？" in turn.text

    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.needs_user_decision is False
    assert snapshot.has_pending_question is True

    # direction 提示进了提示词
    prompt = str(harness.client.calls[-1].messages[0]["content"])
    assert "更基础" in prompt


async def test_decide_show_answer_marks_partial_and_weak_point(context: AppContext) -> None:
    harness = make_harness(context)
    harness.settings = AppSettings(node_max_retries=0)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [
        ScriptedResponse(events=text_events("讲解。")),
        ScriptedResponse(events=text_events("参考答案与简化讲解。")),
    ]
    harness.client.blocking_script = [json_response(verdict_json("fail"))]
    await teaching.start(subject_id)
    await teaching.answer(subject_id, "错答")

    turn = await teaching.decide(subject_id, NodeDecision.SHOW_ANSWER)

    assert turn.ok
    assert turn.phase == "reference"

    node = harness.repos.paths.first_node(harness.repos.paths.get_active(subject_id).id)
    assert node.state is NodeState.PARTIAL

    weak = harness.repos.weak_points.list_open(subject_id)
    assert len(weak) == 1
    assert weak[0].reason is WeakPointReason.SHOW_ANSWER

    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.needs_user_decision is False
    assert snapshot.has_pending_question is False


async def test_decide_advance_marks_partial_and_moves_on(context: AppContext) -> None:
    harness = make_harness(context)
    harness.settings = AppSettings(node_max_retries=0)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解。"))]
    harness.client.blocking_script = [json_response(verdict_json("fail"))]
    await teaching.start(subject_id)
    await teaching.answer(subject_id, "错答")

    turn = await teaching.decide(subject_id, NodeDecision.ADVANCE)

    assert turn.ok
    assert turn.phase == "advanced"
    assert turn.node is not None and turn.node.title == "节点 1"

    first = harness.repos.paths.first_node(harness.repos.paths.get_active(subject_id).id)
    assert first.state is NodeState.PARTIAL
    weak = harness.repos.weak_points.list_open(subject_id)
    assert weak[0].reason is WeakPointReason.ADVANCED


async def test_decide_advance_on_last_node_completes_subject(context: AppContext) -> None:
    harness = make_harness(context)
    harness.settings = AppSettings(node_max_retries=0)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=1)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解。"))]
    harness.client.blocking_script = [json_response(verdict_json("fail"))]
    await teaching.start(subject_id)
    await teaching.answer(subject_id, "错答")

    turn = await teaching.decide(subject_id, NodeDecision.ADVANCE)

    assert turn.completed is True
    assert harness.state.snapshot(subject_id).state is SubjectState.COMPLETED


async def test_decide_without_pending_decision_is_rejected(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)

    with pytest.raises(AnswerNotAllowedError) as exc:
        await teaching.decide(subject_id, NodeDecision.ADVANCE)

    assert exc.value.reason == "decision_not_expected"


# --------------------------------------------------------------------------- #
# 跳过与暂停
# --------------------------------------------------------------------------- #


async def test_skip_node_marks_skipped_and_records_weak_point(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解。"))]
    await teaching.start(subject_id)

    turn = teaching.skip_node(subject_id)

    assert turn.ok
    assert turn.phase == "skipped"
    node = turn.node
    assert node is not None and node.state is NodeState.SKIPPED
    weak = harness.repos.weak_points.list_open(subject_id)
    assert weak[0].reason is WeakPointReason.SKIPPED
    assert harness.state.snapshot(subject_id).has_pending_question is False


async def test_skip_last_node_completes_subject(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=1)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解。"))]
    await teaching.start(subject_id)

    turn = teaching.skip_node(subject_id)

    assert turn.completed is True
    assert harness.state.snapshot(subject_id).state is SubjectState.COMPLETED


async def test_pause_and_resume(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)

    assert teaching.pause(subject_id).state is SubjectState.PAUSED

    with pytest.raises(AnswerNotAllowedError) as exc:
        await teaching.start(subject_id)
    assert exc.value.reason == "paused"

    assert teaching.resume(subject_id).state is SubjectState.TEACHING


# --------------------------------------------------------------------------- #
# 视图
# --------------------------------------------------------------------------- #


async def test_view_reports_node_and_checkpoint(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解。"))]
    await teaching.start(subject_id)

    view = teaching.view(subject_id)

    assert view.node is not None and view.node.title == "节点 0"
    assert view.checkpoint is not None
    assert view.checkpoint.question == "节点 0 的检查点 0？"
    assert view.can_answer is True
    assert view.needs_decision is False
    assert view.node_index == 1
    assert view.node_total == 2


async def test_empty_answer_is_rejected(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解。"))]
    await teaching.start(subject_id)

    turn = await teaching.answer(subject_id, "   ")

    assert turn.ok is False
    assert turn.error_category == "validation"


async def test_answer_without_pending_question_is_rejected(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)

    with pytest.raises(AnswerNotAllowedError) as exc:
        await teaching.answer(subject_id, "还没提问就回答")

    assert exc.value.reason == "no_pending_question"


async def test_teaching_memory_is_separate_from_assessment(context: AppContext) -> None:
    """FR-209：教学记忆里不应出现测评原始对话。"""
    harness = make_harness(context)
    assessments, _, teaching, subject_id = await enter_teaching_after_assessment(harness, node_count=3)
    harness.client.stream_script = [ScriptedResponse(events=text_events("讲解。"))]
    await teaching.start(subject_id)

    from app.contracts.schemas import ConversationKind

    teach_id = harness.ctx.memories.ensure_conversation(ConversationKind.TEACH, subject_id=subject_id)
    teach_text = "\n".join(
        str(message.get("content", ""))
        for message in harness.ctx.memories.load(teach_id).outbound_messages()
    )

    assert "回答一" not in teach_text  # 测评作答只存在于测评记忆
    assert assessments.levels(subject_id)  # 画像本身是有的


async def test_profile_level_override_does_not_leak_into_teaching(context: AppContext) -> None:
    """画像修正（FR-207）只应影响后续路径生成，不应改写教学记忆。"""
    harness = make_harness(context)
    assessments, _, _, subject_id = await enter_teaching_after_assessment(harness, node_count=3)

    assert assessments.override_level(subject_id, "定义", MasteryLevel.BASIC) is True
    levels = assessments.levels(subject_id)
    assert any(level.topic == "定义" and level.user_overridden for level in levels)
