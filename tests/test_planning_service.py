"""学习路径规划服务测试（p2-planning，FR-301~FR-309）。"""

import json

import pytest

from app.contracts.schemas import NodeState, SubjectState
from app.infra.settings import AppSettings
from app.main_context import AppContext
from app.services.learning.assessment import AssessmentService
from app.services.learning.planning import (
    NO_ASSESSMENT_NOTICE,
    PlanningError,
    PlanningService,
)
from tests.learning_harness import LearningHarness, make_harness
from tests.stubs import ScriptedResponse, json_response, text_events

TOPICS_JSON = json.dumps(
    {
        "topics": ["定义与直觉", "关键公式"],
        "first_question": "先说说这个定理想解决什么问题？",
    },
    ensure_ascii=False,
)

PROFILE_JSON = json.dumps(
    {
        "levels": [
            {"topic": "定义与直觉", "level": "proficient", "evidence": "解释准确"},
            {"topic": "关键公式", "level": "none", "evidence": "表示不了解"},
        ],
        "summary": "直觉已经具备，公式需要从头学。",
        "suggested_start_node": "从公式的直观含义开始。",
    },
    ensure_ascii=False,
)


def node_draft(title: str, *, quick: bool = False, prerequisites: list[str] | None = None) -> dict:
    return {
        "title": title,
        "objective": f"能解释 {title}",
        "prerequisites": prerequisites or [],
        "key_points": ["要点一", "要点二", "要点三"],
        "checkpoints": [
            {
                "question": f"{title} 的检查点问题？",
                "expected_points": ["判定要点"],
                "reference_answer": "参考答案",
            }
        ],
        "is_quick_review": quick,
    }


def path_json(count: int, *, quick_first: bool = False) -> str:
    return json.dumps(
        {
            "rationale": "根据画像：直觉已熟练压缩为复习节点，公式拆成递进节点。",
            "nodes": [
                node_draft(f"节点 {index}", quick=(quick_first and index == 0))
                for index in range(count)
            ],
        },
        ensure_ascii=False,
    )


def build_services(harness: LearningHarness) -> tuple[AssessmentService, PlanningService]:
    assessments = AssessmentService(harness.repos, harness.state, harness.sessions, lambda: harness.settings)
    planning = PlanningService(
        harness.repos,
        harness.state,
        harness.sessions,
        assessments,
        lambda: harness.settings,
        harness.policy,
    )
    return assessments, planning


async def complete_assessment(harness: LearningHarness, service: AssessmentService, subject_id: str) -> None:
    """把测评跑完，让状态进入 path_review。"""
    await service.start(subject_id)
    await service.answer(subject_id, "回答一")
    await service.answer(subject_id, "回答二")


def make_planned(
    context: AppContext,
    *,
    nodes: int = 3,
    quick_first: bool = False,
    path_response: ScriptedResponse | None = None,
    extra_blocking: list[ScriptedResponse] | None = None,
    settings: AppSettings | None = None,
):
    """搭好"测评已完成、等待生成路径"的场景。

    ``path_response`` 用于替换默认的路径响应（例如模拟结构化输出失败）。
    """
    blocking = [json_response(TOPICS_JSON), json_response(PROFILE_JSON)]
    blocking.append(
        path_response
        if path_response is not None
        else json_response(path_json(nodes, quick_first=quick_first))
    )
    if extra_blocking:
        blocking.extend(extra_blocking)

    harness = make_harness(
        context,
        blocking=blocking,
        stream=[
            ScriptedResponse(events=text_events("问二")),
            ScriptedResponse(events=text_events("问三")),
        ],
        settings=settings,
    )
    assessments, planning = build_services(harness)
    return harness, assessments, planning


# --------------------------------------------------------------------------- #
# 生成
# --------------------------------------------------------------------------- #


async def test_generate_creates_path_with_nodes_and_checkpoints(context: AppContext) -> None:
    harness, assessments, planning = make_planned(context, nodes=3)
    subject_id = harness.new_subject("贝叶斯定理")
    await complete_assessment(harness, assessments, subject_id)

    result = await planning.generate(subject_id)

    assert result.ok
    assert result.path is not None
    assert result.path.node_count == 3
    assert result.path.version == 1
    assert result.path.is_active is True
    assert "根据画像" in result.rationale
    assert result.notice == ""

    first = result.path.ordered_nodes()[0]
    assert first.title == "节点 0"
    assert first.objective == "能解释 节点 0"
    assert first.key_points == ("要点一", "要点二", "要点三")
    assert len(first.checkpoints) == 1
    assert first.checkpoints[0].question == "节点 0 的检查点问题？"
    assert first.state is NodeState.NOT_STARTED

    # 路径生成不改变状态（确认时才进 teaching）
    assert harness.state.snapshot(subject_id).state is SubjectState.PATH_REVIEW
    # 记录的模型名可用于回溯
    assert result.path.model == "deepseek-flash"


async def test_generate_marks_quick_review_nodes(context: AppContext) -> None:
    """FR-303：画像中已熟练的内容应压缩为"快速复习"节点。"""
    harness, assessments, planning = make_planned(context, nodes=3, quick_first=True)
    subject_id = harness.new_subject()
    await complete_assessment(harness, assessments, subject_id)

    result = await planning.generate(subject_id)

    assert result.ok
    nodes = result.path.ordered_nodes()  # type: ignore[union-attr]
    assert nodes[0].is_quick_review is True
    assert nodes[1].is_quick_review is False


async def test_generate_passes_profile_into_prompt(context: AppContext) -> None:
    """FR-301：路径必须依据画像；画像内容要真的进到提示词里。"""
    harness, assessments, planning = make_planned(context)
    subject_id = harness.new_subject("贝叶斯定理", goal="通过面试", depth="深入")
    await complete_assessment(harness, assessments, subject_id)

    await planning.generate(subject_id)

    prompt = str(harness.client.calls[-1].messages[0]["content"])
    assert "贝叶斯定理" in prompt
    assert "通过面试" in prompt
    assert "深入" in prompt
    assert "定义与直觉" in prompt
    assert "已熟练" in prompt  # proficient → 中文标签
    assert "未掌握" in prompt
    assert "直觉已经具备" in prompt  # 总体评价
    assert "从公式的直观含义开始" in prompt  # 建议起点
    assert "json" in prompt.lower()


async def test_generate_without_assessment_notes_zero_basis(context: AppContext) -> None:
    """FR-205：跳过测评后生成路径要按零基础处理，并向用户说明。"""
    harness = make_harness(
        context,
        blocking=[json_response(path_json(3))],
    )
    assessments, planning = build_services(harness)
    subject_id = harness.new_subject()

    assessments.skip(subject_id)
    result = await planning.generate(subject_id)

    assert result.ok
    assert "未进行知识基础确认" in result.notice

    prompt = str(harness.client.calls[-1].messages[0]["content"])
    assert NO_ASSESSMENT_NOTICE in prompt


async def test_generate_rejects_node_count_outside_configured_range(context: AppContext) -> None:
    """配置的区间是硬约束：不满足时先带约束重试一次，仍不满足则明确报错。"""
    harness, assessments, planning = make_planned(
        context,
        nodes=3,
        extra_blocking=[json_response(path_json(3))],
        settings=AppSettings(path_min_nodes=5, path_max_nodes=8),
    )
    subject_id = harness.new_subject()
    await complete_assessment(harness, assessments, subject_id)

    result = await planning.generate(subject_id)

    assert result.ok is False
    assert result.error_category == "node_count"
    assert result.retryable is True
    assert "5~8" in result.error_message
    # 第二次调用时提示里带上了纠正约束
    second_prompt = str(harness.client.calls[-1].messages[0]["content"])
    assert "必须改为 5~8 个节点" in second_prompt
    assert planning.current_path(subject_id) is None


async def test_generate_retries_then_succeeds(context: AppContext) -> None:
    harness, assessments, planning = make_planned(
        context,
        nodes=3,
        extra_blocking=[json_response(path_json(6))],
        settings=AppSettings(path_min_nodes=5, path_max_nodes=8),
    )
    subject_id = harness.new_subject()
    await complete_assessment(harness, assessments, subject_id)

    result = await planning.generate(subject_id)

    assert result.ok
    assert result.path.node_count == 6  # type: ignore[union-attr]


async def test_generate_failure_is_retryable_and_keeps_state(context: AppContext) -> None:
    harness, assessments, planning = make_planned(
        context,
        path_response=json_response("坏数据"),
        extra_blocking=[json_response("坏数据")],
    )
    subject_id = harness.new_subject()
    await complete_assessment(harness, assessments, subject_id)

    result = await planning.generate(subject_id)

    assert result.ok is False
    assert result.error_category == "structured_output"
    assert result.retryable is True
    assert harness.state.snapshot(subject_id).state is SubjectState.PATH_REVIEW
    assert planning.current_path(subject_id) is None


async def test_regenerate_deactivates_previous_version(context: AppContext) -> None:
    """FR-305/FR-307：重新生成保留旧版本，但只有一条活动路径。"""
    harness, assessments, planning = make_planned(
        context,
        nodes=3,
        extra_blocking=[json_response(path_json(4))],
    )
    subject_id = harness.new_subject()
    await complete_assessment(harness, assessments, subject_id)

    first = await planning.generate(subject_id)
    second = await planning.generate(subject_id)

    assert first.path is not None and second.path is not None
    assert second.path.version == 2
    assert planning.current_path(subject_id).id == second.path.id  # type: ignore[union-attr]

    versions = planning.versions(subject_id)
    assert len(versions) == 2
    assert versions[0][1] == 2


async def test_generate_from_draft_is_rejected(context: AppContext) -> None:
    harness, _, planning = make_planned(context)
    subject_id = harness.new_subject()

    with pytest.raises(PlanningError, match="知识基础确认"):
        await planning.generate(subject_id)


async def test_generate_during_assessment_is_rejected(context: AppContext) -> None:
    harness, assessments, planning = make_planned(context)
    subject_id = harness.new_subject()
    await assessments.start(subject_id)
    assert harness.state.snapshot(subject_id).state is SubjectState.ASSESSING

    with pytest.raises(PlanningError, match="还没结束"):
        await planning.generate(subject_id)


async def test_generate_after_completion_is_rejected(context: AppContext) -> None:
    harness, assessments, planning = make_planned(context)
    subject_id = harness.new_subject()
    await complete_assessment(harness, assessments, subject_id)
    await planning.generate(subject_id)
    planning.confirm(subject_id)
    harness.state.transition(subject_id, SubjectState.COMPLETED)

    with pytest.raises(PlanningError, match="已完成"):
        await planning.generate(subject_id)


async def test_regenerate_from_teaching_returns_to_path_review(context: AppContext) -> None:
    """教学态重新规划：先回 path_review，再生成（FR-305）。"""
    harness, assessments, planning = make_planned(
        context,
        nodes=3,
        extra_blocking=[json_response(path_json(4))],
    )
    subject_id = harness.new_subject()
    await complete_assessment(harness, assessments, subject_id)
    await planning.generate(subject_id)
    planning.confirm(subject_id)
    assert harness.state.snapshot(subject_id).state is SubjectState.TEACHING

    result = await planning.generate(subject_id)

    assert result.ok
    assert harness.state.snapshot(subject_id).state is SubjectState.PATH_REVIEW


# --------------------------------------------------------------------------- #
# 确认门禁
# --------------------------------------------------------------------------- #


async def test_confirm_moves_to_teaching(context: AppContext) -> None:
    harness, assessments, planning = make_planned(context)
    subject_id = harness.new_subject()
    await complete_assessment(harness, assessments, subject_id)
    await planning.generate(subject_id)

    confirmed = planning.confirm(subject_id)

    assert confirmed.is_active is True
    assert harness.state.snapshot(subject_id).state is SubjectState.TEACHING
    assert harness.repos.subjects.get(subject_id).status is SubjectState.TEACHING  # type: ignore[union-attr]


async def test_confirm_without_path_raises(context: AppContext) -> None:
    """FR-306 的门禁：没有路径就不能进入教学。"""
    harness, assessments, planning = make_planned(context)
    subject_id = harness.new_subject()
    assessments.skip(subject_id)

    with pytest.raises(PlanningError, match="还没有学习路径"):
        planning.confirm(subject_id)


# --------------------------------------------------------------------------- #
# 视图
# --------------------------------------------------------------------------- #


async def test_view_reports_current_index_and_completion(context: AppContext) -> None:
    harness, assessments, planning = make_planned(context, nodes=3)
    subject_id = harness.new_subject()
    await complete_assessment(harness, assessments, subject_id)
    result = await planning.generate(subject_id)
    assert result.path is not None
    nodes = result.path.ordered_nodes()

    view = planning.view(subject_id)
    assert view is not None
    assert view.current_index == 0
    assert view.completed is False
    assert len(view.profile_levels) == 2  # 画像随路径一起返回，便于展示"为什么这样安排"

    harness.repos.paths.update_node(nodes[0].id, state=NodeState.MASTERED)
    harness.repos.paths.update_node(nodes[1].id, state=NodeState.PARTIAL)
    harness.repos.paths.update_node(nodes[2].id, state=NodeState.SKIPPED)

    done_view = planning.view(subject_id)
    assert done_view is not None
    assert done_view.completed is True
    assert done_view.current_index == 3


def test_view_returns_none_without_path(context: AppContext) -> None:
    harness, _, planning = make_planned(context)
    subject_id = harness.new_subject()

    assert planning.view(subject_id) is None
    assert planning.current_path(subject_id) is None


def test_all_nodes_done_requires_non_empty_path(context: AppContext) -> None:
    harness, _, planning = make_planned(context)
    subject_id = harness.new_subject()
    # 空路径（理论上不会出现，但判定要安全）
    from app.services.learning.entities import LearningPath

    empty = LearningPath(
        id="p",
        subject_id=subject_id,
        version=1,
        rationale="r",
        model=None,
        is_active=True,
        created_at=harness.repos.subjects.get(subject_id).created_at,  # type: ignore[union-attr]
        nodes=(),
    )
    assert planning.all_nodes_done(empty) is False
