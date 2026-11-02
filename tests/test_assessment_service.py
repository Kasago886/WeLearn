"""测评服务测试（p2-assessment，FR-201~FR-210）。"""

import json

import pytest

from app.contracts.schemas import (
    AssessmentStatus,
    ConversationKind,
    MasteryLevel,
    SubjectState,
)
from app.main_context import AppContext
from app.services.learning.assessment import (
    AssessmentError,
    AssessmentService,
    build_transcript,
)
from app.services.llm.memory import ROLE_KEY, visible_messages
from tests.learning_harness import make_harness
from tests.stubs import ScriptedResponse, json_response, text_events

TOPICS_JSON = json.dumps(
    {
        "topics": ["定义与直觉", "关键公式", "适用条件", "常见误区"],
        "first_question": "先用你自己的话说说，这个定理想解决的问题是什么？",
    },
    ensure_ascii=False,
)

PROFILE_JSON = json.dumps(
    {
        "levels": [
            {"topic": "定义与直觉", "level": "basic", "evidence": "能说出大意"},
            {"topic": "关键公式", "level": "none", "evidence": "表示不了解"},
            {"topic": "适用条件", "level": "none", "evidence": "未作答"},
        ],
        "summary": "你已具备基本直觉，公式部分需要从头学。",
        "suggested_start_node": "建议先从贝叶斯公式的直观含义开始。",
    },
    ensure_ascii=False,
)


def build_service(harness) -> AssessmentService:
    return AssessmentService(harness.repos, harness.state, harness.sessions, lambda: harness.settings)


def make_ready(context: AppContext, *, streams: list[str] | None = None, profile: str = PROFILE_JSON):
    """搭好一个"已推导出子领域"的测评场景。"""
    stream_script = [ScriptedResponse(events=text_events(text)) for text in (streams or [])]
    harness = make_harness(
        context,
        blocking=[json_response(TOPICS_JSON), json_response(profile)],
        stream=stream_script,
    )
    return harness, build_service(harness)


# --------------------------------------------------------------------------- #
# 开始测评
# --------------------------------------------------------------------------- #


async def test_start_derives_topics_and_asks_first_question(context: AppContext) -> None:
    harness, service = make_ready(context)
    subject_id = harness.new_subject("贝叶斯定理", goal="通过面试")

    turn = await service.start(subject_id)

    assert turn.ok
    assert turn.text == "先用你自己的话说说，这个定理想解决的问题是什么？"
    assert turn.topics == ("定义与直觉", "关键公式", "适用条件", "常见误区")
    assert turn.current_topic == "定义与直觉"
    assert turn.total_turns == 4
    assert turn.finished is False

    # 状态机进入 assessing
    assert harness.state.snapshot(subject_id).state is SubjectState.ASSESSING

    # 测评记录落库
    assessment = harness.repos.assessments.get_running(subject_id)
    assert assessment is not None
    assert assessment.topics == turn.topics
    assert assessment.max_turns == 4
    assert assessment.turn_count == 0

    # 首问写进测评记忆，后续轮次才有上下文
    conversation_id = harness.ctx.memories.conversation_id_for(subject_id, ConversationKind.ASSESS)
    assert conversation_id is not None
    messages = harness.ctx.memories.load(conversation_id).outbound_messages()
    assert messages[0][ROLE_KEY] == "system"
    assert messages[-1]["content"] == turn.text


async def test_start_respects_assess_max_turns_setting(context: AppContext) -> None:
    """FR-204：轮次上限可配；子领域数量超过上限时截断。"""
    from app.infra.settings import AppSettings

    harness, service = make_ready(context)
    harness.settings = AppSettings(assess_max_turns=3)
    subject_id = harness.new_subject()

    turn = await service.start(subject_id)

    assert len(turn.topics) == 3
    assert turn.topics == ("定义与直觉", "关键公式", "适用条件")


async def test_start_blocks_when_assessment_running(context: AppContext) -> None:
    harness, service = make_ready(context)
    subject_id = harness.new_subject()
    await service.start(subject_id)

    with pytest.raises(AssessmentError, match="进行中"):
        await service.start(subject_id)


async def test_start_failure_returns_to_draft(context: AppContext) -> None:
    """结构化推导失败时必须能重来，不能把用户卡在 assessing。"""
    harness = make_harness(
        context,
        blocking=[json_response("不是 JSON"), json_response("还是不是")],
    )
    service = build_service(harness)
    subject_id = harness.new_subject()

    turn = await service.start(subject_id)

    assert turn.ok is False
    assert turn.retryable is True
    assert harness.state.snapshot(subject_id).state is SubjectState.DRAFT
    assert harness.repos.assessments.get_running(subject_id) is None


async def test_start_deduplicates_topics(context: AppContext) -> None:
    payload = json.dumps(
        {"topics": ["定义", "定义", " ", "公式"], "first_question": "问题？"}, ensure_ascii=False
    )
    harness = make_harness(context, blocking=[json_response(payload)])
    service = build_service(harness)
    subject_id = harness.new_subject()

    turn = await service.start(subject_id)

    assert turn.topics == ("定义", "公式")


# --------------------------------------------------------------------------- #
# 作答推进
# --------------------------------------------------------------------------- #


async def test_answer_advances_to_next_topic(context: AppContext) -> None:
    harness, service = make_ready(context, streams=["第二个问题：公式长什么样？"])
    subject_id = harness.new_subject()
    await service.start(subject_id)

    turn = await service.answer(subject_id, "它解决的是怎么用新证据更新判断。")

    assert turn.ok
    assert turn.finished is False
    assert turn.current_topic == "关键公式"
    assert turn.turn_number == 1
    assert turn.total_turns == 4
    assert "第二个问题" in turn.text

    # 系统提示必须已经改写成"本轮聚焦关键公式"
    call = harness.client.calls[-1]
    system_content = str(call.messages[0]["content"])
    assert "关键公式" in system_content
    assert "第 2 轮" in system_content


async def test_answer_walks_through_all_topics_then_builds_profile(context: AppContext) -> None:
    streams = ["问二", "问三", "问四"]
    harness, service = make_ready(context, streams=streams)
    subject_id = harness.new_subject()
    await service.start(subject_id)

    turns = [await service.answer(subject_id, f"回答 {index}") for index in range(1, 4)]
    assert all(turn.finished is False for turn in turns)

    # 第四次作答时已覆盖全部子领域 → 产出画像
    final = await service.answer(subject_id, "回答 4")

    assert final.finished is True
    assert final.profile is not None
    assert final.text == "你已具备基本直觉，公式部分需要从头学。"

    levels = service.levels(subject_id)
    assert [level.topic for level in levels] == ["定义与直觉", "关键公式", "适用条件"]
    assert levels[0].level is MasteryLevel.BASIC
    assert levels[1].level is MasteryLevel.NONE

    assessment = harness.repos.assessments.get_latest(subject_id)
    assert assessment is not None
    assert assessment.status is AssessmentStatus.COMPLETED
    assert assessment.summary == "你已具备基本直觉，公式部分需要从头学。"
    assert assessment.suggested_start_node == "建议先从贝叶斯公式的直观含义开始。"
    assert assessment.completed_at is not None

    assert harness.state.snapshot(subject_id).state is SubjectState.PATH_REVIEW


async def test_answer_records_user_message_in_memory(context: AppContext) -> None:
    harness, service = make_ready(context, streams=["下一个问题"])
    subject_id = harness.new_subject()
    await service.start(subject_id)

    await service.answer(subject_id, "我的独特回答内容")

    conversation_id = harness.ctx.memories.conversation_id_for(subject_id, ConversationKind.ASSESS)
    contents = [message["content"] for message in harness.ctx.memories.load(conversation_id).outbound_messages()]
    assert "我的独特回答内容" in contents


async def test_answer_rejects_empty_input(context: AppContext) -> None:
    harness, service = make_ready(context, streams=["x"])
    subject_id = harness.new_subject()
    await service.start(subject_id)

    turn = await service.answer(subject_id, "   ")

    assert turn.ok is False
    assert turn.error_category == "validation"
    assert turn.retryable is False
    assert harness.client.stream_calls == 0


async def test_answer_without_running_assessment_raises(context: AppContext) -> None:
    harness, service = make_ready(context)
    subject_id = harness.new_subject()

    with pytest.raises(AssessmentError, match="没有进行中的测评"):
        await service.answer(subject_id, "随便答")


async def test_answer_propagates_llm_failure(context: AppContext) -> None:
    from app.infra.llm.errors import LlmError

    harness = make_harness(
        context,
        blocking=[json_response(TOPICS_JSON)],
        stream=[ScriptedResponse(error=LlmError.from_status(503))],
    )
    service = build_service(harness)
    subject_id = harness.new_subject()
    await service.start(subject_id)

    turn = await service.answer(subject_id, "回答")

    assert turn.ok is False
    assert turn.error_category == "unavailable"
    assert turn.retryable is True
    # 测评仍在进行中，用户可以重试
    assert harness.repos.assessments.get_running(subject_id) is not None


async def test_failed_turn_leaves_the_question_askable_again(context: AppContext) -> None:
    """失败的一轮必须整轮回退，否则用户会卡住。

    待答问题是由"对话里最后一条 assistant 消息"推出来的。若是把用户的作答留下来，
    最后一条就变成 user、待答问题为空，提交按钮点不动——用户既看不到问题也重试不了。
    真实故障（2026-10-13）正是这样：模型返回空正文，用户提交回答后界面毫无反应。
    """
    from app.infra.llm.errors import LlmError

    harness = make_harness(
        context,
        blocking=[json_response(TOPICS_JSON)],
        stream=[
            ScriptedResponse(error=LlmError.from_status(503)),  # 第一次作答失败
            ScriptedResponse(events=text_events("第二个方面的问题？")),  # 重试成功
        ],
    )
    service = build_service(harness)
    subject_id = harness.new_subject()
    started = await service.start(subject_id)

    failed = await service.answer(subject_id, "第一次作答（会失败）")
    assert failed.ok is False

    # 回退到位：失败的那条作答没有留在对话里，最后一条仍是问题本身
    conversation_id = harness.ctx.memories.conversation_id_for(subject_id, ConversationKind.ASSESS)
    transcript = visible_messages(harness.ctx.memories.load(conversation_id))
    assert [item["content"] for item in transcript] == [started.text]
    # 轮次没有推进：不该因为一次技术故障占掉一个考察方面
    running = harness.repos.assessments.get_running(subject_id)
    assert running is not None and running.turn_count == 0

    # 再提交一次就能继续，且这次成功
    retried = await service.answer(subject_id, "第二次作答")
    assert retried.ok is True
    assert retried.turn_number == 1
    assert retried.current_topic == "关键公式"


async def test_profile_failure_keeps_assessment_running(context: AppContext) -> None:
    """画像失败不能让用户从头再测评一遍。"""
    harness = make_harness(
        context,
        blocking=[json_response(TOPICS_JSON), json_response("坏数据"), json_response("坏数据")],
        stream=[
            ScriptedResponse(events=text_events("问二")),
            ScriptedResponse(events=text_events("问三")),
            ScriptedResponse(events=text_events("问四")),
        ],
    )
    service = build_service(harness)
    subject_id = harness.new_subject()
    await service.start(subject_id)
    # 4 个子领域：前 3 次作答各换来一个问题，第 4 次触发画像
    for index in range(1, 4):
        assert (await service.answer(subject_id, f"回答 {index}")).finished is False

    turn = await service.answer(subject_id, "回答 4")

    assert turn.ok is False
    assert turn.retryable is True
    running = harness.repos.assessments.get_running(subject_id)
    assert running is not None
    # 失败的一轮不该占用一个考察方面：否则界面会显示"已考察 4/4"但其实还没完成
    assert running.turn_count == 3
    # 状态没有推进到 path_review
    assert harness.state.snapshot(subject_id).state is SubjectState.ASSESSING
    # 用户再提交一次即可重试画像（画像调用有重试，因此这里再补两次坏响应）
    harness.client.blocking_script.extend([json_response("坏数据"), json_response("坏数据")])
    assert (await service.answer(subject_id, "回答 5")).ok is False
    assert harness.state.snapshot(subject_id).state is SubjectState.ASSESSING


# --------------------------------------------------------------------------- #
# 跳过与修正
# --------------------------------------------------------------------------- #


def test_skip_marks_skipped_and_moves_to_path_review(context: AppContext) -> None:
    harness = make_harness(context)
    service = build_service(harness)
    subject_id = harness.new_subject()

    turn = service.skip(subject_id)

    assert turn.ok
    assert turn.skipped is True
    assert turn.finished is True
    assert harness.state.snapshot(subject_id).state is SubjectState.PATH_REVIEW

    assessment = harness.repos.assessments.get_latest(subject_id)
    assert assessment is not None
    assert assessment.skipped is True
    assert assessment.status is AssessmentStatus.SKIPPED
    # 未测评 → 没有画像，路径阶段据此提示"按零基础处理"
    assert service.levels(subject_id) == []


async def test_skip_during_running_assessment(context: AppContext) -> None:
    harness, service = make_ready(context)
    subject_id = harness.new_subject()
    await service.start(subject_id)

    turn = service.skip(subject_id)

    assert turn.skipped is True
    assert harness.repos.assessments.get_running(subject_id) is None


async def test_override_level_updates_profile(context: AppContext) -> None:
    """FR-207：允许人工修正某个子领域的水平，并留下"已修正"标记。"""
    harness, service = make_ready(context, streams=["问二", "问三", "问四"])
    subject_id = harness.new_subject()

    await service.start(subject_id)
    for index in range(1, 5):
        await service.answer(subject_id, f"回答 {index}")

    assert service.override_level(subject_id, "关键公式", MasteryLevel.BASIC) is True
    levels = {level.topic: level for level in service.levels(subject_id)}
    assert levels["关键公式"].level is MasteryLevel.BASIC
    assert levels["关键公式"].user_overridden is True

    assert service.override_level(subject_id, "不存在的子领域", MasteryLevel.BASIC) is False


def test_override_level_without_assessment_returns_false(context: AppContext) -> None:
    harness = make_harness(context)
    service = build_service(harness)
    subject_id = harness.new_subject()

    assert service.override_level(subject_id, "任意", MasteryLevel.BASIC) is False


def test_view_returns_none_without_assessment(context: AppContext) -> None:
    harness = make_harness(context)
    service = build_service(harness)
    subject_id = harness.new_subject()

    assert service.view(subject_id) is None
    assert service.levels(subject_id) == []


# --------------------------------------------------------------------------- #
# 记忆隔离与记录整理
# --------------------------------------------------------------------------- #


async def test_assessment_memory_is_separate_from_teaching(context: AppContext) -> None:
    """FR-209：测评与教学必须是两份独立记忆。"""
    harness, service = make_ready(context, streams=["问二"])
    subject_id = harness.new_subject()
    await service.start(subject_id)
    await service.answer(subject_id, "回答")

    assess_id = harness.ctx.memories.conversation_id_for(subject_id, ConversationKind.ASSESS)
    teach_id = harness.ctx.memories.ensure_conversation(ConversationKind.TEACH, subject_id=subject_id)

    assert assess_id != teach_id
    assert harness.ctx.memories.load(teach_id).is_empty is True
    assert harness.ctx.memories.load(assess_id).message_count > 0


def test_build_transcript_keeps_only_dialogue(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id = harness.new_subject()
    session = harness.sessions.open(subject_id, ConversationKind.ASSESS)
    session.memory.add_round().add({"role": "system", "content": "系统提示不该出现"})
    session.memory.add_message_to_last_round({"role": "user", "content": "我的回答"})
    session.memory.add_message_to_last_round({"role": "assistant", "content": "下一个问题"})
    session.memory.add_message_to_last_round({"role": "user", "content": "", "reasoning_content": "思维链"})

    transcript = build_transcript(session)

    assert "系统提示不该出现" not in transcript
    assert "思维链" not in transcript
    assert "学习者：我的回答" in transcript
    assert "诊断者：下一个问题" in transcript


def test_build_transcript_handles_empty_memory(context: AppContext) -> None:
    harness = make_harness(context)
    session = harness.sessions.open(harness.new_subject(), ConversationKind.ASSESS)

    assert build_transcript(session) == "（无对话记录）"


def test_build_transcript_truncates_from_the_end(context: AppContext) -> None:
    harness = make_harness(context)
    session = harness.sessions.open(harness.new_subject(), ConversationKind.ASSESS)
    session.memory.add_round().add({"role": "user", "content": "旧" * 100})
    session.memory.add_message_to_last_round({"role": "user", "content": "最新的回答"})

    transcript = build_transcript(session, limit=20)

    assert len(transcript) == 20
    assert transcript.endswith("最新的回答")
