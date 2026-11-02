"""教学深化测试（T3.2，FR-407、FR-411、FR-413）。

三块能力，各自的核心不变量：

- **分级提示**（FR-407）：级别随连续失败加深，但**不消耗尝试次数**、不改变节点状态，
  也**不换掉检查点问题**——提示是脚手架，不是作答；
- **穿插提问**（FR-411）：回答完回到主线，**绝不改变教学状态**。没有这条约束，
  用户随口问一句就会被当成检查点答案送去判定；消息在存储里标记为 off-topic；
- **重新生成**（FR-413）：按记录的动作重跑，**不撤销已发生的判定**——
  答错了就是答错了，"再来一次"只该换一种说法，不该抹掉记录。
"""

import pytest

from app.contracts.schemas import ConversationKind, NodeDecision, NodeState
from app.infra.settings import AppSettings
from app.main_context import AppContext
from app.services.learning.teaching import (
    ACTION_EXPLAIN,
    ACTION_REFERENCE,
    HINT_LEVEL_LABELS,
    TeachingError,
    hint_action,
    hint_level_for,
)
from app.services.llm.memory import KIND_OFF_TOPIC
from tests.learning_harness import LearningHarness, make_harness
from tests.stubs import ScriptedResponse, json_response, text_events
from tests.test_teaching_service import enter_teaching, verdict_json

CHECKPOINT_QUESTION = "节点 0 的检查点 0？"


def script(
    harness: LearningHarness,
    *,
    stream: list[str] | None = None,
    blocking: list[str] | None = None,
) -> None:
    """设置桩脚本并**重置调用计数**。

    桩是按"第几次调用"取脚本的，不清计数的话第二轮会从 index=1 开始找响应，
    于是"脚本响应不足"——这个坑每个测试文件都要踩一次，所以统一收在这里。
    """
    harness.client.stream_script = [ScriptedResponse(events=text_events(text)) for text in (stream or [])]
    harness.client.blocking_script = [json_response(text) for text in (blocking or [])]
    harness.client.stream_calls = 0
    harness.client.blocking_calls = 0


def setup(harness: LearningHarness, *, node_count: int = 2, retries: int = 2) -> tuple:
    """进入教学并抛出第一个问题。"""
    harness.settings = AppSettings(node_max_retries=retries)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=node_count)
    script(harness, stream=["讲解正文。"])
    return teaching, subject_id


async def start(teaching: object, subject_id: str) -> None:
    await teaching.start(subject_id)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 级别选择
# --------------------------------------------------------------------------- #


def test_hint_level_escalates_with_failures() -> None:
    """还没答过 / 答错 1 次 → 1 级；答错 2 次 → 2 级；3 次及以上 → 3 级。"""
    assert hint_level_for(0) == 1
    assert hint_level_for(1) == 1
    assert hint_level_for(2) == 2
    assert hint_level_for(3) == 3
    assert hint_level_for(99) == 3  # 封顶


def test_hint_level_labels_cover_all_levels() -> None:
    assert set(HINT_LEVEL_LABELS) == {1, 2, 3}
    assert HINT_LEVEL_LABELS[1] == "方向性提示"
    assert HINT_LEVEL_LABELS[3] == "近似答案"


# --------------------------------------------------------------------------- #
# 分级提示
# --------------------------------------------------------------------------- #


async def test_hint_defaults_to_level_one_before_any_answer(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["给你一个方向上的提示。"])

    turn = await teaching.hint(subject_id)

    assert turn.ok
    assert turn.phase == "hint"
    assert turn.hint_level == 1
    assert turn.pending_question == CHECKPOINT_QUESTION
    assert "给你一个方向上的提示。" in turn.text
    assert CHECKPOINT_QUESTION in turn.text  # 提示后重新复述问题
    assert "第 1 级" in harness.client.calls[-1].messages[0]["content"]


async def test_hint_escalates_after_failures(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    # 第一次失败：默认仍是第 1 级（一上来就拆题等于把答案递过去）
    script(harness, stream=["再讲一遍。"], blocking=[verdict_json("fail")])
    await teaching.answer(subject_id, "错答一")
    script(harness, stream=["方向性提示。"])
    first = await teaching.hint(subject_id)
    assert first.hint_level == 1

    # 第二次失败：升到第 2 级
    script(harness, stream=["再次重讲。"], blocking=[verdict_json("fail")])
    await teaching.answer(subject_id, "错答二")
    script(harness, stream=["关键要素提示。"])
    second = await teaching.hint(subject_id)

    assert second.hint_level == 2
    assert HINT_LEVEL_LABELS[2] in harness.client.calls[-1].messages[0]["content"]


async def test_explicit_hint_level_is_respected(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["接近答案的提示。"])

    turn = await teaching.hint(subject_id, level=3)

    assert turn.hint_level == 3
    assert "第 3 级" in harness.client.calls[-1].messages[0]["content"]


async def test_hint_level_is_clamped(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["提示一。", "提示二。"])

    low = await teaching.hint(subject_id, level=0)
    high = await teaching.hint(subject_id, level=99)

    assert low.hint_level == 1
    assert high.hint_level == 3


async def test_hint_does_not_consume_attempts_or_change_state(context: AppContext) -> None:
    """提示是脚手架，不是作答：尝试次数、节点状态、待答问题都不能变。"""
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    before = harness.state.snapshot(subject_id)
    script(harness, stream=["提示。"])

    await teaching.hint(subject_id)

    after = harness.state.snapshot(subject_id)
    assert after.node_attempts == before.node_attempts == 0
    assert after.pending_checkpoint_id == before.pending_checkpoint_id
    assert after.pending_question == before.pending_question
    assert after.current_node_id == before.current_node_id
    node = harness.repos.paths.node(after.current_node_id)
    assert node is not None and node.state is NodeState.LEARNING
    assert harness.repos.answers.list_for_node(node.id) == []


async def test_hint_keeps_the_same_checkpoint(context: AppContext) -> None:
    """提示后问的必须是同一道题：换题会让判定器拿到与用户看到的不一致的题面。"""
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    before = harness.state.snapshot(subject_id).pending_checkpoint_id
    script(harness, stream=["提示。"])

    turn = await teaching.hint(subject_id)

    assert harness.state.snapshot(subject_id).pending_checkpoint_id == before
    assert turn.pending_question == CHECKPOINT_QUESTION


async def test_hint_requires_a_pending_question(context: AppContext) -> None:
    """没有待答问题时不该给提示（否则用户会以为自己在答题）。"""
    from app.services.learning.state import AnswerNotAllowedError

    harness = make_harness(context)
    teaching, subject_id = setup(harness)

    with pytest.raises(AnswerNotAllowedError):
        await teaching.hint(subject_id)


async def test_hint_records_last_action(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["提示。"])

    await teaching.hint(subject_id, level=2)

    assert harness.state.snapshot(subject_id).last_action == hint_action(2)


async def test_hint_propagates_failure_without_side_effects(context: AppContext) -> None:
    from app.infra.llm.errors import LlmError

    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    before = harness.state.snapshot(subject_id)
    harness.client.stream_script = [ScriptedResponse(error=LlmError.timeout("超时"))]
    harness.client.stream_calls = 0

    turn = await teaching.hint(subject_id)

    assert not turn.ok
    assert turn.retryable is True
    after = harness.state.snapshot(subject_id)
    assert after.last_action == before.last_action  # 失败不改动"上一步动作"
    assert after.pending_question == before.pending_question


async def test_hint_is_replayable(context: AppContext) -> None:
    """提示失败后直接重试即可——不需要任何额外状态清理。"""
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["提示成功。", "提示成功。"])

    first = await teaching.hint(subject_id, level=1)
    second = await teaching.hint(subject_id, level=1)

    assert first.ok and second.ok
    assert harness.state.snapshot(subject_id).node_attempts == 0


# --------------------------------------------------------------------------- #
# 穿插提问（FR-411）
# --------------------------------------------------------------------------- #


async def test_ask_answers_and_returns_to_the_main_thread(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["它的区别在于……请回到上面的问题。"])

    turn = await teaching.ask(subject_id, "它和条件概率有什么区别？")

    assert turn.ok
    assert turn.phase == "off_topic"
    assert "它的区别在于" in turn.text
    assert CHECKPOINT_QUESTION in turn.text  # 回到主线：复述待答问题


async def test_ask_does_not_touch_teaching_state(context: AppContext) -> None:
    """核心不变量：随口提问绝不能被当成检查点作答。"""
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    before = harness.state.snapshot(subject_id)
    node_id = before.current_node_id
    assert node_id is not None
    script(harness, stream=["回答。"])

    await teaching.ask(subject_id, "顺便问一下，这个在工程里怎么用？")

    after = harness.state.snapshot(subject_id)
    assert after.node_attempts == before.node_attempts
    assert after.pending_checkpoint_id == before.pending_checkpoint_id
    assert after.pending_question == before.pending_question
    assert after.awaiting_decision == before.awaiting_decision
    # 岔路不该顶替主线上的"上一步"：重新生成要重跑讲解/提示，而不是把提问再答一遍
    assert after.last_action == before.last_action
    assert harness.repos.answers.list_for_node(node_id) == []
    node = harness.repos.paths.node(node_id)
    assert node is not None and node.state is NodeState.LEARNING


async def test_regenerate_after_ask_replays_the_main_thread(context: AppContext) -> None:
    """提问之后点"重新生成"，重跑的应该是提问之前的那条主线回复。"""
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    assert harness.state.snapshot(subject_id).last_action == ACTION_EXPLAIN
    script(harness, stream=["回答。"])
    await teaching.ask(subject_id, "这个在工程里怎么用？")

    script(harness, stream=["换个说法再讲一遍"])
    turn = await teaching.regenerate_last(subject_id)

    assert turn.phase == "explain"
    assert "换个说法再讲一遍" in turn.text


async def test_ask_marks_messages_as_off_topic(context: AppContext) -> None:
    """标记要落到存储里：档案与日志都靠它把主线与随口提问分开。"""
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["回答。"])

    await teaching.ask(subject_id, "随口一问")

    # 从数据库重新读回来，确认标记持久化
    conversation_id = harness.ctx.memories.conversation_id_for(subject_id, ConversationKind.TEACH)
    assert conversation_id is not None
    reloaded = harness.ctx.memories.load(conversation_id)
    kinds = [envelope.kind for envelope in reloaded.envelopes()]
    assert KIND_OFF_TOPIC in kinds
    # 标记不进协议字典：发给模型的仍然是干净的 role/content
    assert all("kind" not in message for message in reloaded.outbound_messages())
    marked = [envelope for envelope in reloaded.envelopes() if envelope.kind == KIND_OFF_TOPIC]
    assert any(envelope.content == "随口一问" for envelope in marked)
    assert any("回答。" in envelope.content for envelope in marked)


async def test_ask_after_failure_keeps_the_decision_state(context: AppContext) -> None:
    """等三选一时提问也是允许的，但不能因此取消三选一。"""
    harness = make_harness(context)
    teaching, subject_id = setup(harness, retries=0)
    await start(teaching, subject_id)
    script(harness, blocking=[verdict_json("fail")])
    await teaching.answer(subject_id, "错答")
    assert harness.state.snapshot(subject_id).needs_user_decision is True

    script(harness, stream=["回答。"])
    await teaching.ask(subject_id, "顺便问个问题")

    assert harness.state.snapshot(subject_id).needs_user_decision is True


async def test_ask_rejects_empty_question(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)

    turn = await teaching.ask(subject_id, "   ")

    assert not turn.ok
    assert turn.error_category == "validation"


async def test_ask_requires_teaching_state(context: AppContext) -> None:
    from app.services.learning.state import AnswerNotAllowedError

    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2, confirmed=False)

    with pytest.raises(AnswerNotAllowedError):
        await teaching.ask(subject_id, "问题")


async def test_ask_prompt_carries_the_pending_question(context: AppContext) -> None:
    """提示词里必须带上"当前在等什么"，模型才知道回答完要把人带回哪道题。"""
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["回答。"])

    await teaching.ask(subject_id, "我的问题")

    system = harness.client.calls[-1].messages[0]["content"]
    assert CHECKPOINT_QUESTION in system
    assert "我的问题" in system


# --------------------------------------------------------------------------- #
# 重新生成上一条回复（FR-413）
# --------------------------------------------------------------------------- #


async def test_regenerate_without_action_is_rejected(context: AppContext) -> None:
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)

    with pytest.raises(TeachingError) as exc:
        await teaching.regenerate_last(subject_id)

    assert "没有可以重新生成" in str(exc.value)


async def test_regenerate_replays_explanation(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    assert harness.state.snapshot(subject_id).last_action == ACTION_EXPLAIN
    script(harness, stream=["换一种讲法的讲解。"])

    turn = await teaching.regenerate_last(subject_id)

    assert turn.ok
    assert turn.phase == "explain"
    assert "换一种讲法的讲解。" in turn.text
    assert harness.client.calls[-1].messages[0]["content"].startswith("# 教学（讲解）")


async def test_regenerate_replays_hint_with_the_same_level(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["第一次提示。", "第二次提示。"])
    await teaching.hint(subject_id, level=2)

    turn = await teaching.regenerate_last(subject_id)

    assert turn.hint_level == 2
    assert "第 2 级" in harness.client.calls[-1].messages[0]["content"]


async def test_regenerate_replays_reteach_with_direction(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["重讲。"], blocking=[verdict_json("fail")])
    await teaching.answer(subject_id, "错答")
    assert harness.state.snapshot(subject_id).last_action == "reteach"
    script(harness, stream=["再重讲一次。"])

    turn = await teaching.regenerate_last(subject_id)

    assert turn.ok
    assert turn.phase == "reteach"
    assert CHECKPOINT_QUESTION in turn.text


async def test_regenerate_does_not_undo_recorded_judgements(context: AppContext) -> None:
    """重生成只是"再说一遍"，不该抹掉已经发生的作答记录。"""
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["重讲。"], blocking=[verdict_json("fail")])
    await teaching.answer(subject_id, "错答")
    before = harness.state.snapshot(subject_id)
    node_id = before.current_node_id
    assert node_id is not None
    records_before = harness.repos.answers.list_for_node(node_id)
    script(harness, stream=["新的讲法。"])

    await teaching.regenerate_last(subject_id)

    after = harness.state.snapshot(subject_id)
    assert [record.user_answer for record in harness.repos.answers.list_for_node(node_id)] == [
        record.user_answer for record in records_before
    ]
    assert after.node_attempts == before.node_attempts


async def test_regenerate_reference_replays_reference(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness, retries=0)
    await start(teaching, subject_id)
    script(harness, blocking=[verdict_json("fail")])
    await teaching.answer(subject_id, "错答")
    script(harness, stream=["参考答案。"])
    await teaching.decide(subject_id, NodeDecision.SHOW_ANSWER)
    assert harness.state.snapshot(subject_id).last_action == ACTION_REFERENCE
    script(harness, stream=["再给一次参考答案。"])

    turn = await teaching.regenerate_last(subject_id)

    assert turn.ok
    assert "再给一次参考答案。" in turn.text


async def test_last_action_clears_when_moving_to_the_next_checkpoint(context: AppContext) -> None:
    """多检查点节点：过了第一个检查点后，上一条回复已翻篇。"""
    harness = make_harness(context)
    harness.settings = AppSettings(node_max_retries=2)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=1, checkpoints=2)
    script(harness, stream=["讲解正文。"])
    await teaching.start(subject_id)
    assert harness.state.snapshot(subject_id).last_action == ACTION_EXPLAIN
    script(harness, stream=["继续。"], blocking=[verdict_json("pass")])

    await teaching.answer(subject_id, "答对")

    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.last_action is None
    assert snapshot.pending_question == "节点 0 的检查点 1？"  # 换到了下一个检查点
    with pytest.raises(TeachingError):
        await teaching.regenerate_last(subject_id)


async def test_last_action_clears_when_the_node_is_mastered(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness)
    await start(teaching, subject_id)
    script(harness, stream=["小结。"], blocking=[verdict_json("pass")])

    await teaching.answer(subject_id, "答对")

    node_id = harness.state.snapshot(subject_id).current_node_id
    node = harness.repos.paths.node(node_id) if node_id else None
    assert node is not None and node.state is NodeState.MASTERED
    assert harness.state.snapshot(subject_id).last_action is None


async def test_last_action_clears_when_switching_nodes(context: AppContext) -> None:
    harness = make_harness(context)
    teaching, subject_id = setup(harness, node_count=2)
    await start(teaching, subject_id)
    path = harness.repos.paths.get_active(subject_id)
    assert path is not None
    harness.state.enter_node(subject_id, path.ordered_nodes()[1])

    assert harness.state.snapshot(subject_id).last_action is None
