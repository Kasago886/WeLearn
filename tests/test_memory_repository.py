"""记忆持久化测试（T1.1，FR-603、FR-410）。"""

from app.contracts.schemas import ConversationKind
from app.infra.storage.models import ChatRoundRow, ConversationRow, MessageRow
from app.main_context import AppContext
from app.services.llm.memory import (
    CONTENT_KEY,
    REASONING_KEY,
    ROLE_KEY,
    Memory,
    assistant_message,
    summary_message,
    system_message,
    user_message,
)


def build_memory() -> Memory:
    memory = Memory()
    memory.add_round().add(system_message("你是耐心的导师"))
    first = memory.add_round()
    first.add(user_message("什么是贝叶斯定理？"))
    first.add(assistant_message("它是用新证据更新信念的方法。", reasoning="先给定义。"))
    second = memory.add_round()
    second.add(user_message("举个例子")),
    second.add(assistant_message("例如医学检测。"))
    return memory


def test_ensure_conversation_is_idempotent(context: AppContext) -> None:
    repo = context.memories

    first = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")
    second = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")

    assert first == second


def test_different_kinds_get_separate_conversations(context: AppContext) -> None:
    """FR-602：测评与教学必须是相互独立的记忆。"""
    repo = context.memories

    teach = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")
    assess = repo.ensure_conversation(ConversationKind.ASSESS, subject_id="s1")
    review = repo.ensure_conversation(ConversationKind.REVIEW, subject_id="s1")

    assert len({teach, assess, review}) == 3
    assert repo.conversation_id_for("s1", ConversationKind.ASSESS) == assess


def test_save_and_load_round_trip(context: AppContext) -> None:
    repo = context.memories
    conversation_id = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")

    written = repo.save(conversation_id, build_memory())
    loaded = repo.load(conversation_id)

    assert written == 3
    assert loaded.round_count == 3
    assert loaded.message_count == 5
    assert loaded.outbound_messages() == build_memory().outbound_messages()


def test_load_preserves_roles_reasoning_and_metadata(context: AppContext) -> None:
    repo = context.memories
    conversation_id = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")
    repo.save(conversation_id, build_memory())

    loaded = repo.load(conversation_id)
    messages = loaded.outbound_messages()

    assert [message[ROLE_KEY] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[1][CONTENT_KEY] == "什么是贝叶斯定理？"
    assert messages[2][REASONING_KEY] == "先给定义。"
    # 空的 reasoning 不应被写成 null 字段
    assert REASONING_KEY not in messages[4]


def test_summary_flag_survives_round_trip(context: AppContext) -> None:
    """FR-605：摘要标记必须持久化，否则重启后会分不清哪条是摘要。"""
    repo = context.memories
    conversation_id = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")

    memory = build_memory()
    memory.replace_rounds_with_summary(1, 2, "前两轮摘要")
    repo.save(conversation_id, memory)

    loaded = repo.load(conversation_id)
    flags = [envelope.is_summary for envelope in loaded.envelopes()]
    assert flags.count(True) == 1

    summary_envelope = next(envelope for envelope in loaded.envelopes() if envelope.is_summary)
    assert "前两轮摘要" in summary_envelope.content
    assert "不是新的用户请求" in summary_envelope.content


def test_save_overwrites_previous_content(context: AppContext) -> None:
    """反复保存不得累积旧数据（整体重写的语义保障）。"""
    repo = context.memories
    conversation_id = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")

    repo.save(conversation_id, build_memory())
    smaller = Memory.from_protocol([[user_message("只剩一轮")]])
    repo.save(conversation_id, smaller)

    loaded = repo.load(conversation_id)
    assert loaded.round_count == 1
    assert loaded.message_count == 1

    with context.db.session() as session:
        assert session.query(MessageRow).count() == 1
        assert session.query(ChatRoundRow).count() == 1


def test_round_count_is_tracked_on_conversation(context: AppContext) -> None:
    repo = context.memories
    conversation_id = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")
    repo.save(conversation_id, build_memory())

    with context.db.session() as session:
        row = session.get(ConversationRow, conversation_id)

    assert row is not None
    assert row.round_count == 3
    assert row.kind == "teach"


def test_load_unknown_conversation_returns_empty_memory(context: AppContext) -> None:
    loaded = context.memories.load("不存在的会话")

    assert loaded.is_empty is True


def test_save_unknown_conversation_raises(context: AppContext) -> None:
    import pytest

    with pytest.raises(KeyError):
        context.memories.save("不存在的会话", Memory())


def test_mark_compressed_records_timestamp(context: AppContext) -> None:
    repo = context.memories
    conversation_id = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")

    repo.mark_compressed(conversation_id)

    with context.db.session() as session:
        row = session.get(ConversationRow, conversation_id)
    assert row is not None
    assert row.compressed_at is not None


def test_delete_conversation_cascades(context: AppContext) -> None:
    repo = context.memories
    conversation_id = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")
    repo.save(conversation_id, build_memory())

    repo.delete_conversation(conversation_id)

    with context.db.session() as session:
        assert session.get(ConversationRow, conversation_id) is None
        assert session.query(ChatRoundRow).count() == 0
        assert session.query(MessageRow).count() == 0


def test_restore_pending_question_scenario(context: AppContext) -> None:
    """FR-410 的数据基础：待回答的检查点问题必须能从落库状态恢复。"""
    repo = context.memories
    conversation_id = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")

    memory = Memory()
    memory.add_round().add(system_message("你是导师"))
    memory.add_round().add(assistant_message("请解释贝叶斯公式，并说说各部分的含义。"))
    repo.save(conversation_id, memory)

    # 模拟刷新页面后重新加载
    restored = repo.load(conversation_id)
    last = restored.outbound_messages()[-1]

    assert last[ROLE_KEY] == "assistant"
    assert "请解释贝叶斯公式" in last[CONTENT_KEY]
    # 尚未有作答 → 最后一轮没有 user 消息
    assert restored.last_round.count == 1


def test_summary_message_helper_matches_stored_form(context: AppContext) -> None:
    """落库后再读回的消息要与内存里构造的一致（避免"存了但读不回"的隐性不一致）。"""
    repo = context.memories
    conversation_id = repo.ensure_conversation(ConversationKind.TEACH, subject_id="s1")

    memory = Memory()
    memory.add_round().add(summary_message("摘要内容"))
    repo.save(conversation_id, memory)

    assert repo.load(conversation_id).outbound_messages() == memory.outbound_messages()
