"""记忆模型测试（T1.1，FR-601、FR-603）。"""

import pytest

from app.services.llm.memory import (
    CONTENT_KEY,
    REASONING_KEY,
    ROLE_KEY,
    TOOL_CALLS_KEY,
    ChatRound,
    Memory,
    MessageEnvelope,
    assistant_message,
    ensure_leading_system,
    get_role,
    set_leading_system,
    summary_message,
    system_message,
    tool_message,
    user_message,
)


def test_message_builders_omit_empty_optional_fields() -> None:
    """空的 reasoning / tool_calls 不写入，避免部分服务端不接受 null 字段。"""
    plain = assistant_message("你好")
    assert plain == {ROLE_KEY: "assistant", CONTENT_KEY: "你好"}
    assert REASONING_KEY not in plain
    assert TOOL_CALLS_KEY not in plain

    with_all = assistant_message("你好", reasoning="想一下", tool_calls=[{"id": "1"}])
    assert with_all[REASONING_KEY] == "想一下"
    assert with_all[TOOL_CALLS_KEY] == [{"id": "1"}]

    assert assistant_message(None)[CONTENT_KEY] == ""


def test_role_helpers() -> None:
    assert get_role(user_message("x")) == "user"
    assert get_role({}) is None
    assert get_role({ROLE_KEY: 123}) is None
    assert system_message("s")[ROLE_KEY] == "system"
    assert tool_message("call_1", None)["tool_call_id"] == "call_1"


def test_summary_message_declares_it_is_not_a_user_request() -> None:
    """FR-605：摘要必须声明"不是新的用户请求"，否则模型会把它当指令执行。"""
    message = summary_message("用户已掌握先验概率。")

    assert message[ROLE_KEY] == "user"
    content = message[CONTENT_KEY]
    assert "压缩摘要" in content
    assert "不是新的用户请求" in content
    assert "以后续消息为准" in content
    assert "用户已掌握先验概率。" in content


def test_metadata_stays_out_of_protocol_payload() -> None:
    """信封元信息（是否摘要）不能进入发给模型的字段，否则会被当作未知字段下发。"""
    memory = Memory()
    round_ = memory.add_round()
    round_.add(user_message("问题"))
    round_.add(summary_message("摘要"), is_summary=True)

    outbound = memory.outbound_messages()
    for message in outbound:
        assert "is_summary" not in message

    assert [envelope.is_summary for envelope in memory.envelopes()] == [False, True]


def test_round_and_memory_counts() -> None:
    memory = Memory.from_protocol(
        [
            [system_message("你是导师")],
            [user_message("什么是贝叶斯定理？"), assistant_message("它是……")],
        ]
    )

    assert memory.round_count == 2
    assert memory.message_count == 3
    assert memory.last_round.count == 2
    assert memory.is_empty is False
    assert Memory().is_empty is True


def test_system_round_detection_and_leading_count() -> None:
    memory = Memory.from_protocol(
        [
            [system_message("s1")],
            [system_message("s2")],
            [user_message("u1"), assistant_message("a1")],
        ]
    )

    assert memory.count_leading_system_rounds() == 2
    assert memory.rounds[0].is_system_round is True
    assert memory.rounds[2].is_system_round is False

    # 开头不是 system 的轮次不计入
    plain = Memory.from_protocol([[user_message("u")]])
    assert plain.count_leading_system_rounds() == 0


def test_snapshot_is_isolated() -> None:
    """FR-610 的结构基础：副本的任何改动都不得影响正式记忆。"""
    memory = Memory.from_protocol([[user_message("原始问题")]])
    snapshot = memory.snapshot()

    snapshot.add_round().add(assistant_message("副本里的回答"))
    snapshot.last_round.add(user_message("副本里的追问"))
    snapshot.outbound_messages()[0][CONTENT_KEY] = "被改写"

    assert memory.round_count == 1
    assert memory.message_count == 1
    assert memory.outbound_messages()[0][CONTENT_KEY] == "原始问题"


def test_outbound_messages_are_deep_copies() -> None:
    memory = Memory.from_protocol([[user_message("问题")]])

    outbound = memory.outbound_messages()
    outbound[0][CONTENT_KEY] = "被改写"
    outbound[0]["嵌套"] = {"a": 1}

    assert memory.outbound_messages()[0][CONTENT_KEY] == "问题"
    assert "嵌套" not in memory.outbound_messages()[0]


def test_deep_copy_preserves_nested_tool_calls() -> None:
    memory = Memory()
    memory.add_round().add(
        assistant_message("", tool_calls=[{"id": "1", "function": {"name": "f", "arguments": "{}"}}])
    )

    copied = memory.snapshot().outbound_messages()
    assert copied[0][TOOL_CALLS_KEY][0]["function"]["name"] == "f"

    copied[0][TOOL_CALLS_KEY][0]["function"]["name"] = "changed"
    assert memory.outbound_messages()[0][TOOL_CALLS_KEY][0]["function"]["name"] == "f"


def test_replace_with_is_the_commit_point() -> None:
    memory = Memory.from_protocol([[user_message("旧")]])
    replacement = Memory.from_protocol([[user_message("新")], [assistant_message("回答")]])

    memory.replace_with(replacement)

    assert memory.round_count == 2
    assert memory.outbound_messages()[0][CONTENT_KEY] == "新"


def test_add_message_to_last_round_creates_round_when_empty() -> None:
    memory = Memory()
    memory.add_message_to_last_round(user_message("第一条"))

    assert memory.round_count == 1
    assert memory.message_count == 1


def test_protocol_range_validates_bounds() -> None:
    memory = Memory.from_protocol([[user_message("a")], [assistant_message("b")]])

    assert memory.protocol_range(0, 2) == [
        {ROLE_KEY: "user", CONTENT_KEY: "a"},
        {ROLE_KEY: "assistant", CONTENT_KEY: "b"},
    ]
    assert memory.protocol_range(1, 5) == []
    assert memory.protocol_range(-1, 1) == []
    assert memory.protocol_range(0, 0) == []


def test_remove_and_insert_rounds() -> None:
    memory = Memory.from_protocol([[user_message("a")], [user_message("b")], [user_message("c")]])

    assert memory.remove_rounds(1, 1) is True
    assert [message[CONTENT_KEY] for message in memory.outbound_messages()] == ["a", "c"]

    assert memory.remove_rounds(5, 1) is False
    assert memory.remove_rounds(0, 0) is False

    memory.insert_round(1)
    assert memory.round_count == 3


def test_replace_rounds_with_summary() -> None:
    memory = Memory.from_protocol(
        [
            [system_message("s")],
            [user_message("a")],
            [assistant_message("b")],
            [user_message("c")],
        ]
    )

    assert memory.replace_rounds_with_summary(1, 2, "前两轮摘要") is True

    assert memory.round_count == 3
    envelopes = memory.envelopes()
    assert [envelope.is_summary for envelope in envelopes] == [False, True, False]
    assert "前两轮摘要" in envelopes[1].content
    # system 仍是第一轮
    assert memory.rounds[0].is_system_round is True

    assert memory.replace_rounds_with_summary(2, 5, "越界") is False


def test_round_clone_is_independent() -> None:
    original = ChatRound([MessageEnvelope(user_message("a"))])
    clone = original.clone()

    clone.add(assistant_message("b"))

    assert original.count == 1
    assert clone.count == 2


def test_empty_memory_has_no_last_round() -> None:
    with pytest.raises(IndexError):
        _ = Memory().last_round


def test_set_leading_system_replaces_first_system_message() -> None:
    memory = Memory.from_protocol([[system_message("旧提示")], [user_message("u"), assistant_message("a")]])

    assert set_leading_system(memory, "新提示") is True

    messages = memory.outbound_messages()
    assert messages[0][CONTENT_KEY] == "新提示"
    assert messages[1][CONTENT_KEY] == "u"
    assert len(messages) == 3  # 没有新增消息


def test_set_leading_system_returns_false_without_system_message() -> None:
    memory = Memory.from_protocol([[user_message("u")]])
    assert set_leading_system(memory, "提示") is False
    assert memory.message_count == 1


def test_ensure_leading_system_inserts_when_missing() -> None:
    """没有 system 消息时必须在最前面插入，否则提示词会被静默丢弃（模型用上一轮的提示）。"""
    memory = Memory.from_protocol([[user_message("用户消息"), assistant_message("回答")]])

    ensure_leading_system(memory, "本轮提示")

    messages = memory.outbound_messages()
    assert messages[0][ROLE_KEY] == "system"
    assert messages[0][CONTENT_KEY] == "本轮提示"
    assert messages[1][CONTENT_KEY] == "用户消息"
    # system 必须独占第一轮，之后才是内容轮次
    assert memory.rounds[0].is_system_round is True
    assert memory.round_count == 2


def test_ensure_leading_system_is_idempotent() -> None:
    memory = Memory.from_protocol([[user_message("u")]])

    ensure_leading_system(memory, "提示一")
    ensure_leading_system(memory, "提示二")

    assert memory.message_count == 2  # system + user，没有重复追加
    assert memory.outbound_messages()[0][CONTENT_KEY] == "提示二"
