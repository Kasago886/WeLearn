"""上下文成本策略测试（T1.7，FR-719、FR-720）。"""

from app.services.llm.context_policy import (
    build_outbound_messages,
    estimate_context_pressure,
    reasoning_payload_size,
    stable_prefix_hash,
    strip_reasoning,
)
from app.services.llm.memory import (
    CONTENT_KEY,
    REASONING_KEY,
    Memory,
    assistant_message,
    user_message,
)


def build_memory() -> Memory:
    memory = Memory()
    round_ = memory.add_round()
    round_.add(user_message("什么是贝叶斯定理？"))
    round_.add(assistant_message("它是……", reasoning="先讲定义，再举例子。"))
    return memory


def test_strip_reasoning_removes_only_that_field() -> None:
    messages = [
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答", "reasoning_content": "思维链"},
    ]

    stripped = strip_reasoning(messages)

    assert stripped[0] == {"role": "user", "content": "问题"}
    assert stripped[1] == {"role": "assistant", "content": "回答"}
    assert REASONING_KEY not in stripped[1]
    # 原列表未被就地修改
    assert messages[1][REASONING_KEY] == "思维链"


def test_outbound_keeps_reasoning_when_tools_enabled() -> None:
    """FR-719：携带 tools 时历史 reasoning_content 必须回传，否则官方会丢弃上下文。"""
    messages = build_outbound_messages(build_memory(), tools_enabled=True)

    assert messages[1][REASONING_KEY] == "先讲定义，再举例子。"


def test_outbound_strips_reasoning_when_tools_disabled() -> None:
    """不携带 tools 时官方会忽略 reasoning_content，剔除只为减小请求体。"""
    messages = build_outbound_messages(build_memory(), tools_enabled=False)

    assert REASONING_KEY not in messages[1]
    assert messages[1][CONTENT_KEY] == "它是……"


def test_reasoning_payload_size() -> None:
    assert reasoning_payload_size(build_outbound_messages(build_memory(), tools_enabled=True)) == len(
        "先讲定义，再举例子。"
    )
    assert reasoning_payload_size(build_outbound_messages(build_memory(), tools_enabled=False)) == 0


def test_prefix_hash_is_stable_across_rounds() -> None:
    """FR-720：新增一轮不得改变历史前缀，否则缓存必然失效。"""
    memory = build_memory()
    before = stable_prefix_hash(build_outbound_messages(memory, tools_enabled=False))

    memory.add_round().add(user_message("再讲讲例子"))
    memory.add_message_to_last_round(assistant_message("例如"))

    after_messages = build_outbound_messages(memory, tools_enabled=False)
    after = stable_prefix_hash(after_messages)

    assert before != after  # 整体哈希变了（新增了消息）
    # 但前缀部分完全一致
    assert before == stable_prefix_hash(after_messages[:2])


def test_prefix_hash_changes_when_history_rewritten() -> None:
    """反面对照：一旦改写历史消息，前缀指纹必须变化（能被测试发现）。"""
    memory = build_memory()
    original = stable_prefix_hash(build_outbound_messages(memory, tools_enabled=False))

    memory.rounds[0].messages[1].data[CONTENT_KEY] = "被改写过的回答"

    assert stable_prefix_hash(build_outbound_messages(memory, tools_enabled=False)) != original


def test_estimate_context_pressure() -> None:
    messages = build_outbound_messages(build_memory(), tools_enabled=False)

    assert estimate_context_pressure(messages, threshold_rounds=2) is True
    assert estimate_context_pressure(messages, threshold_rounds=10) is False
