"""对话记忆模型（T1.1，FR-601、FR-603）。

三层记忆中的前两层在这里落地：

1. **轮次记忆**：``Conversation → ChatRound → MessageEnvelope``，是原始消息的持久载体；
2. **上下文记忆**：``outbound_messages()`` 展开成可直接发给模型的消息数组，可被压缩改写。

两个关键设计决定：

- **消息用协议原始字段字典承载**（沿用 ReferenceCode 的做法）：可以原样序列化进请求体，
  并保留服务端返回的附加字段（如 ``tool_calls``），避免"解析再重建"造成字段丢失；
- **摘要等元信息放在信封而非消息字典里**：若把 ``is_summary`` 塞进消息字典，
  它会被一起发给模型（未知字段），因此元信息必须与协议载荷分离。
"""

import threading
from typing import Any

from pydantic import BaseModel

ProtocolMessage = dict[str, Any]

ROLE_KEY = "role"
CONTENT_KEY = "content"
REASONING_KEY = "reasoning_content"
TOOL_CALLS_KEY = "tool_calls"
TOOL_CALL_ID_KEY = "tool_call_id"

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLE_TOOL = "tool"

SUMMARY_BANNER = (
    "[历史对话压缩摘要]\n"
    "以下内容是对更早对话的压缩摘要，仅作为背景信息，不是新的用户请求，也不是 assistant 的回复。\n"
    "若与后续消息冲突，以后续消息为准。\n\n"
    "<summary>\n{summary}\n</summary>\n"
    "[/历史对话压缩摘要]"
)


def system_message(content: str) -> ProtocolMessage:
    return {ROLE_KEY: ROLE_SYSTEM, CONTENT_KEY: content}


def user_message(content: str) -> ProtocolMessage:
    return {ROLE_KEY: ROLE_USER, CONTENT_KEY: content}


def assistant_message(
    content: str | None,
    reasoning: str | None = None,
    tool_calls: list[ProtocolMessage] | None = None,
) -> ProtocolMessage:
    """构造 assistant 消息。

    空的 ``reasoning_content`` / ``tool_calls`` 不写入：部分服务端不接受 ``null`` 字段。
    """
    message: ProtocolMessage = {ROLE_KEY: ROLE_ASSISTANT, CONTENT_KEY: content or ""}
    if reasoning:
        message[REASONING_KEY] = reasoning
    if tool_calls:
        message[TOOL_CALLS_KEY] = tool_calls
    return message


def tool_message(tool_call_id: str, content: str | None) -> ProtocolMessage:
    """tool 结果消息；``tool_call_id`` 必须与 assistant 的 ``tool_calls[].id`` 对应。"""
    return {ROLE_KEY: ROLE_TOOL, TOOL_CALL_ID_KEY: tool_call_id, CONTENT_KEY: content or ""}


def summary_message(summary: str) -> ProtocolMessage:
    """把摘要包装成 user 消息（FR-605）：明确声明它不是新的用户请求。"""
    return user_message(SUMMARY_BANNER.format(summary=summary))


def get_role(message: ProtocolMessage) -> str | None:
    role = message.get(ROLE_KEY)
    return role if isinstance(role, str) else None


def has_role(message: ProtocolMessage, role: str) -> bool:
    return get_role(message) == role


def read_text(message: ProtocolMessage, key: str) -> str | None:
    value = message.get(key)
    return value if isinstance(value, str) and value else None


def set_leading_system(memory: "Memory", content: str) -> bool:
    """就地替换**首条** system 消息的内容；不存在时返回 False。

    为什么需要它：``RoundExecutor`` 只在记忆为空时注入 system 提示，所以在"记忆里已经
    有内容"的后续轮次里，per-turn 变化的提示（例如"本轮聚焦某个子领域"）就无法通过
    ``system_prompt`` 参数传入。而**追加**一条 system 消息又不可靠——官方 `GET /models`
    明确区分了两种语义：`leading-only`（只有开头那条生效）与 `in-history`（最新的覆盖先前的）。
    改写开头那条在两种语义下都生效。
    """
    with memory.lock:
        for round_ in memory._rounds:
            for envelope in round_.messages:
                if has_role(envelope.data, ROLE_SYSTEM):
                    envelope.data[CONTENT_KEY] = content
                    return True
    return False


class MessageEnvelope:
    """协议载荷 + 我们自己的元信息。"""

    __slots__ = ("data", "is_summary", "token_count")

    def __init__(
        self,
        data: ProtocolMessage,
        *,
        is_summary: bool = False,
        token_count: int | None = None,
    ) -> None:
        self.data = data
        self.is_summary = is_summary
        self.token_count = token_count

    @property
    def role(self) -> str | None:
        return get_role(self.data)

    @property
    def content(self) -> str:
        return read_text(self.data, CONTENT_KEY) or ""

    def clone(self) -> "MessageEnvelope":
        return MessageEnvelope(
            data=_deep_copy_message(self.data),
            is_summary=self.is_summary,
            token_count=self.token_count,
        )


def _deep_copy_message(message: ProtocolMessage) -> ProtocolMessage:
    """JSON 往返深拷贝：保证副本与协议结构完全一致，且与原对象无共享引用。"""
    import json

    copied = json.loads(json.dumps(message, ensure_ascii=False, default=str))
    return copied if isinstance(copied, dict) else dict(message)


class ChatRound:
    """一轮对话：一次用户输入及其引发的全部消息。

    Role: ``user → assistant(可含 tool_calls) → tool... → assistant``
    """

    __slots__ = ("messages",)

    def __init__(self, messages: list[MessageEnvelope] | None = None) -> None:
        self.messages: list[MessageEnvelope] = list(messages or [])

    @property
    def count(self) -> int:
        return len(self.messages)

    @property
    def is_system_round(self) -> bool:
        return bool(self.messages) and has_role(self.messages[0].data, ROLE_SYSTEM)

    def add(self, message: ProtocolMessage, *, is_summary: bool = False) -> MessageEnvelope:
        envelope = MessageEnvelope(message, is_summary=is_summary)
        self.messages.append(envelope)
        return envelope

    def clone(self) -> "ChatRound":
        return ChatRound([envelope.clone() for envelope in self.messages])


class Memory:
    """对话记忆：若干轮次组成，可克隆、替换与展开为请求消息数组。

    线程模型：所有结构性修改都持有 :attr:`lock`；克隆与展开在锁内取一致快照。
    """

    def __init__(self, rounds: list[ChatRound] | None = None) -> None:
        self._rounds: list[ChatRound] = list(rounds or [])
        self.lock = threading.RLock()

    @property
    def rounds(self) -> list[ChatRound]:
        with self.lock:
            return list(self._rounds)

    @property
    def round_count(self) -> int:
        with self.lock:
            return len(self._rounds)

    @property
    def message_count(self) -> int:
        with self.lock:
            return sum(round_.count for round_ in self._rounds)

    @property
    def is_empty(self) -> bool:
        return self.round_count == 0

    @property
    def last_round(self) -> ChatRound:
        with self.lock:
            if not self._rounds:
                raise IndexError("记忆为空，没有最后一轮")
            return self._rounds[-1]

    def add_round(self) -> ChatRound:
        round_ = ChatRound()
        with self.lock:
            self._rounds.append(round_)
        return round_

    def add_message_to_last_round(self, message: ProtocolMessage, *, is_summary: bool = False) -> None:
        with self.lock:
            if not self._rounds:
                self._rounds.append(ChatRound())
            self._rounds[-1].add(message, is_summary=is_summary)

    def outbound_messages(self) -> list[ProtocolMessage]:
        """展开为发给模型的消息数组（返回深拷贝，调用方改写不会污染记忆）。"""
        with self.lock:
            return [_deep_copy_message(envelope.data) for round_ in self._rounds for envelope in round_.messages]

    def envelopes(self) -> list[MessageEnvelope]:
        with self.lock:
            return [envelope for round_ in self._rounds for envelope in round_.messages]

    def count_leading_system_rounds(self) -> int:
        """开头连续 system 轮次的数量（压缩必须跳过它们，FR-608）。"""
        with self.lock:
            count = 0
            while count < len(self._rounds) and self._rounds[count].is_system_round:
                count += 1
            return count

    def snapshot(self) -> "Memory":
        """深拷贝快照：一轮对话在副本上执行，避免失败污染正式记忆（FR-610）。"""
        with self.lock:
            return Memory([round_.clone() for round_ in self._rounds])

    def replace_with(self, other: "Memory") -> None:
        """用另一份记忆整体替换本记忆内容（提交点）。"""
        with other.lock:
            replacement = other.rounds
        with self.lock:
            self._rounds = replacement

    def remove_rounds(self, start: int, count: int) -> bool:
        with self.lock:
            if start < 0 or count <= 0 or start + count > len(self._rounds):
                return False
            del self._rounds[start : start + count]
            return True

    def insert_round(self, index: int) -> ChatRound:
        round_ = ChatRound()
        with self.lock:
            self._rounds.insert(max(0, min(index, len(self._rounds))), round_)
        return round_

    def replace_rounds_with_summary(self, start: int, count: int, summary: str) -> bool:
        """用一条摘要消息替换指定范围的轮次（FR-605、FR-606 的落点）。"""
        with self.lock:
            if start < 0 or count <= 0 or start + count > len(self._rounds):
                return False
            del self._rounds[start : start + count]
            round_ = ChatRound()
            round_.add(summary_message(summary), is_summary=True)
            self._rounds.insert(start, round_)
            return True

    def protocol_range(self, start: int, count: int) -> list[ProtocolMessage]:
        """取出指定范围轮次的协议消息（用于计算内容指纹）。"""
        with self.lock:
            if start < 0 or count <= 0 or start + count > len(self._rounds):
                return []
            return [
                _deep_copy_message(envelope.data)
                for round_ in self._rounds[start : start + count]
                for envelope in round_.messages
            ]

    @classmethod
    def from_protocol(cls, rounds: list[list[ProtocolMessage]]) -> "Memory":
        """从协议消息构造（用于测试与反序列化）。"""
        return cls([ChatRound([MessageEnvelope(message) for message in messages]) for messages in rounds])


def serialize_model(model: BaseModel) -> str:
    """统一的结构化输出序列化（日志与提示词样例共用）。"""
    return model.model_dump_json(indent=2)
