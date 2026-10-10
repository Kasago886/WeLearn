"""测试替身：可编程的 LLM 客户端桩（NFR-115）。

作用与 ReferenceCode 的 ``ILlmService`` 一致：让状态机、判定、压缩、解析这些逻辑
**完全离线可测**，不消耗真实额度、不受网络影响。
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.contracts.schemas import LlmPurpose
from app.infra.deepseek.errors import LlmError
from app.infra.deepseek.protocol import BlockingResult, StreamEvent, TokenUsage


@dataclass
class RecordedCall:
    """一次调用的入参快照，供断言检查。"""

    kind: str
    model: str
    messages: list[dict[str, Any]]
    thinking: bool
    reasoning_effort: str
    max_tokens: int
    json_object: bool
    purpose: LlmPurpose
    timeout_seconds: int | None


@dataclass
class ScriptedResponse:
    """一次脚本化响应：可以是事件序列，也可以是异常。"""

    events: list[StreamEvent] = field(default_factory=list)
    content: str = ""
    reasoning: str | None = None
    usage: TokenUsage | None = None
    error: LlmError | None = None


def text_events(*chunks: str, usage: TokenUsage | None = None, reasoning: str | None = None) -> list[StreamEvent]:
    """构造一串正文增量事件（可附思维链与用量）。"""
    events: list[StreamEvent] = []
    if reasoning:
        events.append(StreamEvent(reasoning=reasoning))
    events.extend(StreamEvent(content=chunk) for chunk in chunks)
    if usage is not None:
        events.append(StreamEvent(usage=usage))
    return events


class FakeLlmClient:
    """按脚本返回响应的客户端桩。"""

    def __init__(
        self,
        *,
        stream_script: list[ScriptedResponse] | None = None,
        blocking_script: list[ScriptedResponse] | None = None,
        fallback: ScriptedResponse | None = None,
    ) -> None:
        self.stream_script = list(stream_script or [])
        self.blocking_script = list(blocking_script or [])
        self.fallback = fallback
        self.calls: list[RecordedCall] = []
        self.stream_calls = 0
        self.blocking_calls = 0

    # ------------------------------------------------------------------ #

    def _record(self, kind: str, **kwargs: Any) -> RecordedCall:
        call = RecordedCall(kind=kind, **kwargs)
        self.calls.append(call)
        return call

    def _next(self, script: list[ScriptedResponse], index: int) -> ScriptedResponse:
        if index < len(script):
            return script[index]
        if self.fallback is not None:
            return self.fallback
        raise AssertionError(f"脚本响应不足：第 {index + 1} 次调用没有可用的响应")

    async def stream_chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        thinking: bool = True,
        reasoning_effort: str = "high",
        max_tokens: int = 8192,
        json_object: bool = False,
        purpose: LlmPurpose = LlmPurpose.TEACH,
        timeout_seconds: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self._record(
            "stream",
            model=model,
            messages=[dict(message) for message in messages],
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            json_object=json_object,
            purpose=purpose,
            timeout_seconds=timeout_seconds,
        )
        response = self._next(self.stream_script, self.stream_calls)
        self.stream_calls += 1

        if response.error is not None:
            raise response.error

        for event in response.events:
            yield event

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        thinking: bool = True,
        reasoning_effort: str = "high",
        max_tokens: int = 8192,
        json_object: bool = False,
        purpose: LlmPurpose = LlmPurpose.TEACH,
        timeout_seconds: int | None = None,
    ) -> BlockingResult:
        self._record(
            "blocking",
            model=model,
            messages=[dict(message) for message in messages],
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            json_object=json_object,
            purpose=purpose,
            timeout_seconds=timeout_seconds,
        )
        response = self._next(self.blocking_script, self.blocking_calls)
        self.blocking_calls += 1

        if response.error is not None:
            raise response.error

        return BlockingResult(content=response.content, reasoning=response.reasoning, usage=response.usage)


def json_response(payload: str, *, usage: TokenUsage | None = None) -> ScriptedResponse:
    """构造 JSON 输出响应。"""
    return ScriptedResponse(content=payload, usage=usage)


def error_response(error: LlmError) -> ScriptedResponse:
    return ScriptedResponse(error=error)
