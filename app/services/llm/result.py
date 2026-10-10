"""一轮对话的结果类型（FR-610、FR-412）。

沿用 ReferenceCode ``AgentResponse`` 的语义：**失败不抛异常，而是返回带错误标志的结果**，
使"失败轮次不提交记忆"成为一个返回值判断，而不是散落各处的异常处理。
"""

from dataclasses import dataclass, field

from app.infra.deepseek.protocol import TokenUsage

TIMEOUT_MESSAGE = "请求超时，可点击重试或调大超时时间。"


@dataclass
class RoundResult:
    """一次请求的结果。"""

    content: str = ""
    reasoning_chunks: list[str] = field(default_factory=list)
    usage: TokenUsage | None = None
    error: bool = False
    error_category: str | None = None
    retryable: bool = False
    message: str = ""
    interrupted: bool = False

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_chunks)

    @property
    def ok(self) -> bool:
        return not self.error

    @classmethod
    def success(
        cls,
        content: str,
        reasoning_chunks: list[str] | None = None,
        usage: TokenUsage | None = None,
        *,
        interrupted: bool = False,
    ) -> "RoundResult":
        return cls(
            content=content,
            reasoning_chunks=list(reasoning_chunks or []),
            usage=usage,
            interrupted=interrupted,
        )

    @classmethod
    def failure(cls, message: str, *, category: str | None = None, retryable: bool = False) -> "RoundResult":
        return cls(error=True, message=message, error_category=category, retryable=retryable)

    @classmethod
    def timeout(cls) -> "RoundResult":
        return cls.failure(TIMEOUT_MESSAGE, category="timeout", retryable=True)

    def to_error_payload(self) -> dict[str, object]:
        """转成 SSE 错误事件载荷（不含敏感信息）。"""
        return {
            "category": self.error_category or "unknown",
            "message": self.message,
            "retryable": self.retryable,
        }
