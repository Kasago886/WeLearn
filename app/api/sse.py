"""SSE 输出设施（p2-api-sse，FR-704、FR-412）。

**为什么用队列 + 后台任务**：服务层的方法是协程（``await service.answer(...)``），
而 HTTP 响应要逐帧 ``yield``。若等服务返回再一次性发送，流式就名存实亡了。
因此这里把服务跑在后台任务里，增量回调把事件推进队列，响应生成器随取随发。

错误统一转成 ``error`` 事件而不是 HTTP 状态码：SSE 的头一旦发出就无法改状态码，
而且前端需要的是"发生了什么 + 能不能重试"。真正的请求级错误（主题不存在等）在建立
流之前就抛出 HTTPException，不会走到这里。
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from app.contracts.events import (
    ErrorPayload,
    SseEventType,
    encode_frame,
)
from app.infra.llm.errors import LlmError
from app.infra.llm.protocol import StreamEvent
from app.infra.logging.queue_logger import CHANNEL_APP, get_logger
from app.services.learning.planning import PlanningError
from app.services.learning.state import AnswerNotAllowedError
from app.services.learning.teaching import TeachingError

DeltaCallback = Callable[[StreamEvent], None]
Turn = Any

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # 关闭 Nginx 等反向代理的缓冲，否则流式会被攒成一整块
    "X-Accel-Buffering": "no",
}

_log = get_logger(CHANNEL_APP)


class TurnStream:
    """把一次"服务调用 + 流式增量"转成 SSE 帧序列。

    使用方式::

        stream = TurnStream(lambda on_delta: teaching.start(subject_id, on_delta=on_delta))
        frames = stream.frames(postprocess=lambda turn: ...)
    """

    def __init__(self, worker: Callable[[DeltaCallback], Awaitable[Any]]) -> None:
        self._worker = worker
        self._emitted = False

    async def frames(
        self,
        *,
        postprocess: Callable[[Any], list[str]] | None = None,
    ) -> AsyncIterator[str]:
        """产出 SSE 帧。``postprocess`` 用于在收尾前追加领域事件（如 path/options）。"""
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        def on_delta(event: StreamEvent) -> None:
            queue.put_nowait(("delta", event))

        async def run() -> None:
            try:
                turn = await self._worker(on_delta)
                queue.put_nowait(("turn", turn))
            except Exception as exc:  # 统一转成 error 事件，避免 SSE 流里抛出未处理异常
                queue.put_nowait(("error", exc))
            finally:
                queue.put_nowait(("end", None))

        task = asyncio.create_task(run())
        try:
            while True:
                kind, payload = await queue.get()

                if kind == "end":
                    break

                if kind == "delta":
                    frame = self._delta_frame(payload)
                    if frame is not None:
                        yield frame
                    continue

                if kind == "error":
                    yield self._error_frame(payload)
                    continue

                # kind == "turn"
                for frame in self._turn_frames(payload, postprocess):
                    yield frame
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    # ------------------------------------------------------------------ #

    def _delta_frame(self, event: StreamEvent) -> str | None:
        if event.reasoning:
            self._emitted = True
            return encode_frame(SseEventType.REASONING, {"text": event.reasoning})
        if event.content:
            self._emitted = True
            return encode_frame(SseEventType.TOKEN, {"text": event.content})
        return None

    def _turn_frames(
        self,
        turn: Any,
        postprocess: Callable[[Any], list[str]] | None,
    ) -> list[str]:
        frames: list[str] = []

        error_message = getattr(turn, "error_message", "") or ""
        if error_message:
            frames.append(
                encode_frame(
                    SseEventType.ERROR,
                    ErrorPayload(
                        category=getattr(turn, "error_category", None) or "unknown",
                        message=error_message,
                        retryable=bool(getattr(turn, "retryable", False)),
                    ),
                )
            )
        elif not self._emitted:
            # 本轮没有任何流式增量（例如"跳过节点""直接进入下一节点"这类纯本地动作，
            # 或小结不是流式生成的），把完整文本作为一次 token 推送。
            # 反之若已有增量，就不能再发一次完整文本，否则前端会重复渲染。
            text = getattr(turn, "text", "") or ""
            if text:
                frames.append(encode_frame(SseEventType.TOKEN, {"text": text}))
                self._emitted = True

        if postprocess is not None:
            frames.extend(postprocess(turn))

        frames.append(
            encode_frame(
                SseEventType.DONE,
                {
                    "conversation_id": None,
                    "usage": None,
                },
            )
        )
        return frames

    def _error_frame(self, exc: Exception) -> str:
        """把异常转成面向用户的错误事件。"""
        if isinstance(exc, AnswerNotAllowedError):
            _log.info(f"拒绝作答/操作: {exc.reason}")
            return encode_frame(
                SseEventType.ERROR,
                ErrorPayload(category=exc.reason, message=str(exc), retryable=False),
            )

        if isinstance(exc, LlmError):
            _log.warning(f"LLM 调用失败: {exc.category.value} {exc.detail}")
            return encode_frame(
                SseEventType.ERROR,
                ErrorPayload(category=exc.category.value, message=exc.user_message, retryable=exc.retryable),
            )

        if isinstance(exc, (PlanningError, TeachingError, KeyError)):
            _log.warning(f"流程错误: {type(exc).__name__}: {exc}")
            message = str(exc).strip("\"'")
            return encode_frame(
                SseEventType.ERROR,
                ErrorPayload(category="workflow", message=message, retryable=False),
            )

        _log.error(f"未预期的流式错误: {type(exc).__name__}: {exc}")
        return encode_frame(
            SseEventType.ERROR,
            ErrorPayload(
                category="unknown",
                message="服务出现异常，请稍后重试；若持续出现请查看日志。",
                retryable=True,
            ),
        )


def sse_response(stream: TurnStream, *, postprocess: Callable[[Any], list[str]] | None = None) -> Any:
    """构造 SSE ``StreamingResponse``。"""
    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        stream.frames(postprocess=postprocess),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
