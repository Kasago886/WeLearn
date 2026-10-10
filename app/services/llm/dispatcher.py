"""一轮对话的调度器（T1.3，FR-611、FR-712）。

职责（沿用 ReferenceCode ``KasaAgent`` 的成熟设计）：

1. **互斥**：同一时刻只允许一轮对话在执行，避免并发改写同一份记忆；
2. **重试**：失败且可重试时按上限重试，指数退避交给 HTTP 客户端；
3. **成功才提交**：由 :class:`~app.services.llm.agent.RoundExecutor` 保证，调度器无需回滚。

与 ReferenceCode 的一处刻意差异：原实现"忙碌时静默忽略用户输入"，而 FR-611/AC-17 要求
**明确告知用户**，因此这里返回一个 ``category="busy"`` 的失败结果。
"""

import asyncio
from collections.abc import Callable

from app.contracts.schemas import LlmPurpose
from app.infra.logging.queue_logger import CHANNEL_LLM, get_logger
from app.services.llm.agent import DeltaCallback, RoundExecutor
from app.services.llm.result import RoundResult

NotifyCallback = Callable[[str], None]
ErrorCallback = Callable[[RoundResult], None]


class RoundDispatcher:
    """串行化并重试某份记忆上的对话。"""

    def __init__(
        self,
        executor: RoundExecutor,
        *,
        max_retries: int = 2,
        notify: NotifyCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        self._executor = executor
        self._max_retries = max(0, max_retries)
        self._notify = notify
        self._on_error = on_error
        self._running = False
        self._log = get_logger(CHANNEL_LLM)

    @property
    def busy(self) -> bool:
        return self._running

    async def submit(
        self,
        user_prompt: str | None = None,
        *,
        on_delta: DeltaCallback | None = None,
        purpose: LlmPurpose = LlmPurpose.TEACH,
        stop_event: asyncio.Event | None = None,
        system_prompt: str | None = None,
    ) -> RoundResult:
        """提交一轮输入。

        忙碌时立即返回 ``category="busy"`` 的失败结果，不排队、不静默丢弃（FR-611）。

        这里刻意**不使用锁**：若在锁内判断忙碌状态，第二次提交会等待锁而不是被拒绝——
        那正是 FR-611 想避免的"输入被吞掉"（会表现为界面卡住）。单事件循环下
        "先判断、再同步置位、然后才出现第一个 await"已足够互斥。
        """
        if self._running:
            self._log.warning("调度器正忙，拒绝本次输入（FR-611：明确拒绝而非静默丢弃）")
            return RoundResult.failure(BUSY_HINT, category="busy", retryable=True)

        self._running = True
        try:
            return await self._run_with_retry(
                user_prompt,
                on_delta=on_delta,
                purpose=purpose,
                stop_event=stop_event,
                system_prompt=system_prompt,
            )
        finally:
            self._running = False

    async def _run_with_retry(
        self,
        user_prompt: str | None,
        *,
        on_delta: DeltaCallback | None,
        purpose: LlmPurpose,
        stop_event: asyncio.Event | None,
        system_prompt: str | None,
    ) -> RoundResult:
        attempt = 0
        while True:
            result = await self._executor.stream(
                user_prompt,
                on_delta=on_delta,
                purpose=purpose,
                stop_event=stop_event,
                system_prompt=system_prompt,
            )

            if result.ok:
                return result

            if self._on_error is not None:
                self._on_error(result)

            # 用户主动中断不算失败
            if result.interrupted:
                return result

            if not result.retryable or attempt >= self._max_retries:
                self._notify_failure(result, attempt)
                return result

            attempt += 1
            self._notify_retry(result, attempt)
            await asyncio.sleep(self._backoff_seconds(attempt))

    def _backoff_seconds(self, attempt: int) -> float:
        """与客户端退避区分开：这里只做"整轮重试"的间隔，避免叠加成长时间静默。"""
        return min(0.2 * (2 ** (attempt - 1)), 2.0)

    def _notify_retry(self, result: RoundResult, attempt: int) -> None:
        self._log.warning(
            f"第 {attempt}/{self._max_retries} 次重试（category={result.error_category} message={result.message}）"
        )

    def _notify_failure(self, result: RoundResult, attempts: int) -> None:
        message = f"重试 {attempts} 次后仍失败（max_retries={self._max_retries}）: {result.message}"
        self._log.error(message)
        if self._notify is not None:
            self._notify(message)


BUSY_HINT = "正在生成上一轮内容，请稍候或先点击停止生成。"
