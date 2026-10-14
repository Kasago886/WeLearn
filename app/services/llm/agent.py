"""一轮对话执行器（T1.2，FR-610、FR-412）。

核心不变量：**在副本上执行，成功后才提交**。

```text
working = memory.snapshot()       # 副本
working += 用户输入
调用模型（流式或非流式）
成功 → memory.replace_with(working)   # 提交点：原子替换
失败 → 丢弃副本，正式记忆一个字节都没变
```

这样"失败轮次污染上下文"这个 bug 在结构上不可能发生（对应 `tests/test_agent.py`
里的失败用例），也让重试变得廉价——重试只需重跑副本，无需回滚。

工具调用（FR-714）不在此实现，属于 P2 范围。
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

from app.contracts.schemas import LlmPurpose
from app.infra.deepseek.client import LlmClientProtocol
from app.infra.deepseek.errors import LlmError
from app.infra.deepseek.protocol import StreamEvent, TokenUsage
from app.infra.logging.queue_logger import CHANNEL_LLM, get_logger
from app.services.llm.context_policy import build_outbound_messages
from app.services.llm.memory import Memory, assistant_message, system_message, user_message
from app.services.llm.policy import ModelPolicy
from app.services.llm.result import RoundResult

DeltaCallback = Callable[[StreamEvent], None]

UsageCallback = Callable[[LlmPurpose, str, TokenUsage | None, int | None], None]
"""用量回调：``(用途, 模型, 用量, 耗时毫秒)``。

耗时随用量一起回调，是因为完成报告（FR-417）要报"累计耗时"，而"一次调用花了多久"
只有执行器知道——客户端只负责发请求，不关心统计。
"""


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))



class RoundExecutor:
    """对一份记忆执行一轮对话。"""

    def __init__(
        self,
        client: LlmClientProtocol,
        memory: Memory,
        policy: ModelPolicy,
        *,
        on_usage: UsageCallback | None = None,
        tools_enabled: bool = False,
        conversation_id: str | None = None,
    ) -> None:
        self._client = client
        self._memory = memory
        self._policy = policy
        self._on_usage = on_usage
        self._tools_enabled = tools_enabled
        self._conversation_id = conversation_id
        self._log = get_logger(CHANNEL_LLM)

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def tools_enabled(self) -> bool:
        """是否按"携带 tools"的规则处理上下文（FR-719）。"""
        return self._tools_enabled

    @property
    def policy(self) -> ModelPolicy:
        """当前用途策略（结构化输出通道需要它来取默认 max_tokens）。"""
        return self._policy

    def with_memory(self, memory: Memory) -> "RoundExecutor":
        """换一份记忆复用同一套配置（教学/测评/复习各持自己的记忆）。"""
        return RoundExecutor(
            self._client,
            memory,
            self._policy,
            on_usage=self._on_usage,
            tools_enabled=self._tools_enabled,
            conversation_id=self._conversation_id,
        )

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #

    async def stream(
        self,
        user_prompt: str | None = None,
        *,
        on_delta: DeltaCallback | None = None,
        purpose: LlmPurpose = LlmPurpose.TEACH,
        stop_event: asyncio.Event | None = None,
        system_prompt: str | None = None,
    ) -> RoundResult:
        """流式执行一轮。

        ``stop_event`` 用于"停止生成"（FR-412）：置位后**优雅中断**，
        已生成内容会被提交并标记 ``interrupted=True``，避免用户已看到的内容丢失。
        与之相对，任务被硬取消（``CancelledError``）时不提交，保持记忆干净。
        """
        working = self._prepare(user_prompt, system_prompt)
        choice = self._policy.resolve(purpose)
        messages = build_outbound_messages(working, tools_enabled=self._tools_enabled)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: TokenUsage | None = None
        interrupted = False
        started = time.monotonic()

        try:
            async for event in self._client.stream_chat(
                model=choice.model,
                messages=messages,
                thinking=choice.thinking,
                reasoning_effort=choice.reasoning_effort,
                max_tokens=choice.max_tokens,
                purpose=purpose,
                timeout_seconds=choice.timeout_seconds,
            ):
                if stop_event is not None and stop_event.is_set():
                    interrupted = True
                    break

                if event.is_usage_only:
                    usage = event.usage
                    continue

                if event.reasoning:
                    reasoning_parts.append(event.reasoning)
                if event.content:
                    content_parts.append(event.content)

                if on_delta is not None:
                    on_delta(event)
        except LlmError as exc:
            return self._fail(exc, purpose, choice.model, usage, _elapsed_ms(started))
        except asyncio.CancelledError:
            # 硬取消：不提交，交由上层决定如何收尾
            self._log.warning(f"一轮对话被取消，未提交记忆 purpose={purpose.value}")
            raise

        duration_ms = _elapsed_ms(started)
        content = "".join(content_parts)
        if not content.strip():
            # 用户按了停止、却恰好赶上流自己结束：循环里已经没机会再观察 stop_event，
            # 这里补判一次。**只在没吐出正文时才补**——回答已经完整给出时不该被标成"未完成"。
            if stop_event is not None and stop_event.is_set():
                interrupted = True
            else:
                # 空正文绝不提交：客户端已经为这种情况自动重试过一轮，还能走到这里说明它确实
                # 什么都没给。落一条空回复比报错更糟——界面上用户看不到任何东西，只会以为卡死。
                return self._fail(
                    LlmError.empty_content(f"purpose={purpose.value} model={choice.model}"),
                    purpose,
                    choice.model,
                    usage,
                    duration_ms,
                )

        return self._commit(
            working,
            content,
            reasoning_parts,
            usage,
            purpose=purpose,
            model=choice.model,
            interrupted=interrupted,
            duration_ms=duration_ms,
        )

    async def run(
        self,
        user_prompt: str | None = None,
        *,
        purpose: LlmPurpose = LlmPurpose.TEACH,
        system_prompt: str | None = None,
    ) -> RoundResult:
        """非流式执行一轮（判定、压缩等场景）。"""
        working = self._prepare(user_prompt, system_prompt)
        choice = self._policy.resolve(purpose)
        messages = build_outbound_messages(working, tools_enabled=self._tools_enabled)

        started = time.monotonic()
        try:
            result = await self._client.chat(
                model=choice.model,
                messages=messages,
                thinking=choice.thinking,
                reasoning_effort=choice.reasoning_effort,
                max_tokens=choice.max_tokens,
                purpose=purpose,
                timeout_seconds=choice.timeout_seconds,
            )
        except LlmError as exc:
            return self._fail(exc, purpose, choice.model, None, _elapsed_ms(started))

        reasoning_parts = [result.reasoning] if result.reasoning else []
        return self._commit(
            working,
            result.content,
            reasoning_parts,
            result.usage,
            purpose=purpose,
            model=choice.model,
            interrupted=False,
            duration_ms=_elapsed_ms(started),
        )

    async def call_blocking(
        self,
        messages: list[dict[str, Any]],
        *,
        purpose: LlmPurpose,
        max_tokens: int | None = None,
        json_object: bool = False,
    ) -> tuple[str, TokenUsage | None]:
        """**不触碰记忆**的一次调用，用于判定与压缩（T1.4、T1.6）。

        这些调用自带完整上下文，不应污染教学记忆——因此它刻意绕过副本/提交流程。
        """
        choice = self._policy.resolve(purpose)
        started = time.monotonic()
        result = await self._client.chat(
            model=choice.model,
            messages=messages,
            thinking=choice.thinking,
            reasoning_effort=choice.reasoning_effort,
            max_tokens=max_tokens or choice.max_tokens,
            json_object=json_object,
            purpose=purpose,
            timeout_seconds=choice.timeout_seconds,
        )
        self._record_usage(purpose, choice.model, result.usage, _elapsed_ms(started))
        return result.content, result.usage

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #

    def _prepare(self, user_prompt: str | None, system_prompt: str | None) -> Memory:
        working = self._memory.snapshot()
        if system_prompt and working.is_empty:
            working.add_round().add(system_message(system_prompt))
        if user_prompt:
            working.add_round().add(user_message(user_prompt))
        return working

    def _commit(
        self,
        working: Memory,
        content: str,
        reasoning_parts: list[str],
        usage: TokenUsage | None,
        *,
        purpose: LlmPurpose,
        model: str,
        interrupted: bool,
        duration_ms: int | None = None,
    ) -> RoundResult:
        """提交点：写入 assistant 消息并整体替换正式记忆。"""
        reasoning = "".join(reasoning_parts)
        working.add_message_to_last_round(assistant_message(content, reasoning or None))
        self._memory.replace_with(working)
        self._record_usage(purpose, model, usage, duration_ms)

        if interrupted:
            self._log.info(f"生成被用户中断，已保留部分内容 purpose={purpose.value} chars={len(content)}")

        return RoundResult.success(content, reasoning_parts, usage, interrupted=interrupted)

    def _fail(
        self,
        error: LlmError,
        purpose: LlmPurpose,
        model: str,
        usage: TokenUsage | None,
        duration_ms: int | None = None,
    ) -> RoundResult:
        """失败路径：**不提交记忆**（FR-610）。"""
        self._log.warning(
            f"一轮对话失败，未提交记忆 purpose={purpose.value} {error.category.value} detail={error.detail}"
        )
        # 失败的调用同样消耗了额度，用量仍需记录
        self._record_usage(purpose, model, usage, duration_ms)
        return RoundResult.failure(
            error.user_message,
            category=error.category.value,
            retryable=error.retryable,
        )

    def _record_usage(
        self,
        purpose: LlmPurpose,
        model: str,
        usage: TokenUsage | None,
        duration_ms: int | None = None,
    ) -> None:
        if self._on_usage is None:
            return
        try:
            self._on_usage(purpose, model, usage, duration_ms)
        except Exception as exc:  # 统计属于旁路，失败不得影响对话
            self._log.warning(f"用量记录失败（已忽略）: {type(exc).__name__}: {exc}")


class StopSignal:
    """把"停止生成"从 UI 线程传到事件循环的轻量信号。

    比直接取消任务更温和：执行器在两块增量之间检查它，从而**保留已生成内容**（FR-412）。
    """

    def __init__(self) -> None:
        self._event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> asyncio.Event:
        self._loop = loop
        self._event = asyncio.Event()
        return self._event

    def signal(self) -> bool:
        """从任意线程置位；未绑定时返回 False。"""
        event, loop = self._event, self._loop
        if event is None or loop is None:
            return False
        if loop.is_closed():
            return False
        loop.call_soon_threadsafe(event.set)
        return True

    @property
    def is_set(self) -> bool:
        return self._event is not None and self._event.is_set()
