"""一轮对话执行器（T1.2，FR-610、FR-412；工具调用 T4.6，FR-714）。

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

工具调用（FR-714）也发生在**同一个工作副本**上：assistant 的 `tool_calls` 消息与随后的
`tool` 结果消息都只写进副本，最终正文成功后才整体提交。中间轮失败时整轮不提交，
不会留下"有 tool_calls 却没有对应 tool 结果"的非法历史。

两条官方限制在这里的落点：

1. **思考模式不支持 `tool_choice="required"` 与指定具体工具**——执行器从不下发
   `tool_choice`，只把 `tools` 交给协议层（见 :mod:`app.infra.llm.protocol`）；
2. **DeepSeek 在携带 `tools` 时要求历史轮次的 `reasoning_content` 必须回传**——因此只要持有工具，
   :attr:`RoundExecutor.tools_enabled` 就为真，:func:`build_outbound_messages` 会按接口类型
   决定是否保留思维链（OpenAI 兼容服务不要求，且可能拒收，见 :mod:`app.infra.llm.providers`）。
   这两件事必须同时成立，否则就是"带 tools 却不回传思维链"的自相矛盾状态，
   所以由 :attr:`tools_enabled` 的计算属性兜住，而不是让调用方记得传两个参数。
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.contracts.schemas import LlmPurpose
from app.infra.llm.client import LlmClientProtocol
from app.infra.llm.errors import LlmError
from app.infra.llm.protocol import StreamEvent, TokenUsage, ToolCall
from app.infra.logging.queue_logger import CHANNEL_LLM, get_logger
from app.services.llm.context_policy import build_outbound_messages
from app.services.llm.memory import (
    Memory,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from app.services.llm.policy import ModelPolicy
from app.services.llm.result import RoundResult
from app.services.llm.tools import ToolRegistry

DeltaCallback = Callable[[StreamEvent], None]

UsageCallback = Callable[[LlmPurpose, str, TokenUsage | None, int | None], None]
"""用量回调：``(用途, 模型, 用量, 耗时毫秒)``。

耗时随用量一起回调，是因为完成报告（FR-417）要报"累计耗时"，而"一次调用花了多久"
只有执行器知道——客户端只负责发请求，不关心统计。
"""

MAX_TOOL_ROUNDS = 3
"""工具执行轮次上限。

不设上限的风险很具体：模型可能反复调用同一个工具（参数每次略有不同），
于是整轮对话永不结束、额度被烧光。到达上限后再发一次**不带 tools** 的请求，
逼它把已有信息组织成正文。
"""


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.monotonic() - started) * 1000))


def tool_kwargs(tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """把 ``tools`` 组装成"只在需要时传"的关键字参数。

    一律传 ``tools=None`` 会改变既有桩客户端/第三方实现的调用签名兼容性；
    不携带工具的请求本就不该出现这个字段（官方不接受 ``"tools": null``）。
    """
    return {"tools": tools} if tools else {}


def _merge_usage(left: TokenUsage | None, right: TokenUsage | None) -> TokenUsage | None:
    """合并多轮调用的用量：一次工具调用链实际发生了多次计费，只记最后一轮会漏账。"""
    if left is None:
        return right
    if right is None:
        return left
    return TokenUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        cached_prompt_tokens=left.cached_prompt_tokens + right.cached_prompt_tokens,
        cache_miss_tokens=left.cache_miss_tokens + right.cache_miss_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


@dataclass
class _RoundCapture:
    """单次模型调用的产出，用于判断"这是中间轮还是最终轮"。"""

    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage | None = None
    interrupted: bool = False

    @property
    def content_text(self) -> str:
        return "".join(self.content)

    @property
    def reasoning_text(self) -> str:
        return "".join(self.reasoning)


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
        echo_reasoning: bool = True,
        conversation_id: str | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self._client = client
        self._memory = memory
        self._policy = policy
        self._on_usage = on_usage
        self._tools_enabled = tools_enabled
        self._echo_reasoning = echo_reasoning
        self._conversation_id = conversation_id
        self._tools = tools
        self._log = get_logger(CHANNEL_LLM)

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def tools_enabled(self) -> bool:
        """是否按"携带 tools"的规则处理上下文（FR-719）。

        持有工具就必然为真：请求里带了 `tools` 却剔除 `reasoning_content`，
        正是官方明确不允许的组合（见模块文档）。
        """
        return self._tools_enabled or self._tools is not None

    @property
    def tools(self) -> ToolRegistry | None:
        return self._tools

    @property
    def policy(self) -> ModelPolicy:
        """当前用途策略（结构化输出通道需要它来取默认 max_tokens）。"""
        return self._policy

    def with_tools(self, tools: ToolRegistry | None) -> "RoundExecutor":
        """挂上/摘掉工具，返回新执行器（记忆与其余配置沿用）。"""
        return RoundExecutor(
            self._client,
            self._memory,
            self._policy,
            on_usage=self._on_usage,
            tools_enabled=self._tools_enabled,
            echo_reasoning=self._echo_reasoning,
            conversation_id=self._conversation_id,
            tools=tools,
        )

    def with_memory(self, memory: Memory) -> "RoundExecutor":
        """换一份记忆复用同一套配置（教学/测评/复习各持自己的记忆）。"""
        return RoundExecutor(
            self._client,
            memory,
            self._policy,
            on_usage=self._on_usage,
            tools_enabled=self._tools_enabled,
            echo_reasoning=self._echo_reasoning,
            conversation_id=self._conversation_id,
            tools=self._tools,
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

        启用工具时，一次"轮"内可能发生多次模型调用（工具调用 → 结果 → 再调用）。
        最终正文照常提交；中间轮若也生成了正文，**丢弃并记日志**——
        "我先查一下档案"这种话对用户没有价值，展示出来反而像卡住了。
        """
        working = self._prepare(user_prompt, system_prompt)
        choice = self._policy.resolve(purpose)
        tool_specs = self._tools.specs() if self._tools is not None else None
        # 带 tools 时正文先暂存、等本轮确认没有 tool_calls 再推给前端；
        # 不带 tools 时保持既有的逐块实时推送（这是默认路径，行为不能变）
        defer_content = tool_specs is not None

        def flush_content(text: str) -> None:
            if on_delta is not None and defer_content and text:
                on_delta(StreamEvent(content=text))

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: TokenUsage | None = None
        interrupted = False
        started = time.monotonic()
        tool_rounds = 0

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    interrupted = True
                    break

                # 到达上限后不再给工具：逼模型把已有信息组织成正文，避免无限循环
                use_tools = tool_specs if (tool_specs is not None and tool_rounds < MAX_TOOL_ROUNDS) else None
                messages = build_outbound_messages(
                    working,
                    tools_enabled=self.tools_enabled,
                    echo_reasoning=self._echo_reasoning,
                )
                capture = _RoundCapture()

                async for event in self._client.stream_chat(
                    model=choice.model,
                    messages=messages,
                    thinking=choice.thinking,
                    reasoning_effort=choice.reasoning_effort,
                    max_tokens=choice.max_tokens,
                    purpose=purpose,
                    timeout_seconds=choice.timeout_seconds,
                    **tool_kwargs(use_tools),
                ):
                    if stop_event is not None and stop_event.is_set():
                        capture.interrupted = True
                        break

                    if event.usage is not None:
                        capture.usage = event.usage
                    if event.is_usage_only:
                        continue

                    if event.reasoning:
                        capture.reasoning.append(event.reasoning)
                    if event.content:
                        capture.content.append(event.content)
                    if event.tool_calls:
                        capture.tool_calls.extend(event.tool_calls)

                    if on_delta is not None:
                        if not defer_content:
                            on_delta(event)
                        elif event.reasoning:
                            # 思维链不是正文，实时推送不会泄露中间轮的过渡语
                            on_delta(StreamEvent(reasoning=event.reasoning))

                usage = _merge_usage(usage, capture.usage)
                if capture.interrupted:
                    interrupted = True
                    content_parts.extend(capture.content)
                    reasoning_parts.extend(capture.reasoning)
                    flush_content(capture.content_text)
                    break

                if not capture.tool_calls:
                    content_parts.extend(capture.content)
                    reasoning_parts.extend(capture.reasoning)
                    flush_content(capture.content_text)
                    break

                if use_tools is None:
                    # 没带 tools 却仍返回 tool_calls：协议异常。执行它/再请求都可能死循环
                    self._log.warning(
                        f"未携带 tools 的请求仍返回 tool_calls，按空正文收尾 purpose={purpose.value} "
                        f"count={len(capture.tool_calls)}"
                    )
                    break

                if capture.content_text.strip():
                    self._log.info(
                        f"丢弃工具中间轮的正文 purpose={purpose.value} chars={len(capture.content_text)}"
                    )
                self._apply_tool_calls(working, capture)
                tool_rounds += 1
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
        tool_specs = self._tools.specs() if self._tools is not None else None

        started = time.monotonic()
        usage: TokenUsage | None = None
        reasoning_parts: list[str] = []
        tool_rounds = 0
        try:
            while True:
                use_tools = tool_specs if (tool_specs is not None and tool_rounds < MAX_TOOL_ROUNDS) else None
                messages = build_outbound_messages(
                    working,
                    tools_enabled=self.tools_enabled,
                    echo_reasoning=self._echo_reasoning,
                )
                result = await self._client.chat(
                    model=choice.model,
                    messages=messages,
                    thinking=choice.thinking,
                    reasoning_effort=choice.reasoning_effort,
                    max_tokens=choice.max_tokens,
                    purpose=purpose,
                    timeout_seconds=choice.timeout_seconds,
                    **tool_kwargs(use_tools),
                )
                usage = _merge_usage(usage, result.usage)
                reasoning_parts = [result.reasoning] if result.reasoning else []

                if not result.tool_calls:
                    break
                if use_tools is None:
                    self._log.warning(
                        f"未携带 tools 的请求仍返回 tool_calls，按空正文收尾 purpose={purpose.value} "
                        f"count={len(result.tool_calls)}"
                    )
                    break
                if result.content.strip():
                    self._log.info(f"丢弃工具中间轮的正文 purpose={purpose.value} chars={len(result.content)}")

                capture = _RoundCapture(
                    content=[result.content] if result.content else [],
                    reasoning=[result.reasoning] if result.reasoning else [],
                    tool_calls=list(result.tool_calls),
                    usage=result.usage,
                )
                self._apply_tool_calls(working, capture)
                tool_rounds += 1
        except LlmError as exc:
            return self._fail(exc, purpose, choice.model, usage, _elapsed_ms(started))

        return self._commit(
            working,
            result.content,
            reasoning_parts,
            usage,
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

    def _apply_tool_calls(self, working: Memory, capture: _RoundCapture) -> None:
        """把中间轮的 ``assistant(tool_calls)`` 与工具结果写进**本轮工作副本**。

        只写副本、不写正式记忆：中间轮之后如果模型调用失败，整轮不提交，
        否则会留下"有 tool_calls 却没有对应 tool 结果"的非法历史——下一轮请求会被服务端拒绝。
        """
        working.add_message_to_last_round(
            assistant_message(
                capture.content_text or None,
                capture.reasoning_text or None,
                [call.to_payload() for call in capture.tool_calls],
            )
        )
        for call in capture.tool_calls:
            working.add_message_to_last_round(tool_message(call.id, self._execute_tool(call)))

    def _execute_tool(self, call: ToolCall) -> str:
        """执行一次工具调用；**任何错误都变成给模型看的结果文本**（见 tools 模块说明）。"""
        if self._tools is None:
            return "错误：当前会话没有可用的工具。"
        try:
            result = self._tools.execute(call)
        except Exception as exc:  # 兜底：execute 本身保证不抛，这里防的是将来改坏
            self._log.warning(f"工具执行异常（已转为错误文本） name={call.name} {type(exc).__name__}: {exc}")
            result = f"错误：工具 {call.name} 执行失败（{type(exc).__name__}）。请换一种做法或先读档案。"
        self._log.info(f"工具调用 name={call.name} id={call.id} chars={len(result)}")
        return result

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
