"""一轮执行器测试（T1.2，FR-610、FR-412）。"""

import asyncio

import pytest

from app.contracts.schemas import LlmPurpose
from app.infra.deepseek.errors import LlmError
from app.infra.deepseek.protocol import StreamEvent, TokenUsage
from app.infra.settings import AppSettings
from app.services.llm.agent import RoundExecutor, StopSignal
from app.services.llm.memory import CONTENT_KEY, REASONING_KEY, ROLE_KEY, Memory, user_message
from app.services.llm.policy import ModelPolicy
from tests.stubs import FakeLlmClient, ScriptedResponse, error_response, text_events

USAGE = TokenUsage(
    prompt_tokens=40,
    cached_prompt_tokens=16,
    cache_miss_tokens=24,
    completion_tokens=20,
    reasoning_tokens=5,
    total_tokens=60,
)


def make_executor(
    client: FakeLlmClient,
    *,
    memory: Memory | None = None,
    settings: AppSettings | None = None,
    on_usage=None,
) -> RoundExecutor:
    return RoundExecutor(
        client,
        memory if memory is not None else Memory(),
        ModelPolicy(settings or AppSettings()),
        on_usage=on_usage,
    )


# --------------------------------------------------------------------------- #
# 成功路径
# --------------------------------------------------------------------------- #


async def test_stream_commits_user_and_assistant_messages() -> None:
    client = FakeLlmClient(
        stream_script=[ScriptedResponse(events=text_events("贝叶斯", "定理", usage=USAGE, reasoning="想一下"))]
    )
    memory = Memory()
    deltas: list[StreamEvent] = []
    usage_calls: list[tuple] = []

    executor = make_executor(client, memory=memory, on_usage=lambda *args: usage_calls.append(args))
    result = await executor.stream("什么是贝叶斯定理？", on_delta=deltas.append)

    assert result.ok
    assert result.content == "贝叶斯定理"
    assert result.reasoning == "想一下"
    assert result.usage is USAGE
    assert result.interrupted is False

    # 记忆里是「用户问题 + 助手回答」两轮
    messages = memory.outbound_messages()
    assert [message[ROLE_KEY] for message in messages] == ["user", "assistant"]
    assert messages[0][CONTENT_KEY] == "什么是贝叶斯定理？"
    assert messages[1][CONTENT_KEY] == "贝叶斯定理"
    assert messages[1][REASONING_KEY] == "想一下"

    # 增量回调只推本次新增内容
    assert [event.content for event in deltas if event.content] == ["贝叶斯", "定理"]

    assert usage_calls == [(LlmPurpose.TEACH, "deepseek-flash", USAGE)]


async def test_stream_passes_policy_parameters_to_client() -> None:
    client = FakeLlmClient(stream_script=[ScriptedResponse(events=text_events("x"))])
    settings = AppSettings(thinking_enabled=True, reasoning_effort="max", max_tokens=4096, timeout_seconds=120)

    await make_executor(client, settings=settings).stream("hi", purpose=LlmPurpose.ASSESS)

    call = client.calls[0]
    assert call.model == "deepseek-flash"
    assert call.thinking is True
    assert call.reasoning_effort == "max"
    assert call.max_tokens == 4096
    assert call.timeout_seconds == 120
    assert call.purpose is LlmPurpose.ASSESS
    assert call.json_object is False


async def test_blocking_run_commits_memory() -> None:
    client = FakeLlmClient(blocking_script=[ScriptedResponse(content="非流式回答", reasoning="思考", usage=USAGE)])
    memory = Memory()

    result = await make_executor(client, memory=memory).run("问题")

    assert result.ok
    assert result.content == "非流式回答"
    assert memory.message_count == 2
    assert memory.outbound_messages()[1][REASONING_KEY] == "思考"


async def test_system_prompt_added_only_when_memory_empty() -> None:
    client = FakeLlmClient(
        stream_script=[ScriptedResponse(events=text_events("a")), ScriptedResponse(events=text_events("b"))]
    )
    memory = Memory()
    executor = make_executor(client, memory=memory)

    await executor.stream("第一问", system_prompt="你是耐心的导师")
    await executor.stream("第二问", system_prompt="你是耐心的导师")

    messages = memory.outbound_messages()
    assert sum(1 for message in messages if message[ROLE_KEY] == "system") == 1
    assert messages[0][ROLE_KEY] == "system"


# --------------------------------------------------------------------------- #
# 失败路径：记忆不得被污染（FR-610）
# --------------------------------------------------------------------------- #


async def test_failed_round_does_not_touch_memory() -> None:
    """FR-610 的核心断言：失败轮次的用户输入与半截回复都不许进记忆。"""
    client = FakeLlmClient(stream_script=[error_response(LlmError.network("连接被关闭"))])
    memory = Memory.from_protocol([[user_message("已有历史")]])
    before = memory.outbound_messages()

    result = await make_executor(client, memory=memory).stream("这次会失败")

    assert result.error is True
    assert result.error_category == "network"
    assert result.retryable is True
    assert memory.outbound_messages() == before
    assert memory.round_count == 1


async def test_failure_after_partial_output_still_does_not_commit() -> None:
    """流中途断开：已吐给用户的内容不写进记忆，避免半截回答留在上下文里。"""

    class MidStreamFailure(FakeLlmClient):
        async def stream_chat(self, **kwargs):  # type: ignore[override]
            yield StreamEvent(content="前半句")
            raise LlmError.network("流中途断开")

    memory = Memory()
    result = await make_executor(MidStreamFailure(), memory=memory).stream("问题")

    assert result.error is True
    assert memory.is_empty is True


async def test_timeout_returns_retryable_failure() -> None:
    client = FakeLlmClient(stream_script=[error_response(LlmError.timeout("超时"))])
    result = await make_executor(client).stream("问题")

    assert result.error is True
    assert result.error_category == "timeout"
    assert result.retryable is True
    assert "超时" in result.message


async def test_config_error_is_not_retryable() -> None:
    client = FakeLlmClient(stream_script=[error_response(LlmError.config("未配置 API Key"))])
    result = await make_executor(client).stream("问题")

    assert result.error is True
    assert result.error_category == "config"
    assert result.retryable is False


async def test_cancellation_does_not_commit() -> None:
    """硬取消（任务被 cancel）时不提交，保持记忆干净；异常继续向上传播。"""

    class Cancelled(FakeLlmClient):
        async def stream_chat(self, **kwargs):  # type: ignore[override]
            yield StreamEvent(content="半截")
            raise asyncio.CancelledError

    memory = Memory()
    with pytest.raises(asyncio.CancelledError):
        await make_executor(Cancelled(), memory=memory).stream("问题")

    assert memory.is_empty is True


# --------------------------------------------------------------------------- #
# 停止生成（FR-412）
# --------------------------------------------------------------------------- #


async def test_stop_event_keeps_partial_content_and_commits() -> None:
    """FR-412：停止生成要保留已生成内容并标记未完成，而不是把它丢掉。"""

    class Gated(FakeLlmClient):
        def __init__(self, stop_event: asyncio.Event) -> None:
            super().__init__()
            self.stop_event = stop_event

        async def stream_chat(self, **kwargs):  # type: ignore[override]
            yield StreamEvent(content="第一段")
            yield StreamEvent(content="第二段")
            self.stop_event.set()  # 模拟用户在两块之间点了"停止"
            yield StreamEvent(content="第三段（不应到达）")

    stop_event = asyncio.Event()
    memory = Memory()
    result = await make_executor(Gated(stop_event), memory=memory).stream("问题", stop_event=stop_event)

    assert result.interrupted is True
    assert result.content == "第一段第二段"
    assert memory.message_count == 2
    assert memory.outbound_messages()[1][CONTENT_KEY] == "第一段第二段"


async def test_stop_signal_from_another_thread() -> None:
    """停止信号可能来自 UI 线程，因此必须线程安全。"""
    signal = StopSignal()
    loop = asyncio.get_running_loop()
    event = signal.bind(loop)

    assert signal.is_set is False
    assert signal.signal() is True

    await asyncio.sleep(0)
    assert event.is_set() is True
    assert signal.is_set is True


def test_stop_signal_without_binding_is_safe() -> None:
    assert StopSignal().signal() is False


# --------------------------------------------------------------------------- #
# call_blocking：不触碰记忆
# --------------------------------------------------------------------------- #


async def test_call_blocking_does_not_modify_memory() -> None:
    client = FakeLlmClient(blocking_script=[ScriptedResponse(content='{"ok":true}', usage=USAGE)])
    memory = Memory.from_protocol([[user_message("教学历史")]])
    before = memory.outbound_messages()
    usage_calls: list[tuple] = []

    executor = make_executor(client, memory=memory, on_usage=lambda *args: usage_calls.append(args))
    content, usage = await executor.call_blocking(
        [{"role": "user", "content": "判定请求"}],
        purpose=LlmPurpose.JUDGE,
        max_tokens=512,
        json_object=True,
    )

    assert content == '{"ok":true}'
    assert usage is USAGE
    assert memory.outbound_messages() == before

    call = client.calls[0]
    assert call.json_object is True
    assert call.max_tokens == 512
    assert call.purpose is LlmPurpose.JUDGE
    assert usage_calls == [(LlmPurpose.JUDGE, "deepseek-flash", USAGE)]


async def test_usage_callback_failure_is_swallowed() -> None:
    """用量统计属于旁路，它的失败不能让一轮对话失败（对照 FR-905 的思路）。"""
    client = FakeLlmClient(stream_script=[ScriptedResponse(events=text_events("ok", usage=USAGE))])

    def broken_callback(*_args: object) -> None:
        raise RuntimeError("统计后端不可用")

    result = await make_executor(client, on_usage=broken_callback).stream("问题")

    assert result.ok
    assert result.content == "ok"


def test_with_memory_keeps_configuration() -> None:
    client = FakeLlmClient()
    executor = RoundExecutor(client, Memory(), ModelPolicy(AppSettings()), tools_enabled=True)

    other = executor.with_memory(Memory.from_protocol([[user_message("x")]]))

    assert other.memory is not executor.memory
    assert other.memory.message_count == 1
    assert other.tools_enabled is True
