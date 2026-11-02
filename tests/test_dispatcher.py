"""调度器测试（T1.3，FR-611、FR-712）。"""

import asyncio

from app.contracts.schemas import LlmPurpose
from app.infra.llm.errors import LlmError
from app.infra.llm.protocol import StreamEvent
from app.infra.settings import AppSettings
from app.services.llm.agent import RoundExecutor
from app.services.llm.dispatcher import RoundDispatcher
from app.services.llm.memory import CONTENT_KEY, Memory
from app.services.llm.policy import ModelPolicy
from tests.stubs import FakeLlmClient, ScriptedResponse, error_response, text_events


def make_dispatcher(
    client: FakeLlmClient,
    *,
    memory: Memory | None = None,
    max_retries: int = 2,
    notify=None,
    on_error=None,
) -> RoundDispatcher:
    executor = RoundExecutor(client, memory if memory is not None else Memory(), ModelPolicy(AppSettings()))
    return RoundDispatcher(executor, max_retries=max_retries, notify=notify, on_error=on_error)


async def test_successful_submit_commits_once() -> None:
    client = FakeLlmClient(stream_script=[ScriptedResponse(events=text_events("回答"))])
    memory = Memory()

    result = await make_dispatcher(client, memory=memory).submit("问题")

    assert result.ok
    assert memory.message_count == 2
    assert client.stream_calls == 1


async def test_retryable_failure_is_retried_then_succeeds() -> None:
    client = FakeLlmClient(
        stream_script=[
            error_response(LlmError.network("连接被关闭")),
            ScriptedResponse(events=text_events("第二次成功")),
        ]
    )
    memory = Memory()
    errors: list[str] = []

    result = await make_dispatcher(client, memory=memory, on_error=lambda r: errors.append(r.message)).submit("问题")

    assert result.ok
    assert result.content == "第二次成功"
    assert client.stream_calls == 2
    assert len(errors) == 1
    # 失败那次没有污染记忆
    assert memory.message_count == 2


async def test_retry_exhaustion_notifies_and_stops() -> None:
    client = FakeLlmClient(
        stream_script=[error_response(LlmError.from_status(503)), error_response(LlmError.from_status(503))]
    )
    notifications: list[str] = []
    memory = Memory()

    result = await make_dispatcher(client, memory=memory, max_retries=1, notify=notifications.append).submit("问题")

    assert result.error is True
    assert client.stream_calls == 2  # 首次 + 1 次重试
    assert len(notifications) == 1
    assert "重试" in notifications[0]
    assert memory.is_empty is True


async def test_non_retryable_failure_is_not_retried() -> None:
    client = FakeLlmClient(stream_script=[error_response(LlmError.from_status(401))])

    result = await make_dispatcher(client, max_retries=3).submit("问题")

    assert result.error is True
    assert result.error_category == "auth"
    assert client.stream_calls == 1


async def test_busy_submission_is_rejected_explicitly() -> None:
    """FR-611/AC-17：忙碌时必须明确拒绝，而不是静默丢弃或并发改写记忆。"""
    gate = asyncio.Event()
    sent_prompts: list[str] = []

    class SlowClient(FakeLlmClient):
        async def stream_chat(self, **kwargs):  # type: ignore[override]
            sent_prompts.append(str(kwargs["messages"][-1].get("content", "")))
            yield StreamEvent(content="开始")
            await gate.wait()
            yield StreamEvent(content="结束")

    client = SlowClient()
    memory = Memory()
    dispatcher = make_dispatcher(client, memory=memory)

    first = asyncio.create_task(dispatcher.submit("第一问"))
    await asyncio.sleep(0.05)  # 让第一次进入执行状态

    assert dispatcher.busy is True
    second = await dispatcher.submit("第二问")

    assert second.error is True
    assert second.error_category == "busy"
    assert "停止生成" in second.message
    # 第二问根本没被送出去
    assert sent_prompts == ["第一问"]

    gate.set()
    assert (await first).ok is True
    # 第二问也没有被写入记忆
    assert [message[CONTENT_KEY] for message in memory.outbound_messages()] == ["第一问", "开始结束"]


async def test_dispatcher_is_reusable_after_completion() -> None:
    client = FakeLlmClient(
        stream_script=[ScriptedResponse(events=text_events("a")), ScriptedResponse(events=text_events("b"))]
    )
    dispatcher = make_dispatcher(client)

    assert (await dispatcher.submit("一")).ok
    assert dispatcher.busy is False
    assert (await dispatcher.submit("二")).ok
    assert client.stream_calls == 2


async def test_interrupted_round_is_not_treated_as_failure() -> None:
    """用户主动停止不该被当成失败去重试（否则会立刻重新开始生成，体验很怪）。"""
    stop_event = asyncio.Event()
    sent_prompts: list[str] = []

    class Stopping(FakeLlmClient):
        async def stream_chat(self, **kwargs):  # type: ignore[override]
            sent_prompts.append(str(kwargs["messages"][-1].get("content", "")))
            yield StreamEvent(content="半截")
            stop_event.set()
            yield StreamEvent(content="不会到达")

    client = Stopping()
    result = await make_dispatcher(client, max_retries=2).submit("问题", stop_event=stop_event)

    assert result.ok is True
    assert result.interrupted is True
    # 只发了一次请求，没有因为"中断"而重试
    assert sent_prompts == ["问题"]


async def test_purpose_and_system_prompt_are_forwarded() -> None:
    client = FakeLlmClient(stream_script=[ScriptedResponse(events=text_events("x"))])

    await make_dispatcher(client).submit("问题", purpose=LlmPurpose.ASSESS, system_prompt="你是诊断者")

    assert client.calls[0].purpose is LlmPurpose.ASSESS
    assert client.calls[0].messages[0]["role"] == "system"
