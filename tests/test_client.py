"""客户端测试：流式解析、重试边界、错误分类与脱敏（FR-704、FR-709、FR-711、FR-712）。

全部通过 ``httpx.MockTransport`` 离线运行，不消耗真实额度。
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from app.contracts.schemas import LlmPurpose
from app.infra.llm.client import LlmClient
from app.infra.llm.errors import LlmError, LlmErrorCategory
from app.infra.llm.protocol import SamplingOptions, StreamEvent
from app.infra.logging import queue_logger
from app.infra.logging.queue_logger import QueueLogger
from app.infra.redaction import Redactor
from tests.conftest import ClientFactory, fixture_text

API_KEY = "sk-test-abcdef123456"


def _message(text: str) -> dict[str, object]:
    return {"role": "assistant", "content": text}


def _ok_blocking(content: str = "好的") -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"index": 0, "message": _message(content), "finish_reason": "stop"}], "usage": {}},
    )


async def _collect(client: LlmClient, *, model: str = "deepseek-flash") -> list[StreamEvent]:
    return [
        event
        async for event in client.stream_chat(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            purpose=LlmPurpose.TEACH,
        )
    ]


# --------------------------------------------------------------------------- #
# 流式解析
# --------------------------------------------------------------------------- #


async def test_stream_parses_content_reasoning_and_usage(make_client: ClientFactory) -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=fixture_text("stream_normal.sse").encode("utf-8"))

    client = make_client(handler)
    events = await _collect(client)

    deltas = [e for e in events if not e.is_usage_only]
    assert [d.content for d in deltas if d.content] == ["贝叶斯定理", "用先验概率与新证据", "更新后验概率。"]
    assert deltas[0].reasoning == "用户问的是贝叶斯定理。"

    usage = events[-1].usage
    assert usage is not None
    assert usage.total_tokens == 60
    assert usage.reasoning_tokens == 12

    # 请求体符合官方协议
    body = captured[0]
    assert body["stream"] is True
    assert body["model"] == "deepseek-flash"
    assert body["thinking"] == {"type": "enabled"}
    assert "tools" not in body


async def test_stream_tolerates_keepalive_and_broken_chunk(make_client: ClientFactory) -> None:
    """AC-28：保活注释与畸形数据块都不得中断整轮对话。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=fixture_text("stream_keepalive_and_broken.sse").encode("utf-8"))

    client = make_client(handler)
    events = await _collect(client)

    contents = [e.content for e in events if e.content]
    assert contents == ["第一块", "第二块"]
    assert events[-1].usage is not None
    assert events[-1].usage.cached_prompt_tokens == 64


async def test_stream_without_usage_is_tolerated(make_client: ClientFactory) -> None:
    stream = 'data: {"choices":[{"index":0,"delta":{"content":"x"}}]}\n\ndata: [DONE]\n\n'
    client = make_client(lambda _r: httpx.Response(200, content=stream.encode("utf-8")))

    events = await _collect(client)
    assert [e.content for e in events] == ["x"]


# --------------------------------------------------------------------------- #
# 错误分类与脱敏
# --------------------------------------------------------------------------- #


async def test_non_retryable_error_is_not_retried(make_client: ClientFactory) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, json={"error": {"message": "Authentication Fails"}})

    client = make_client(handler)

    with pytest.raises(LlmError) as exc:
        await client.chat(model="deepseek-flash", messages=[])

    assert exc.value.category is LlmErrorCategory.AUTH
    assert exc.value.retryable is False
    assert calls == 1


async def test_balance_error_maps_to_actionable_message(make_client: ClientFactory) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, content=fixture_text("error_402.json").encode("utf-8"))

    client = make_client(handler)

    with pytest.raises(LlmError) as exc:
        await client.chat(model="deepseek-flash", messages=[])

    assert exc.value.category is LlmErrorCategory.BALANCE
    assert "余额" in exc.value.user_message


async def test_retryable_error_is_retried_then_succeeds(make_client: ClientFactory) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return _ok_blocking("恢复成功")

    client = make_client(handler, max_retries=2)
    result = await client.chat(model="deepseek-flash", messages=[])

    assert result.content == "恢复成功"
    assert calls == 2


async def test_retry_is_exhausted(make_client: ClientFactory) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, json={"error": {"message": "boom"}})

    client = make_client(handler, max_retries=2)

    with pytest.raises(LlmError) as exc:
        await client.chat(model="deepseek-flash", messages=[])

    assert exc.value.category is LlmErrorCategory.SERVER_ERROR
    assert calls == 3  # 首次 + 2 次重试


async def test_error_detail_is_redacted(make_client: ClientFactory, ) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"echo": f"Bearer {API_KEY}"})

    client = make_client(handler, redactor=Redactor())

    with pytest.raises(LlmError) as exc:
        await client.chat(model="deepseek-flash", messages=[])

    assert API_KEY not in exc.value.detail
    assert "****" in exc.value.detail


async def test_missing_api_key_fails_with_config_error(make_client: ClientFactory) -> None:
    client = make_client(lambda _r: _ok_blocking(), api_key_getter=lambda: None)

    with pytest.raises(LlmError) as exc:
        await client.chat(model="deepseek-flash", messages=[])

    assert exc.value.category is LlmErrorCategory.CONFIG
    assert "API Key" in exc.value.user_message


# --------------------------------------------------------------------------- #
# 重试边界：已经吐出的内容不重试（FR-412）
# --------------------------------------------------------------------------- #


async def test_stream_retries_before_first_delta(monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory) -> None:
    attempts = 0

    def fake_stream_once(**_kwargs: object) -> AsyncIterator[StreamEvent]:
        async def generator() -> AsyncIterator[StreamEvent]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise LlmError.network("连接被关闭")
            yield StreamEvent(content="第二次成功")

        return generator()

    client = make_client(lambda _r: _ok_blocking(), max_retries=2)
    monkeypatch.setattr(client, "_stream_once", fake_stream_once)

    events = await _collect(client)

    assert attempts == 2
    assert [e.content for e in events] == ["第二次成功"]


async def test_stream_does_not_retry_after_first_delta(
    monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory
) -> None:
    attempts = 0

    def fake_stream_once(**_kwargs: object) -> AsyncIterator[StreamEvent]:
        async def generator() -> AsyncIterator[StreamEvent]:
            nonlocal attempts
            attempts += 1
            yield StreamEvent(content="已展示一半")
            raise LlmError.network("流中途断开")

        return generator()

    client = make_client(lambda _r: _ok_blocking(), max_retries=2)
    monkeypatch.setattr(client, "_stream_once", fake_stream_once)

    received: list[str] = []
    with pytest.raises(LlmError):
        async for event in client.stream_chat(model="deepseek-flash", messages=[]):
            if event.content:
                received.append(event.content)

    assert attempts == 1
    assert received == ["已展示一半"]


# --------------------------------------------------------------------------- #
# 空正文：思考模式下模型有概率只给思维链、不给正文
# --------------------------------------------------------------------------- #


def _content_stream(text: str) -> AsyncIterator[StreamEvent]:
    async def generator() -> AsyncIterator[StreamEvent]:
        yield StreamEvent(content=text)

    return generator()


def _reasoning_only(text: str = "让我想想……") -> AsyncIterator[StreamEvent]:
    async def generator() -> AsyncIterator[StreamEvent]:
        yield StreamEvent(reasoning=text)

    return generator()


async def test_reasoning_only_reply_is_retried(
    monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory
) -> None:
    """思维链不算正文：一个字都没给就该重试，否则会落一条空回复。"""
    attempts = 0

    def fake_stream_once(**_kwargs: object) -> AsyncIterator[StreamEvent]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return _reasoning_only()
        return _content_stream("第二次给出了正文")

    client = make_client(lambda _r: _ok_blocking(), max_retries=2)
    monkeypatch.setattr(client, "_stream_once", fake_stream_once)

    events = await _collect(client)

    assert attempts == 2
    assert [e.content for e in events if e.content] == ["第二次给出了正文"]


async def test_reasoning_only_reply_raises_empty_content(
    monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory
) -> None:
    """重试用尽仍是空正文 → 可重试的 empty_content 错误，而不是静默成功。"""
    attempts = 0

    def fake_stream_once(**_kwargs: object) -> AsyncIterator[StreamEvent]:
        nonlocal attempts
        attempts += 1
        return _reasoning_only()

    client = make_client(lambda _r: _ok_blocking(), max_retries=2)
    monkeypatch.setattr(client, "_stream_once", fake_stream_once)

    with pytest.raises(LlmError) as exc:
        await _collect(client)

    assert attempts == 3  # 首次 + 2 次重试
    assert exc.value.category is LlmErrorCategory.EMPTY_CONTENT
    assert exc.value.retryable is True


async def test_empty_reply_does_not_block_retry_like_content_does(
    monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory
) -> None:
    """对照：吐过正文后再失败不重试，只有思维链时允许重试。"""
    attempts = 0

    def fake_stream_once(**_kwargs: object) -> AsyncIterator[StreamEvent]:
        async def generator() -> AsyncIterator[StreamEvent]:
            nonlocal attempts
            attempts += 1
            yield StreamEvent(reasoning="先想了想")
            raise LlmError.network("流中途断开")

        return generator()

    client = make_client(lambda _r: _ok_blocking(), max_retries=1)
    monkeypatch.setattr(client, "_stream_once", fake_stream_once)

    with pytest.raises(LlmError) as exc:
        await _collect(client)

    assert attempts == 2  # 思维链不阻止重试
    assert exc.value.category is LlmErrorCategory.NETWORK


async def test_timeout_is_retryable(make_client: ClientFactory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("官方 10 分钟未开始推理时也会这样断开", request=request)

    client = make_client(handler, max_retries=0, timeout_seconds=1)

    with pytest.raises(LlmError) as exc:
        await client.chat(model="deepseek-flash", messages=[])

    assert exc.value.category is LlmErrorCategory.TIMEOUT
    assert exc.value.retryable is True


# --------------------------------------------------------------------------- #
# 辅助接口
# --------------------------------------------------------------------------- #


async def test_test_connection_reports_success(make_client: ClientFactory) -> None:
    client = make_client(lambda _r: _ok_blocking("pong"))
    result = await client.test_connection(model="deepseek-flash")

    assert result.ok is True
    assert result.message == "连接正常。"
    assert result.latency_ms >= 0


async def test_test_connection_reports_failure(make_client: ClientFactory) -> None:
    client = make_client(lambda _r: httpx.Response(401, json={"error": {"message": "bad key"}}))
    result = await client.test_connection(model="deepseek-flash")

    assert result.ok is False
    assert result.category == "auth"
    assert "Key" in result.message


async def test_list_models(make_client: ClientFactory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, content=fixture_text("models.json").encode("utf-8"))

    client = make_client(handler)
    models = await client.list_models()

    assert [m.id for m in models] == ["deepseek-flash", "deepseek-v4-pro"]


async def test_get_balance(make_client: ClientFactory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/user/balance")
        return httpx.Response(200, json={"is_available": True, "balance_infos": [{"currency": "CNY"}]})

    client = make_client(handler)
    balance = await client.get_balance()

    assert balance["is_available"] is True


async def test_update_policy_changes_retry_behavior(make_client: ClientFactory) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": {"message": "busy"}})

    client = make_client(handler, max_retries=0)
    client.update_policy(max_retries=1)

    with pytest.raises(LlmError):
        await client.chat(model="deepseek-flash", messages=[])

    assert calls == 2


# --------------------------------------------------------------------------- #
# 采样参数下发（FR-718）
# --------------------------------------------------------------------------- #


async def test_sampling_params_come_from_the_getter(make_client: ClientFactory) -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=fixture_text("stream_normal.sse").encode("utf-8"))

    client = make_client(handler, sampling_getter=lambda: SamplingOptions(temperature=0.3, top_p=0.95))
    await _collect(client)

    assert captured[0]["temperature"] == 0.3
    assert captured[0]["top_p"] == 0.95


async def test_sampling_params_default_when_no_getter(make_client: ClientFactory) -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=fixture_text("stream_normal.sse").encode("utf-8"))

    client = make_client(handler)
    await _collect(client)

    assert captured[0]["temperature"] == 1.0
    assert captured[0]["top_p"] == 1.0


async def test_sampling_getter_is_read_per_request(make_client: ClientFactory) -> None:
    """每次请求现取（FR-718）：设置页改完不需要重建客户端。"""
    captured: list[dict[str, object]] = []
    current = {"temperature": 0.2}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=fixture_text("stream_normal.sse").encode("utf-8"))

    client = make_client(handler, sampling_getter=lambda: SamplingOptions(temperature=current["temperature"]))
    await _collect(client)
    current["temperature"] = 1.7
    await _collect(client)

    assert [body["temperature"] for body in captured] == [0.2, 1.7]


# --------------------------------------------------------------------------- #
# 请求审计（FR-715）
# --------------------------------------------------------------------------- #


def _audit_logger(tmp_path: Path, redactor: Redactor) -> QueueLogger:
    return QueueLogger(tmp_path, level="DEBUG", redactor=redactor)


async def test_audit_logs_full_request_and_response_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory
) -> None:
    """开关打开时能看到完整请求体与响应体，且两者都经过脱敏（FR-715、FR-708）。"""
    redactor = Redactor()
    redactor.register(API_KEY)
    logger = _audit_logger(tmp_path, redactor)
    monkeypatch.setattr(queue_logger, "_LOGGER", logger)
    try:
        # 响应体里刻意回显 Key（模拟服务端把请求头回显出来），以验证脱敏真的生效
        body = fixture_text("stream_normal.sse")

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=(f": {API_KEY}\n{body}").encode())

        client = make_client(handler, redactor=redactor, audit_getter=lambda: True)
        await _collect(client)
        assert logger.flush()
    finally:
        logger.close()

    text = (tmp_path / "llm.log").read_text(encoding="utf-8")
    assert "LLM 请求体" in text
    assert '"role": "user"' in text  # 审计记的是完整请求体，不只是"发生过一次请求"
    assert "LLM 响应体" in text
    assert "贝叶斯定理" in text  # 响应正文也在日志里
    assert API_KEY not in text


async def test_audit_is_silent_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory
) -> None:
    """默认关闭：不审计时一行正文都不写，避免正常运行时把日志刷满。"""
    redactor = Redactor()
    redactor.register(API_KEY)
    logger = _audit_logger(tmp_path, redactor)
    monkeypatch.setattr(queue_logger, "_LOGGER", logger)
    try:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=fixture_text("stream_normal.sse").encode("utf-8"))

        client = make_client(handler, redactor=redactor)
        await _collect(client)
        assert logger.flush()
    finally:
        logger.close()

    text = (tmp_path / "llm.log").read_text(encoding="utf-8")
    assert "LLM 请求 purpose=" in text  # 常规日志仍在
    assert "LLM 请求体" not in text
    assert "LLM 响应体" not in text


async def test_environment_variable_forces_audit_on(make_client: ClientFactory) -> None:
    """环境变量是与运行期设置**取或**的强制手段：配置文件坏了也能抓到一次现场。"""
    client = make_client(lambda _r: _ok_blocking(), log_payload=True)

    assert client.audit_enabled() is True


async def test_audit_getter_is_read_per_request(make_client: ClientFactory) -> None:
    enabled = {"value": False}
    client = make_client(lambda _r: _ok_blocking(), audit_getter=lambda: enabled["value"])

    assert client.audit_enabled() is False
    enabled["value"] = True
    assert client.audit_enabled() is True


async def test_blocking_audit_logs_response_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_client: ClientFactory
) -> None:
    """非流式（判定、压缩）也必须记到响应体，否则审计只覆盖一半调用。"""
    redactor = Redactor()
    logger = _audit_logger(tmp_path, redactor)
    monkeypatch.setattr(queue_logger, "_LOGGER", logger)
    try:
        client = make_client(lambda _r: _ok_blocking("判定通过"), redactor=redactor, audit_getter=lambda: True)
        await client.chat(model="deepseek-flash", messages=[{"role": "user", "content": "hi"}])
        assert logger.flush()
    finally:
        logger.close()

    text = (tmp_path / "llm.log").read_text(encoding="utf-8")
    assert "LLM 响应体" in text
    assert "判定通过" in text
