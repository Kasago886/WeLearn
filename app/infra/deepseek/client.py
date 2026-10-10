"""DeepSeek HTTP 客户端（FR-701、FR-704、FR-709、FR-711、FR-712）。

设计要点：

- **连接配置按需读取**：base_url / API Key 通过 getter 注入，设置页改完即时生效，无需重建客户端；
- **重试边界**：只在"尚未产出任何增量"时重试。已经吐给用户的流不重试，避免重复内容（FR-412）；
- **错误统一为** :class:`LlmError`：状态码映射、网络异常、协议异常都不外泄原始堆栈；
- **可注入**：实现 :class:`LlmClientProtocol`，测试可传入桩实现或 httpx.MockTransport（NFR-115）。
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.contracts.schemas import LlmPurpose
from app.infra.deepseek.errors import LlmError
from app.infra.deepseek.protocol import (
    BALANCE_PATH,
    CHAT_COMPLETIONS_PATH,
    MODELS_PATH,
    BlockingResult,
    ModelInfo,
    StreamEvent,
    build_chat_request,
    build_endpoint,
    extract_delta,
    extract_usage,
    parse_blocking_response,
    parse_models_response,
)
from app.infra.deepseek.sse import SseEventParser
from app.infra.logging.queue_logger import CHANNEL_LLM, get_logger
from app.infra.redaction import Redactor

DEFAULT_BACKOFF_SECONDS = 0.5
CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_TEST_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class ConnectionTestResult:
    """"测试连接"结果（FR-709）。"""

    ok: bool
    message: str
    category: str | None = None
    latency_ms: int = 0


class LlmClientProtocol(Protocol):
    """LLM 客户端契约，便于测试注入桩实现（NFR-115）。"""

    def stream_chat(
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
    ) -> AsyncIterator[StreamEvent]: ...

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
    ) -> BlockingResult: ...


class DeepSeekClient:
    """DeepSeek 的 OpenAI 兼容客户端。"""

    def __init__(
        self,
        *,
        base_url_getter: Callable[[], str],
        api_key_getter: Callable[[], str | None],
        redactor: Redactor | None = None,
        max_retries: int = 2,
        timeout_seconds: int = 60,
        backoff_base_seconds: float = DEFAULT_BACKOFF_SECONDS,
        log_payload: bool = False,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url_getter = base_url_getter
        self._api_key_getter = api_key_getter
        self._redactor = redactor if redactor is not None else Redactor()
        self._max_retries = max(0, max_retries)
        self._timeout_seconds = timeout_seconds
        self._backoff_base = backoff_base_seconds
        self._log_payload = log_payload
        self._log = get_logger(CHANNEL_LLM)
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(follow_redirects=True)

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def update_policy(self, *, max_retries: int | None = None, timeout_seconds: int | None = None) -> None:
        """更新重试与超时策略（设置页保存后调用）。"""
        if max_retries is not None:
            self._max_retries = max(0, max_retries)
        if timeout_seconds is not None:
            self._timeout_seconds = timeout_seconds

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------ #
    # 流式对话
    # ------------------------------------------------------------------ #

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
        """流式对话。失败时按 :attr:`LlmError.retryable` 重试，但**仅在尚未产出增量之前**。"""
        attempt = 0
        emitted = False

        while True:
            try:
                async for event in self._stream_once(
                    model=model,
                    messages=messages,
                    thinking=thinking,
                    reasoning_effort=reasoning_effort,
                    max_tokens=max_tokens,
                    json_object=json_object,
                    purpose=purpose,
                    timeout_seconds=timeout_seconds,
                ):
                    if not event.is_usage_only:
                        emitted = True
                    yield event
                return
            except LlmError as exc:
                if emitted or not exc.retryable or attempt >= self._max_retries:
                    raise
                attempt += 1
                self._log.warning(
                    f"流式请求失败将重试（第 {attempt}/{self._max_retries} 次）: "
                    f"category={exc.category.value} detail={exc.detail}"
                )
                await self._sleep_backoff(attempt)

    async def _stream_once(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        thinking: bool,
        reasoning_effort: str,
        max_tokens: int,
        json_object: bool,
        purpose: LlmPurpose,
        timeout_seconds: int | None,
    ) -> AsyncIterator[StreamEvent]:
        url = build_endpoint(self._base_url_getter(), CHAT_COMPLETIONS_PATH)
        body = build_chat_request(
            model=model,
            messages=messages,
            stream=True,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            json_object=json_object,
        )
        self._log_request(purpose, model, messages, thinking, reasoning_effort, json_object, body)

        last_usage = None
        try:
            async with self._client.stream(
                "POST",
                url,
                json=body,
                headers=self._headers(),
                timeout=self._timeout(timeout_seconds),
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise self._error_from_response(response.status_code, raw)

                parser = SseEventParser()
                # 在同一异步循环中驱动同步解析器：忽略空行、`: keep-alive` 注释与其他字段行
                async for line in response.aiter_lines():
                    payloads = parser.feed(line)
                    if parser.finished:
                        break

                    for payload in payloads:
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError as exc:
                            # 单个数据块异常只跳过该块，不得中断整轮（FR-704、AC-28）
                            self._log.warning(f"跳过无法解析的流式数据块: {exc.msg}")
                            continue

                        usage = extract_usage(chunk)
                        if usage is not None:
                            last_usage = usage

                        content, reasoning = extract_delta(chunk)
                        if content or reasoning:
                            yield StreamEvent(content=content, reasoning=reasoning)
        except httpx.TimeoutException as exc:
            raise LlmError.timeout(f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            # 官方会在"请求发出后 10 分钟仍未开始推理"时关闭连接（FR-711）
            raise LlmError.network(f"{type(exc).__name__}: {exc}") from exc

        if last_usage is not None:
            self._log_usage(purpose, model, last_usage)
            yield StreamEvent(usage=last_usage)
        else:
            self._log.warning(f"流式响应未返回 usage（purpose={purpose.value} model={model}）")

    # ------------------------------------------------------------------ #
    # 非流式对话
    # ------------------------------------------------------------------ #

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
        """非流式对话（判定、压缩等场景）。"""
        url = build_endpoint(self._base_url_getter(), CHAT_COMPLETIONS_PATH)
        body = build_chat_request(
            model=model,
            messages=messages,
            stream=False,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            json_object=json_object,
        )
        self._log_request(purpose, model, messages, thinking, reasoning_effort, json_object, body)

        attempt = 0
        while True:
            try:
                response = await self._client.post(
                    url,
                    json=body,
                    headers=self._headers(),
                    timeout=self._timeout(timeout_seconds),
                )
            except httpx.TimeoutException as exc:
                error = LlmError.timeout(f"{type(exc).__name__}: {exc}")
            except httpx.HTTPError as exc:
                error = LlmError.network(f"{type(exc).__name__}: {exc}")
            else:
                if response.status_code >= 400:
                    error = self._error_from_response(response.status_code, response.content)
                else:
                    result = self._parse_blocking(response)
                    if result.usage is not None:
                        self._log_usage(purpose, model, result.usage)
                    return result

            if not error.retryable or attempt >= self._max_retries:
                raise error
            attempt += 1
            self._log.warning(
                f"请求失败将重试（第 {attempt}/{self._max_retries} 次）: "
                f"category={error.category.value} detail={error.detail}"
            )
            await self._sleep_backoff(attempt)

    def _parse_blocking(self, response: httpx.Response) -> BlockingResult:
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise LlmError.protocol("响应不是合法 JSON，无法解析。", detail=str(exc)) from exc

        if not isinstance(body, Mapping):
            raise LlmError.protocol("响应结构不符合预期（根节点不是对象）。")
        return parse_blocking_response(body)

    # ------------------------------------------------------------------ #
    # 辅助接口
    # ------------------------------------------------------------------ #

    async def list_models(self, *, timeout_seconds: int | None = None) -> list[ModelInfo]:
        """获取可用模型与元数据（FR-702），避免模型名写死失效。"""
        body = await self._get_json(MODELS_PATH, timeout_seconds)
        return parse_models_response(body)

    async def get_balance(self, *, timeout_seconds: int | None = None) -> Mapping[str, Any]:
        """查询账户余额（FR-710，可选能力）。"""
        return await self._get_json(BALANCE_PATH, timeout_seconds)

    async def test_connection(
        self,
        *,
        model: str,
        timeout_seconds: int | None = None,
    ) -> ConnectionTestResult:
        """最小代价的连通性测试（FR-709）。"""
        started = time.perf_counter()
        try:
            await self.chat(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                thinking=False,  # 关闭思考以缩短延迟、降低费用
                max_tokens=16,
                purpose=LlmPurpose.TEACH,
                timeout_seconds=timeout_seconds or DEFAULT_TEST_TIMEOUT_SECONDS,
            )
        except LlmError as exc:
            return ConnectionTestResult(
                ok=False,
                message=exc.user_message,
                category=exc.category.value,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        return ConnectionTestResult(
            ok=True,
            message="连接正常。",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    async def _get_json(self, path: str, timeout_seconds: int | None) -> Mapping[str, Any]:
        url = build_endpoint(self._base_url_getter(), path)
        try:
            response = await self._client.get(url, headers=self._headers(), timeout=self._timeout(timeout_seconds))
        except httpx.TimeoutException as exc:
            raise LlmError.timeout(f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LlmError.network(f"{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            raise self._error_from_response(response.status_code, response.content)

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise LlmError.protocol("响应不是合法 JSON。", detail=str(exc)) from exc

        if not isinstance(body, Mapping):
            raise LlmError.protocol("响应结构不符合预期（根节点不是对象）。")
        return body

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _headers(self) -> dict[str, str]:
        api_key = (self._api_key_getter() or "").strip()
        if not api_key:
            raise LlmError.config("尚未配置 API Key，请在设置页填写，或设置环境变量 WELEARN_API_KEY。")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _timeout(self, override: int | None) -> httpx.Timeout:
        seconds = float(override or self._timeout_seconds)
        return httpx.Timeout(seconds, connect=min(CONNECT_TIMEOUT_SECONDS, seconds))

    def _error_from_response(self, status_code: int, raw: bytes) -> LlmError:
        text = raw.decode("utf-8", errors="replace") if raw else ""
        detail = self._redactor.truncate(text, 500)
        error = LlmError.from_status(status_code, detail=detail)
        self._log.warning(
            f"LLM 请求失败 status={status_code} category={error.category.value} body={detail}"
        )
        return error

    async def _sleep_backoff(self, attempt: int) -> None:
        delay = self._backoff_base * (2 ** (attempt - 1))
        if delay > 0:
            await asyncio.sleep(delay)

    def _log_request(
        self,
        purpose: LlmPurpose,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        thinking: bool,
        reasoning_effort: str,
        json_object: bool,
        body: Mapping[str, Any],
    ) -> None:
        self._log.info(
            f"LLM 请求 purpose={purpose.value} model={model} messages={len(messages)} "
            f"thinking={thinking} effort={reasoning_effort} json={json_object}"
        )
        if self._log_payload:
            # 审计模式（FR-715）：请求体不含密钥，仍统一过一遍脱敏
            self._log.debug(f"LLM 请求体: {self._redactor.truncate(json.dumps(body, ensure_ascii=False), 4000)}")

    def _log_usage(self, purpose: LlmPurpose, model: str, usage: Any) -> None:
        self._log.info(
            f"LLM 用量 purpose={purpose.value} model={model} total={usage.total_tokens} "
            f"prompt={usage.prompt_tokens} cached={usage.cached_prompt_tokens} "
            f"miss={usage.cache_miss_tokens} completion={usage.completion_tokens} "
            f"reasoning={usage.reasoning_tokens}"
        )
