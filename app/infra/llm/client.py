"""OpenAI 兼容的 LLM HTTP 客户端（FR-701、FR-704、FR-709、FR-711、FR-712、FR-715、FR-718）。

协议本身是 OpenAI 的 ``/chat/completions``；DeepSeek 的额外能力（思考开关、缓存统计、
余额接口、空闲半价）由 :mod:`app.infra.llm.providers` 的档案按接口类型开启。

设计要点：

- **连接配置按需读取**：base_url / API Key / 采样参数 / 审计开关 / 接口类型都通过 getter 注入，
  设置页改完即时生效，无需重建客户端；
- **重试边界**：只在"尚未产出任何增量"时重试。已经吐给用户的流不重试，避免重复内容（FR-412）；
- **能力降级**：接口明确回 400/422 且点出某个参数名时，记下来重发一次（不带该参数），
  而不是把"这个网关不认 reasoning_effort"变成用户面前的一条报错；
- **错误统一为** :class:`LlmError`：状态码映射、网络异常、协议异常都不外泄原始堆栈；
- **可注入**：实现 :class:`LlmClientProtocol`，测试可传入桩实现或 httpx.MockTransport（NFR-115）。
"""

import asyncio
import json
import ssl
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import httpx

from app.contracts.schemas import LlmPurpose
from app.infra.llm.errors import LlmError
from app.infra.llm.protocol import (
    CHAT_COMPLETIONS_PATH,
    MODELS_PATH,
    BlockingResult,
    ModelInfo,
    SamplingOptions,
    StreamEvent,
    ToolCallAccumulator,
    build_chat_request,
    build_endpoint,
    extract_delta,
    extract_usage,
    parse_blocking_response,
    parse_models_response,
)
from app.infra.llm.providers import ProviderProfile, profile_for
from app.infra.llm.sse import SseEventParser
from app.infra.logging.queue_logger import CHANNEL_LLM, get_logger
from app.infra.redaction import Redactor

DEFAULT_BACKOFF_SECONDS = 0.5
CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_TEST_TIMEOUT_SECONDS = 30
# 审计正文的长度上限（FR-715）：审计是为了排障，不是做归档，超长正文会把日志通道撑爆
AUDIT_TEXT_LIMIT = 4000

# "该接口不认这个参数"时的降级候选：参数名 → 错误正文里的线索词。
# 只有 400/422 且正文里确实出现线索词才降级，避免把"模型名写错"这类真实错误也重试一遍。
_DEGRADABLE_PARAMS: dict[str, tuple[str, ...]] = {
    "reasoning_effort": ("reasoning_effort",),
    "response_format": ("response_format", "json_object"),
    "thinking": ("thinking",),
}
# 少数网关只认新版字段名（OpenAI 的推理模型就是这样）：
# 报错里出现它时把 max_tokens 换成 max_completion_tokens 重发一次
_ALT_MAX_TOKENS_FIELD = "max_completion_tokens"


@lru_cache(maxsize=1)
def _shared_ssl_context() -> ssl.SSLContext:
    """复用同一个默认 SSL 上下文。

    ``httpx.AsyncClient()`` 默认会在**每次构造时**重新解析 CA 证书包
    （本机实测 0.33s，占装配耗时九成以上），而证书包在一个进程的生命周期内不会变，
    没有理由每个客户端付一次。

    取值方式直接复用 httpx 的公开工厂，不抄它的内部逻辑——httpx 改行为时这里跟着变。
    """
    return httpx.create_ssl_context(verify=True)


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
        tools: Sequence[Mapping[str, Any]] | None = None,
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
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> BlockingResult: ...


class LlmClient:
    """OpenAI 兼容客户端（DeepSeek 的额外能力按接口类型开启）。"""

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
        audit_getter: Callable[[], bool] | None = None,
        sampling_getter: Callable[[], SamplingOptions] | None = None,
        provider_getter: Callable[[], str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url_getter = base_url_getter
        self._api_key_getter = api_key_getter
        self._redactor = redactor if redactor is not None else Redactor()
        self._max_retries = max(0, max_retries)
        self._timeout_seconds = timeout_seconds
        self._backoff_base = backoff_base_seconds
        self._log_payload = log_payload
        self._audit_getter = audit_getter
        self._sampling_getter = sampling_getter
        self._provider_getter = provider_getter
        # 已证实不被该接口接受的参数（按错误正文自动发现），只在内存里记，进程重启即忘
        self._drops: set[str] = set()
        self._max_tokens_field = "max_tokens"
        self._log = get_logger(CHANNEL_LLM)
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            follow_redirects=True,
            verify=_shared_ssl_context(),
        )

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def update_policy(self, *, max_retries: int | None = None, timeout_seconds: int | None = None) -> None:
        """更新重试与超时策略（设置页保存后调用）。"""
        if max_retries is not None:
            self._max_retries = max(0, max_retries)
        if timeout_seconds is not None:
            self._timeout_seconds = timeout_seconds

    def audit_enabled(self) -> bool:
        """当前是否记录完整请求体与完整响应（FR-715）。

        **取或**：运行期设置 ``audit_enabled`` 或环境变量 ``WELEARN_LOG_LLM_PAYLOAD``——
        环境变量保留为"一定能开"的兜底手段（配置文件坏了也能抓到一次现场）。
        """
        if self._log_payload:
            return True
        return bool(self._audit_getter is not None and self._audit_getter())

    def _sampling(self) -> SamplingOptions:
        """采样参数每次请求现取（FR-718）：与 base_url 一样不缓存，改了立刻生效。"""
        if self._sampling_getter is None:
            return SamplingOptions()
        return self._sampling_getter()

    def profile(self) -> ProviderProfile:
        """当前接口类型的能力档案（每次现取，切换后下一个请求即生效）。"""
        if self._provider_getter is None:
            return profile_for(None)
        return profile_for(self._provider_getter())

    def _degrade(self, error: LlmError) -> bool:
        """遇到"该接口不认这个参数"时记下来，返回 True 表示值得原样重发一次。

        只认 400/422 且错误正文明确点到参数名的情形：把"模型名写错""Key 无效"
        也当降级重试，只会让用户多等一倍时间、看到的还是同一个错。
        """
        if error.status_code not in (400, 422):
            return False

        detail = (error.detail or "").lower()
        changed: list[str] = []
        for param, hints in _DEGRADABLE_PARAMS.items():
            if param in self._drops:
                continue
            if any(hint in detail for hint in hints):
                self._drops.add(param)
                changed.append(param)

        if self._max_tokens_field != _ALT_MAX_TOKENS_FIELD and _ALT_MAX_TOKENS_FIELD in detail:
            self._max_tokens_field = _ALT_MAX_TOKENS_FIELD
            changed.append(f"max_tokens→{_ALT_MAX_TOKENS_FIELD}")

        if changed:
            self._log.warning(f"接口不支持以下参数，已自动降级并重发: {', '.join(changed)}")
            return True
        return False

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
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """流式对话。

        重试有两条边界：

        - **正文已经吐出去过就不再重试**（FR-412）：重试会让同一段话在界面上出现两次；
        - **正文一个字都没有则算失败、可以重试**：思考模式下模型有概率把输出预算全花在
          思维链上，正文为空。此时用户并没有看到任何正文，重试不会造成重复内容；
          而如果不重试，就会落一条空回复——界面上什么都不显示，看着像卡死了。

        工具调用（FR-714）与正文同等看待：只要已经产出过 tool_calls，就**不再重试**，
        否则同一批工具可能被执行两次。
        """
        attempt = 0

        while True:
            content_emitted = False
            tool_calls_emitted = False
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
                    tools=tools,
                ):
                    if event.content:
                        content_emitted = True
                    if event.tool_calls:
                        tool_calls_emitted = True
                    yield event
            except LlmError as exc:
                if content_emitted or tool_calls_emitted:
                    raise
                # 参数不被接受：降级后立刻重发，不占用重试次数（否则第一次就要用光重试）
                if self._degrade(exc):
                    continue
                error = exc
            else:
                if content_emitted or tool_calls_emitted:
                    return
                error = LlmError.empty_content(f"purpose={purpose.value} model={model}")

            if not error.retryable or attempt >= self._max_retries:
                raise error
            attempt += 1
            self._log.warning(
                f"流式请求失败将重试（第 {attempt}/{self._max_retries} 次）: "
                f"category={error.category.value} detail={error.detail}"
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
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        url = build_endpoint(self._base_url_getter(), CHAT_COMPLETIONS_PATH)
        profile = self.profile()
        body = self._body(
            model=model,
            messages=messages,
            stream=True,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            json_object=json_object,
            tools=tools,
            profile=profile,
        )
        self._log_request(purpose, model, messages, thinking, reasoning_effort, json_object, body)

        # 审计时顺带把原始流攒起来，一轮结束时整段落日志（FR-715）；
        # 不审计时一行业都不留，避免"关了开关还付一次拼接成本"
        audit_enabled = self.audit_enabled()
        audit_lines: list[str] = []

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
                accumulator = ToolCallAccumulator()
                # 在同一异步循环中驱动同步解析器：忽略空行、`: keep-alive` 注释与其他字段行
                async for line in response.aiter_lines():
                    if audit_enabled:
                        audit_lines.append(line)
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

                        usage = extract_usage(chunk, profile=profile)
                        if usage is not None:
                            last_usage = usage

                        content, reasoning = extract_delta(chunk, profile=profile)
                        if content or reasoning:
                            yield StreamEvent(content=content, reasoning=reasoning)
                        # 工具调用增量没有可展示的正文，累积到流结束一次性给出
                        accumulator.feed(chunk)
        except httpx.TimeoutException as exc:
            raise LlmError.timeout(f"{type(exc).__name__}: {exc}") from exc
        except httpx.HTTPError as exc:
            # 官方会在"请求发出后 10 分钟仍未开始推理"时关闭连接（FR-711）
            raise LlmError.network(f"{type(exc).__name__}: {exc}") from exc

        if audit_enabled:
            self._log_response(purpose, model, "\n".join(audit_lines))

        tool_calls = accumulator.finish()
        if tool_calls:
            yield StreamEvent(tool_calls=tool_calls)

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
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> BlockingResult:
        """非流式对话（判定、压缩等场景）。"""
        url = build_endpoint(self._base_url_getter(), CHAT_COMPLETIONS_PATH)
        profile = self.profile()

        attempt = 0
        while True:
            body = self._body(
                model=model,
                messages=messages,
                stream=False,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
                json_object=json_object,
                tools=tools,
                profile=profile,
            )
            if attempt == 0:
                self._log_request(purpose, model, messages, thinking, reasoning_effort, json_object, body)
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
                    if self.audit_enabled():
                        self._log_response(purpose, model, response.text)
                    result = self._parse_blocking(response, profile)
                    if result.usage is not None:
                        self._log_usage(purpose, model, result.usage)
                    return result

            # 参数不被接受：降级后立刻重发，不占用重试次数
            if self._degrade(error):
                continue

            if not error.retryable or attempt >= self._max_retries:
                raise error
            attempt += 1
            self._log.warning(
                f"请求失败将重试（第 {attempt}/{self._max_retries} 次）: "
                f"category={error.category.value} detail={error.detail}"
            )
            await self._sleep_backoff(attempt)

    def _body(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        stream: bool,
        thinking: bool,
        reasoning_effort: str,
        max_tokens: int,
        json_object: bool,
        tools: Sequence[Mapping[str, Any]] | None,
        profile: ProviderProfile,
    ) -> dict[str, Any]:
        """按当前接口类型与已发现的降级项组装请求体。"""
        return build_chat_request(
            model=model,
            messages=messages,
            stream=stream,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            json_object=json_object,
            sampling=self._sampling(),
            tools=tools,
            profile=profile,
            drop=frozenset(self._drops),
            max_tokens_field=self._max_tokens_field,
        )

    def _parse_blocking(self, response: httpx.Response, profile: ProviderProfile) -> BlockingResult:
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise LlmError.protocol("响应不是合法 JSON，无法解析。", detail=str(exc)) from exc

        if not isinstance(body, Mapping):
            raise LlmError.protocol("响应结构不符合预期（根节点不是对象）。")
        return parse_blocking_response(body, profile=profile)

    # ------------------------------------------------------------------ #
    # 辅助接口
    # ------------------------------------------------------------------ #

    async def list_models(self, *, timeout_seconds: int | None = None) -> list[ModelInfo]:
        """获取可用模型与元数据（FR-702），避免模型名写死失效。

        只有部分服务（如 DeepSeek）会在 ``/models`` 里带上下文窗口与档位；
        其余返回的只是 id 列表，元数据字段留空即可——界面按"未知"展示，不编造数字。
        """
        body = await self._get_json(MODELS_PATH, timeout_seconds)
        return parse_models_response(body)

    async def get_balance(self, *, timeout_seconds: int | None = None) -> Mapping[str, Any]:
        """查询账户余额（FR-710，可选能力）。

        余额是 DeepSeek 特有的接口；当前接口类型没有这个能力时抛配置错误，
        由 API 层翻译成"该接口不支持余额查询"，不打扰主流程。
        """
        path = self.profile().balance_path
        if path is None:
            raise LlmError.config(f"当前接口类型（{self.profile().kind.value}）不提供余额查询接口。")
        return await self._get_json(path, timeout_seconds)

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
        if self.audit_enabled():
            # 审计模式（FR-715）：请求体不含密钥，仍统一过一遍脱敏
            self._log.debug(
                f"LLM 请求体: {self._redactor.truncate(json.dumps(body, ensure_ascii=False), AUDIT_TEXT_LIMIT)}"
            )

    def _log_response(self, purpose: LlmPurpose, model: str, raw: str) -> None:
        """审计模式（FR-715）：记录完整响应。

        同样过 :class:`Redactor`：响应体本不该带密钥，但"服务端回显了什么"恰好是排障要看的东西，
        不能因为"理论上不会出现"就跳过脱敏。
        """
        self._log.debug(
            f"LLM 响应体 purpose={purpose.value} model={model} "
            f"body={self._redactor.truncate(raw, AUDIT_TEXT_LIMIT)}"
        )

    def _log_usage(self, purpose: LlmPurpose, model: str, usage: Any) -> None:
        self._log.info(
            f"LLM 用量 purpose={purpose.value} model={model} total={usage.total_tokens} "
            f"prompt={usage.prompt_tokens} cached={usage.cached_prompt_tokens} "
            f"miss={usage.cache_miss_tokens} completion={usage.completion_tokens} "
            f"reasoning={usage.reasoning_tokens}"
        )
