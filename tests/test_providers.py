"""OpenAI 兼容接口与供应商能力档案（P6）。

这一层要守住的核心事实是：**同一份协议，不同服务的可选能力不同**。

- 通用 OpenAI 兼容接口不认 `thinking`，也不回传 `prompt_cache_hit_tokens`；
- DeepSeek 认 `thinking`、回传缓存命中字段、有余额接口、空闲半价，且带 tools 时必须回传思维链。

用例按"档案 → 请求体 → 响应解析 → 端到端行为"四层组织，避免只测其中一层就看不出错配。
"""

import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from app.infra.llm.errors import LlmError, LlmErrorCategory
from app.infra.llm.protocol import (
    SamplingOptions,
    TokenUsage,
    build_chat_request,
    extract_delta,
    parse_blocking_response,
)
from app.infra.llm.providers import (
    DEFAULT_PROVIDER,
    ProviderKind,
    profile_for,
    provider_options,
)
from tests.conftest import ClientFactory

OPENAI = profile_for(ProviderKind.OPENAI)
DEEPSEEK = profile_for(ProviderKind.DEEPSEEK)

BASE_MESSAGES = [{"role": "user", "content": "hi"}]


def _request(profile: Any, **overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": "some-model",
        "messages": BASE_MESSAGES,
        "stream": False,
        "thinking": True,
        "reasoning_effort": "high",
        "max_tokens": 512,
        "profile": profile,
    }
    params.update(overrides)
    return build_chat_request(**params)


# --------------------------------------------------------------------------- #
# 档案本身
# --------------------------------------------------------------------------- #


def test_unknown_or_empty_provider_falls_back_to_the_default_profile() -> None:
    """配置损坏不得让应用不可用（FR-902 的同一原则）：认不出的取值按默认处理。"""
    assert profile_for(None).kind is DEFAULT_PROVIDER
    assert profile_for("").kind is DEFAULT_PROVIDER
    assert profile_for("some-new-gateway").kind is DEFAULT_PROVIDER
    # 字符串与枚举都接受：设置从库里读出来是字符串
    assert profile_for("openai").kind is ProviderKind.OPENAI
    assert profile_for(ProviderKind.OPENAI).kind is ProviderKind.OPENAI


def test_provider_options_expose_capabilities_for_the_settings_page() -> None:
    options = {item["value"]: item for item in provider_options()}
    assert set(options) == {"deepseek", "openai"}

    deepseek_caps = options["deepseek"]["capabilities"]
    openai_caps = options["openai"]["capabilities"]
    assert deepseek_caps["thinking_switch"] is True
    assert deepseek_caps["balance"] is True
    assert deepseek_caps["off_peak_pricing"] is True
    # 通用接口既没有思考开关，也没有余额与空闲折扣
    assert openai_caps["thinking_switch"] is False
    assert openai_caps["balance"] is False
    assert openai_caps["off_peak_pricing"] is False
    assert openai_caps["cache_stats"] is True  # 通过 prompt_tokens_details 仍能拿到命中数


def test_effort_is_mapped_into_the_range_each_service_accepts() -> None:
    """本项目有 max，OpenAI 没有：越界取值会让请求直接 400。"""
    assert DEEPSEEK.map_effort("max") == "max"
    assert OPENAI.map_effort("max") == "high"
    assert OPENAI.map_effort("low") == "low"
    # 未知档位按 high 处理，而不是把脏值原样发出去
    assert OPENAI.map_effort("turbo") == "high"


# --------------------------------------------------------------------------- #
# 请求体
# --------------------------------------------------------------------------- #


def test_openai_profile_omits_thinking_and_sends_mapped_effort() -> None:
    body = _request(OPENAI)
    assert "thinking" not in body
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 512


def test_openai_profile_drops_effort_when_thinking_is_off() -> None:
    body = _request(OPENAI, thinking=False)
    assert "thinking" not in body
    assert "reasoning_effort" not in body


def test_deepseek_profile_keeps_thinking_switch() -> None:
    body = _request(DEEPSEEK)
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"


def test_dropped_params_are_not_sent_again() -> None:
    """客户端发现接口不认某个参数后会把它放进 drop，后续请求不得再带上。"""
    body = _request(OPENAI, drop=frozenset({"reasoning_effort"}), json_object=True)
    assert "reasoning_effort" not in body
    assert body["response_format"] == {"type": "json_object"}

    without_json = _request(OPENAI, drop=frozenset({"response_format"}), json_object=True)
    assert "response_format" not in without_json


def test_max_tokens_field_can_be_renamed_for_gateways_that_require_it() -> None:
    body = _request(OPENAI, max_tokens_field="max_completion_tokens")
    assert body["max_completion_tokens"] == 512
    assert "max_tokens" not in body


def test_sampling_is_still_sent_for_openai_compatible_services() -> None:
    body = _request(OPENAI, sampling=SamplingOptions(temperature=0.3, top_p=0.8))
    assert (body["temperature"], body["top_p"]) == (0.3, 0.8)
    assert "presence_penalty" not in body


# --------------------------------------------------------------------------- #
# 响应解析
# --------------------------------------------------------------------------- #


def test_openai_profile_accepts_the_reasoning_field_name() -> None:
    """OpenRouter 系用 ``reasoning``，vLLM 系用 ``reasoning_content``，两种都要认。"""
    chunk = {"choices": [{"delta": {"content": "正文", "reasoning": "想一下"}}]}
    assert extract_delta(chunk, profile=OPENAI) == ("正文", "想一下")
    # 没传档案时按 DeepSeek 口径，只认 reasoning_content（既有行为不变）
    assert extract_delta({"choices": [{"delta": {"reasoning": "x"}}]}) == (None, None)


def test_openai_usage_reads_cached_tokens_from_details_and_derives_the_miss() -> None:
    usage = TokenUsage.from_usage_dict(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 800},
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
        profile=OPENAI,
    )
    assert usage is not None
    assert usage.cached_prompt_tokens == 800
    # 未命中 = 总数 - 命中：不推算的话这部分会被当成"全部命中"，成本凭空少一半
    assert usage.cache_miss_tokens == 200
    assert usage.reasoning_tokens == 10


def test_deepseek_usage_keeps_explicit_cache_fields() -> None:
    usage = TokenUsage.from_usage_dict(
        {
            "prompt_tokens": 100,
            "prompt_cache_hit_tokens": 64,
            "prompt_cache_miss_tokens": 36,
        },
        profile=DEEPSEEK,
    )
    assert usage is not None
    assert (usage.cached_prompt_tokens, usage.cache_miss_tokens) == (64, 36)


def test_usage_without_cache_info_is_all_miss_for_openai() -> None:
    """没有缓存字段时不能凭空"命中"：全部按未命中算，宁可高估不要低估。"""
    usage = TokenUsage.from_usage_dict({"prompt_tokens": 300}, profile=OPENAI)
    assert usage is not None
    assert (usage.cached_prompt_tokens, usage.cache_miss_tokens) == (0, 300)


def test_deepseek_profile_still_understands_the_openai_usage_shape() -> None:
    """网关可能把 DeepSeek 的缓存字段改写成了 OpenAI 写法。

    这时若只认 DeepSeek 字段，命中与未命中都会是 0，输入 token 既不计命中也不计未命中，
    成本只剩输出部分——少算一半以上，而且界面上看不出异常。
    """
    usage = TokenUsage.from_usage_dict(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 900},
        },
        profile=DEEPSEEK,
    )
    assert usage is not None
    assert usage.cached_prompt_tokens == 900
    assert usage.cache_miss_tokens == 100


def test_usage_never_loses_prompt_tokens_between_the_two_buckets() -> None:
    """不变量：命中 + 未命中 == 输入总数（有显式未命中字段时以字段为准）。"""
    shapes = [
        ({"prompt_tokens": 500}, DEEPSEEK),
        ({"prompt_tokens": 500}, OPENAI),
        ({"prompt_tokens": 500, "prompt_cache_hit_tokens": 200, "prompt_cache_miss_tokens": 300}, DEEPSEEK),
        ({"prompt_tokens": 500, "prompt_tokens_details": {"cached_tokens": 380}}, OPENAI),
    ]
    for raw, profile in shapes:
        usage = TokenUsage.from_usage_dict(raw, profile=profile)
        assert usage is not None
        assert usage.cached_prompt_tokens + usage.cache_miss_tokens == 500, (raw, profile.kind, usage)


def test_blocking_response_parses_openai_shaped_payload() -> None:
    result = parse_blocking_response(
        {
            "choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
        profile=OPENAI,
    )
    assert result.content == '{"ok": true}'
    assert result.reasoning is None
    assert result.usage is not None and result.usage.total_tokens == 15


# --------------------------------------------------------------------------- #
# 客户端：参数降级的闭环
# --------------------------------------------------------------------------- #


async def test_client_retries_once_without_a_param_the_endpoint_rejects(make_client: ClientFactory) -> None:
    """网关回 400 并点出参数名时，自动去掉该参数重发一次，而不是把 400 抛给用户。"""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "Unsupported parameter: 'reasoning_effort'"}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "好"}}], "usage": {}},
        )

    client = make_client(handler, provider_getter=lambda: "openai")
    result = await client.chat(
        model="gw-model",
        messages=BASE_MESSAGES,
        thinking=True,
        reasoning_effort="high",
    )

    assert result.content == "好"
    assert len(seen) == 2
    assert "reasoning_effort" in seen[0]
    assert "reasoning_effort" not in seen[1]
    # 只降级一次：同一参数不会被反复尝试
    assert client._drops == {"reasoning_effort"}


async def test_client_switches_to_max_completion_tokens_when_the_gateway_demands_it(
    make_client: ClientFactory,
) -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "max_tokens" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead."
                    }
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "好"}}], "usage": {}},
        )

    client = make_client(handler, provider_getter=lambda: "openai")
    result = await client.chat(model="o-series", messages=BASE_MESSAGES, max_tokens=256)

    assert result.content == "好"
    assert "max_tokens" in seen[0]
    assert seen[1]["max_completion_tokens"] == 256


async def test_a_real_input_error_is_not_retried_as_a_degradation(make_client: ClientFactory) -> None:
    """模型名写错也是 400，但没有点出参数名——不能当降级重试，否则用户要多等一轮。"""
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": "Model not found"}})

    client = make_client(handler, provider_getter=lambda: "openai")
    with pytest.raises(LlmError) as exc:
        await client.chat(model="nope", messages=BASE_MESSAGES)

    assert exc.value.category is LlmErrorCategory.BAD_REQUEST
    assert len(calls) == 1


async def test_balance_is_unsupported_on_openai_profile(make_client: ClientFactory) -> None:
    client = make_client(lambda request: httpx.Response(200, json={}), provider_getter=lambda: "openai")
    with pytest.raises(LlmError) as exc:
        await client.get_balance()
    assert exc.value.category is LlmErrorCategory.CONFIG


async def test_models_list_tolerates_ids_without_metadata(make_client: ClientFactory) -> None:
    """OpenAI 的 /models 只有 id：元数据留空即可，不能因此解析失败。"""
    payload = {"object": "list", "data": [{"id": "gpt-x", "object": "model"}, {"id": "gpt-y"}]}
    client = make_client(lambda request: httpx.Response(200, json=payload), provider_getter=lambda: "openai")

    models = await client.list_models()

    assert [model.id for model in models] == ["gpt-x", "gpt-y"]
    assert models[0].context_window is None
    assert models[0].effort_levels == ()


# --------------------------------------------------------------------------- #
# 思维链回传与计价：两处 DeepSeek 专有规则
# --------------------------------------------------------------------------- #


def test_reasoning_is_only_echoed_when_the_provider_requires_it() -> None:
    from app.services.llm.context_policy import build_outbound_messages
    from app.services.llm.memory import Memory, assistant_message, user_message

    memory = Memory()
    memory.add_round()
    memory.add_message_to_last_round(user_message("问题"))
    memory.add_message_to_last_round(assistant_message("回答", reasoning="思考过程"))

    deepseek_out = build_outbound_messages(memory, tools_enabled=True, echo_reasoning=True)
    openai_out = build_outbound_messages(memory, tools_enabled=True, echo_reasoning=False)

    assert any("reasoning_content" in message for message in deepseek_out)
    assert all("reasoning_content" not in message for message in openai_out)
    # 正文一个字节都不能少：剔除的只是思维链
    assert [m.get("content") for m in deepseek_out] == [m.get("content") for m in openai_out]


def test_off_peak_discount_follows_the_provider_profile() -> None:
    from datetime import UTC, datetime

    from app.services.llm.cost import estimate_cost

    # 2026-10-21 是周三，UTC 02:00 = 北京 10:00，属高峰
    peak = datetime(2026, 10, 21, 2, 0, tzinfo=UTC)
    off_peak = datetime(2026, 10, 21, 14, 0, tzinfo=UTC)  # 北京 22:00

    table = {"m": {"cache_hit": 1.0, "cache_miss": 2.0, "output": 4.0}}
    deepseek = estimate_cost(model="m", cache_miss_tokens=1_000_000, at=off_peak, price_table=table)
    openai = estimate_cost(
        model="m",
        cache_miss_tokens=1_000_000,
        at=off_peak,
        price_table=table,
        off_peak_multiplier=OPENAI.off_peak_multiplier,
    )
    peak_cost = estimate_cost(model="m", cache_miss_tokens=1_000_000, at=peak, price_table=table)

    assert deepseek.amount == pytest.approx(peak_cost.amount * 0.5)
    # 非 DeepSeek 接口没有空闲折扣：夜间与高峰同价，否则金额会少算一半
    assert openai.amount == pytest.approx(peak_cost.amount)
    assert openai.off_peak_tokens > 0


# --------------------------------------------------------------------------- #
# 设置在设置页往返
# --------------------------------------------------------------------------- #


def _put_settings(client: Any, **changes: object) -> dict[str, Any]:
    current = client.get("/api/settings").json()["settings"]
    read_only = {"api_key", "model_plan", "access_token_enabled", "read_only_fields"}
    payload = {key: value for key, value in current.items() if key not in read_only}
    payload.update(changes)
    response = client.put("/api/settings", json=payload)
    assert response.status_code == 200, response.text
    return response.json()["settings"]


def test_switching_provider_round_trips_and_updates_capabilities(api_client: Any) -> None:
    saved = _put_settings(api_client, provider="openai", base_url="https://api.openai.com/v1")

    assert saved["provider"] == "openai"
    assert saved["provider_capabilities"]["thinking_switch"] is False
    assert saved["provider_capabilities"]["balance"] is False

    # 落库后才叫"生效"：重新读一遍设置，不能只是这一次响应里对
    again = api_client.get("/api/settings").json()["settings"]
    assert again["provider"] == "openai"
    assert api_client.get("/api/health").json()["model"]["provider"] == "openai"

    back = _put_settings(api_client, provider="deepseek")
    assert back["provider_capabilities"]["thinking_switch"] is True


def test_balance_endpoint_explains_why_it_is_unavailable(api_client: Any) -> None:
    _put_settings(api_client, provider="openai")

    body = api_client.get("/api/balance").json()

    assert body["ok"] is False
    assert body["category"] == "unsupported"
    assert "余额" in body["message"]


def test_invalid_provider_value_is_rejected_before_it_reaches_the_database(api_client: Any) -> None:
    current = api_client.get("/api/settings").json()["settings"]
    read_only = {"api_key", "model_plan", "access_token_enabled", "read_only_fields"}
    payload = {key: value for key, value in current.items() if key not in read_only}
    payload["provider"] = "not-a-provider"

    assert api_client.put("/api/settings", json=payload).status_code == 422


@pytest.fixture
def api_client(env: dict[str, str]) -> Sequence[Any]:
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(env)) as client:
        yield client
