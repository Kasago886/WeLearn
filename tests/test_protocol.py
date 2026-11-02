"""协议层测试：请求体构建与响应解析（FR-701、FR-705、FR-716、FR-717）。"""

import json

import pytest

from app.infra.llm.errors import LlmError, LlmErrorCategory
from app.infra.llm.protocol import (
    SamplingOptions,
    TokenUsage,
    build_chat_request,
    build_endpoint,
    extract_delta,
    extract_usage,
    parse_blocking_response,
    parse_models_response,
)
from tests.conftest import fixture_json


def test_endpoint_keeps_subpath_and_requires_scheme() -> None:
    assert build_endpoint("https://api.deepseek.com", "/chat/completions") == (
        "https://api.deepseek.com/chat/completions"
    )
    assert build_endpoint("https://api.deepseek.com/", "/chat/completions") == (
        "https://api.deepseek.com/chat/completions"
    )
    # 保留子路径（部分网关需要 /v1）
    assert build_endpoint("https://host/v1", "/chat/completions") == "https://host/v1/chat/completions"


def test_endpoint_rejects_missing_or_invalid_base_url() -> None:
    with pytest.raises(LlmError) as empty:
        build_endpoint("", "/chat/completions")
    assert empty.value.category is LlmErrorCategory.CONFIG

    with pytest.raises(LlmError) as bad:
        build_endpoint("api.deepseek.com", "/chat/completions")
    assert bad.value.category is LlmErrorCategory.CONFIG


def test_request_body_includes_thinking_and_effort() -> None:
    body = build_chat_request(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        thinking=True,
        reasoning_effort="high",
        max_tokens=1024,
    )
    assert body["thinking"] == {"type": "enabled"}
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 1024
    assert body["stream"] is True


def test_request_body_omits_effort_when_thinking_disabled() -> None:
    body = build_chat_request(
        model="deepseek-flash",
        messages=[],
        stream=False,
        thinking=False,
        reasoning_effort="high",
        max_tokens=256,
    )
    assert body["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in body


def test_request_body_omits_tools_when_empty() -> None:
    """官方语义：无 tool 时 tool_choice 默认 none；但不接受 "tools": null。"""
    base = {
        "model": "deepseek-flash",
        "messages": [],
        "stream": False,
        "thinking": False,
        "reasoning_effort": "high",
        "max_tokens": 256,
    }
    assert "tools" not in build_chat_request(**base, tools=None)
    assert "tools" not in build_chat_request(**base, tools=[])

    with_tools = build_chat_request(**base, tools=[{"type": "function", "function": {"name": "f"}}])
    assert len(with_tools["tools"]) == 1  # type: ignore[arg-type]


def test_request_body_sets_response_format_for_json_output() -> None:
    body = build_chat_request(
        model="deepseek-flash",
        messages=[],
        stream=False,
        thinking=False,
        reasoning_effort="high",
        max_tokens=512,
        json_object=True,
    )
    assert body["response_format"] == {"type": "json_object"}


def test_request_body_omits_max_tokens_when_zero() -> None:
    body = build_chat_request(
        model="deepseek-flash",
        messages=[],
        stream=False,
        thinking=False,
        reasoning_effort="high",
        max_tokens=0,
    )
    assert "max_tokens" not in body


def test_request_body_carries_sampling_params() -> None:
    """FR-718：temperature 与 top_p 按设置下发；本项目不使用惩罚项，一律不下发。"""
    body = build_chat_request(
        model="deepseek-flash",
        messages=[],
        stream=False,
        thinking=True,
        reasoning_effort="high",
        max_tokens=256,
        sampling=SamplingOptions(temperature=0.7, top_p=0.9),
    )

    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.9
    assert "presence_penalty" not in body
    assert "frequency_penalty" not in body


def test_request_body_omits_sampling_when_not_given() -> None:
    """不传采样参数时不写进请求体：让官方默认值生效，而不是我们猜一个值。"""
    body = build_chat_request(
        model="deepseek-flash",
        messages=[],
        stream=False,
        thinking=False,
        reasoning_effort="high",
        max_tokens=256,
    )

    assert "temperature" not in body
    assert "top_p" not in body


def test_default_sampling_options_match_official_defaults() -> None:
    sampling = SamplingOptions()

    assert (sampling.temperature, sampling.top_p) == (1.0, 1.0)


def test_request_body_rejects_empty_model() -> None:
    with pytest.raises(LlmError) as exc:
        build_chat_request(
            model="  ",
            messages=[],
            stream=False,
            thinking=False,
            reasoning_effort="high",
            max_tokens=256,
        )
    assert exc.value.category is LlmErrorCategory.CONFIG


def test_usage_parsing_reads_cache_and_reasoning_fields() -> None:
    usage = TokenUsage.from_usage_dict(
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_cache_hit_tokens": 64,
            "prompt_cache_miss_tokens": 36,
            "completion_tokens_details": {"reasoning_tokens": 20},
        }
    )
    assert usage is not None
    assert (usage.prompt_tokens, usage.cached_prompt_tokens, usage.cache_miss_tokens) == (100, 64, 36)
    assert usage.reasoning_tokens == 20
    assert usage.to_payload().total_tokens == 150


def test_usage_parsing_tolerates_missing_fields() -> None:
    usage = TokenUsage.from_usage_dict({"prompt_tokens": 5})
    assert usage is not None
    assert usage.completion_tokens == 0
    assert usage.reasoning_tokens == 0
    assert TokenUsage.from_usage_dict(None) is None


def test_extract_delta_and_usage_from_stream_chunks() -> None:
    chunk = {"choices": [{"index": 0, "delta": {"content": "你好", "reasoning_content": "想一下"}}]}
    assert extract_delta(chunk) == ("你好", "想一下")
    assert extract_delta({"choices": []}) == (None, None)
    assert extract_delta({}) == (None, None)

    assert extract_usage({"usage": {"total_tokens": 3}}) is not None
    assert extract_usage({"choices": []}) is None


def test_parse_blocking_response_from_fixture() -> None:
    result = parse_blocking_response(fixture_json("response_blocking.json"))
    assert "贝叶斯定理" in result.content
    assert result.reasoning == "先给定义，再补一个例子。"
    assert result.usage is not None
    assert result.usage.total_tokens == 60
    assert result.usage.reasoning_tokens == 21


def test_parse_blocking_response_allows_empty_content() -> None:
    """官方已知问题：JSON Output 有概率返回空 content，解析层不应报错（FR-716）。"""
    result = parse_blocking_response(fixture_json("response_json_empty_content.json"))
    assert result.content == ""
    assert result.usage is not None
    assert result.usage.completion_tokens == 0


def test_truncated_json_sample_is_recognisable() -> None:
    """截断时必须能被识别（finish_reason=length + JSON 解析失败），以便按 FR-716 重试。"""
    body = fixture_json("response_json_truncated.json")
    assert body["choices"][0]["finish_reason"] == "length"

    result = parse_blocking_response(body)
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.content)


def test_parse_blocking_response_rejects_missing_choices() -> None:
    with pytest.raises(LlmError) as exc:
        parse_blocking_response({"usage": {}})
    assert exc.value.category is LlmErrorCategory.PROTOCOL


def test_parse_models_response_from_fixture() -> None:
    models = parse_models_response(fixture_json("models.json"))
    assert [m.id for m in models] == ["deepseek-flash", "deepseek-v4-pro"]

    flash = models[0]
    assert flash.name == "DeepSeek-V4.1-Flash"
    assert flash.context_window == 1048576
    assert flash.max_output_tokens == 393216
    assert flash.effort_levels == ("low", "high", "max")
    assert flash.default_effort == "high"


def test_parse_models_response_rejects_bad_shape() -> None:
    with pytest.raises(LlmError) as exc:
        parse_models_response({"data": "nope"})
    assert exc.value.category is LlmErrorCategory.PROTOCOL
