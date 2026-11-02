"""结构化输出通道测试（T1.5，FR-716）。"""

import pytest
from pydantic import BaseModel, Field

from app.contracts.schemas import LlmPurpose
from app.infra.llm.protocol import TokenUsage
from app.infra.settings import AppSettings
from app.services.llm.agent import RoundExecutor
from app.services.llm.memory import Memory
from app.services.llm.policy import ModelPolicy
from app.services.llm.structured import (
    ParseOutcome,
    StructuredChannel,
    StructuredOutputError,
    parse_json_payload,
)
from tests.stubs import FakeLlmClient, json_response

USAGE = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)


class Demo(BaseModel):
    answer: str = Field(min_length=1)
    score: int = 0


def make_channel(client: FakeLlmClient, **kwargs) -> StructuredChannel:
    executor = RoundExecutor(client, Memory(), ModelPolicy(AppSettings(max_tokens=512)))
    return StructuredChannel(executor, **kwargs)


# --------------------------------------------------------------------------- #
# 解析容错
# --------------------------------------------------------------------------- #


def test_parse_strict_json() -> None:
    parsed = parse_json_payload('{"answer": "x"}')
    assert parsed.outcome is ParseOutcome.OK
    assert parsed.payload == {"answer": "x"}


def test_parse_json_inside_code_fence() -> None:
    parsed = parse_json_payload('```json\n{"answer": "x"}\n```')
    assert parsed.outcome is ParseOutcome.OK
    assert parsed.payload == {"answer": "x"}


def test_parse_json_with_surrounding_prose() -> None:
    """模型常在前言后语里包 JSON，截取最外层 {} 必须能救回来。"""
    parsed = parse_json_payload('好的，结果如下：\n{"answer": "x"}\n希望有帮助。')
    assert parsed.outcome is ParseOutcome.OK
    assert parsed.payload == {"answer": "x"}


def test_parse_empty_and_blank_content() -> None:
    assert parse_json_payload("").outcome is ParseOutcome.EMPTY
    assert parse_json_payload("   \n ").outcome is ParseOutcome.EMPTY
    assert parse_json_payload(None).outcome is ParseOutcome.EMPTY


def test_parse_invalid_shapes() -> None:
    assert parse_json_payload("这不是 JSON").outcome is ParseOutcome.INVALID
    assert parse_json_payload('{"answer": "未闭合"').outcome is ParseOutcome.INVALID
    # 合法 JSON 但不是对象
    assert parse_json_payload("[1, 2]").outcome is ParseOutcome.INVALID
    assert parse_json_payload('"字符串"').outcome is ParseOutcome.INVALID


# --------------------------------------------------------------------------- #
# 请求与重试
# --------------------------------------------------------------------------- #


async def test_successful_structured_request() -> None:
    client = FakeLlmClient(blocking_script=[json_response('{"answer": "贝叶斯定理", "score": 3}', usage=USAGE)])

    result, usage = await make_channel(client).request(
        LlmPurpose.JUDGE,
        system="你是评分者，请输出 json。",
        user="请判定。",
        schema=Demo,
    )

    assert result.answer == "贝叶斯定理"
    assert result.score == 3
    assert usage is USAGE

    call = client.calls[0]
    assert call.json_object is True  # 必须带 response_format
    assert call.max_tokens == 512  # 必须显式设置 max_tokens（防截断）


async def test_empty_content_is_retried_with_larger_max_tokens() -> None:
    """官方已知问题：JSON Output 有概率返回空 content；重试时把 max_tokens 翻倍。"""
    client = FakeLlmClient(
        blocking_script=[
            json_response(""),
            json_response('{"answer": "第二次成功"}'),
        ]
    )

    result, _ = await make_channel(client).request(
        LlmPurpose.JUDGE, system="输出 json", user="判定", schema=Demo
    )

    assert result.answer == "第二次成功"
    assert client.blocking_calls == 2
    assert client.calls[0].max_tokens == 512
    assert client.calls[1].max_tokens == 1024


async def test_truncated_json_is_retried() -> None:
    client = FakeLlmClient(
        blocking_script=[
            json_response('{"answer": "被截断'),
            json_response('{"answer": "完整"}'),
        ]
    )

    result, _ = await make_channel(client).request(
        LlmPurpose.COMPRESS, system="输出 json", user="压缩", schema=Demo
    )

    assert result.answer == "完整"
    assert client.blocking_calls == 2


async def test_schema_violation_is_retried_then_reported() -> None:
    client = FakeLlmClient(
        blocking_script=[
            json_response('{"score": 1}'),  # 缺 answer
            json_response('{"answer": "", "score": 1}'),  # answer 太短
        ]
    )

    with pytest.raises(StructuredOutputError) as exc:
        await make_channel(client).request(LlmPurpose.JUDGE, system="输出 json", user="判定", schema=Demo)

    assert client.blocking_calls == 2
    assert exc.value.retryable is True
    assert "结构校验失败" in exc.value.detail or "结构校验失败" in exc.value.user_message


async def test_exhausted_retries_raise_retryable_error() -> None:
    client = FakeLlmClient(blocking_script=[json_response(""), json_response("还是空的")])

    with pytest.raises(StructuredOutputError) as exc:
        await make_channel(client).request(LlmPurpose.JUDGE, system="输出 json", user="判定", schema=Demo)

    assert exc.value.retryable is True
    assert "2 次" in exc.value.user_message


async def test_max_tokens_growth_respects_ceiling() -> None:
    client = FakeLlmClient(blocking_script=[json_response(""), json_response('{"answer": "ok"}')])

    await make_channel(client, max_tokens_ceiling=700).request(
        LlmPurpose.JUDGE, system="输出 json", user="判定", schema=Demo
    )

    assert client.calls[1].max_tokens == 700


async def test_prompt_without_json_keyword_is_a_programming_error() -> None:
    """官方要求提示词含 json 字样，缺失属于编码缺陷，应尽早暴露而不是等模型返回垃圾。"""
    client = FakeLlmClient(blocking_script=[json_response('{"answer": "x"}')])

    with pytest.raises(ValueError) as exc:
        await make_channel(client).request(
            LlmPurpose.JUDGE,
            system="你是评分者。",
            user="请判定。",
            schema=Demo,
        )

    assert "json" in str(exc.value)
    assert client.blocking_calls == 0


async def test_json_keyword_check_is_case_insensitive() -> None:
    client = FakeLlmClient(blocking_script=[json_response('{"answer": "x"}')])

    result, _ = await make_channel(client).request(
        LlmPurpose.JUDGE, system="输出 JSON 格式结果", user="判定", schema=Demo
    )

    assert result.answer == "x"


async def test_single_attempt_mode_raises_on_first_failure() -> None:
    client = FakeLlmClient(blocking_script=[json_response("")])

    with pytest.raises(StructuredOutputError):
        await make_channel(client, max_attempts=1).request(
            LlmPurpose.JUDGE, system="输出 json", user="判定", schema=Demo
        )

    assert client.blocking_calls == 1
