"""SSE 解析测试（FR-704、AC-28）。"""

import json

import pytest

from app.infra.deepseek.sse import SseEventParser, SseLineKind, iter_payloads, parse_line
from tests.conftest import fixture_lines


def test_ignores_blank_comment_and_field_lines() -> None:
    for raw in ["", "   ", ": keep-alive", ": 保活注释", "event: message", "id: 42", "retry: 3000"]:
        kind, payload = parse_line(raw)
        assert kind is SseLineKind.IGNORE, raw
        assert payload == ""


def test_none_line_is_ignored() -> None:
    assert parse_line(None) == (SseLineKind.IGNORE, "")


def test_data_line_forms() -> None:
    assert parse_line('data: {"a":1}') == (SseLineKind.DATA, '{"a":1}')
    assert parse_line('data:{"a":1}') == (SseLineKind.DATA, '{"a":1}')
    assert parse_line("data:   ") == (SseLineKind.IGNORE, "")


def test_done_marker() -> None:
    assert parse_line("data: [DONE]") == (SseLineKind.DONE, "")


def test_normal_stream_emits_deltas_and_usage() -> None:
    payloads = list(iter_payloads(fixture_lines("stream_normal.sse")))

    assert len(payloads) == 5
    chunks = [json.loads(item) for item in payloads]
    assert chunks[0]["choices"][0]["delta"]["reasoning_content"] == "用户问的是贝叶斯定理。"
    assert chunks[-1]["usage"]["total_tokens"] == 60


def test_keepalive_and_broken_chunk_do_not_break_stream() -> None:
    payloads = list(iter_payloads(fixture_lines("stream_keepalive_and_broken.sse")))

    # 保活注释与 event: 字段被忽略，剩余 4 个数据块（含 1 个坏块）全部透出
    assert len(payloads) == 4
    assert json.loads(payloads[0])["choices"][0]["delta"]["content"] == "第一块"

    # 坏块由上层决定跳过（客户端记日志后继续），解析器本身不做 JSON 校验
    with pytest.raises(json.JSONDecodeError):
        json.loads(payloads[1])

    assert json.loads(payloads[2])["choices"][0]["delta"]["content"] == "第二块"
    assert json.loads(payloads[3])["usage"]["prompt_cache_hit_tokens"] == 64


def test_multi_line_data_is_joined() -> None:
    payloads = list(iter_payloads(['data: {"a":', "data: 1}", ""]))
    assert payloads == ['{"a":\n1}']


def test_parser_stops_after_done() -> None:
    parser = SseEventParser()
    assert parser.feed('data: {"a":1}') == []
    assert parser.feed("") == ['{"a":1}']
    assert parser.feed("data: [DONE]") == []
    assert parser.finished is True
    # [DONE] 之后的内容不再产出
    assert parser.feed('data: {"b":2}') == []


def test_complete_event_without_done_is_flushed() -> None:
    """连接被服务端直接关闭（无 [DONE]）时，已收到的完整事件不得丢失。"""
    assert list(iter_payloads(['data: {"a":1}', ""])) == ['{"a":1}']
