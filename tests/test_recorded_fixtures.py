"""真实响应样本回放测试（T0.7）。

样本由 ``python scripts/llm_smoke.py --record`` 生成，存放在 ``recorded/``，
是**生产数据结构的唯一权威来源**——手写夹具只能反映我们的预期，真实样本才能暴露
"我们没预料到的字段"。

2026-10-10 录制实测得到的三条结论（已据此确认实现正确）：

1. 真实数据块除 ``choices`` 外还带 ``object``/``created``/``system_fingerprint``，
   ``choices[0]`` 还带 ``logprobs``；首块 ``delta`` 为 ``{role, content:null, reasoning_content:""}``
   ——解析器必须容忍空串与未知字段；
2. 空串 ``reasoning_content`` 必须按"无增量"处理，否则会向前端推空事件；
3. 余额接口的金额是**字符串**（``"17.39"``），不能按数字解析（FR-710 按币种展示）。

模块级 skip：若样本目录缺失（例如克隆时被清理），跳过而不是失败。
"""

import json
import re
from pathlib import Path

import httpx
import pytest

from app.contracts.schemas import LlmPurpose
from app.infra.llm.client import LlmClient
from app.infra.llm.protocol import ModelInfo, parse_blocking_response, parse_models_response

RECORDED_DIR = Path(__file__).parent / "fixtures" / "llm" / "recorded"

REQUIRED_FILES = (
    "stream_normal.sse",
    "response_blocking.json",
    "response_json_object.json",
    "models.json",
    "balance.json",
    "meta.json",
)

pytestmark = pytest.mark.skipif(
    not all((RECORDED_DIR / name).exists() for name in REQUIRED_FILES),
    reason="未找到完整的真实录制样本，请先执行 scripts/llm_smoke.py --record",
)


def read_text(name: str) -> str:
    return (RECORDED_DIR / name).read_text(encoding="utf-8")


def read_json(name: str) -> dict:
    loaded = json.loads(read_text(name))
    assert isinstance(loaded, dict)
    return loaded


# --------------------------------------------------------------------------- #
# 安全护栏：录制样本绝不能包含密钥
# --------------------------------------------------------------------------- #

SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{10,}")


def test_recorded_samples_contain_no_secrets() -> None:
    """录制脚本必须只记录模型输出与结构；此测试防止将来误开审计日志后录制出密钥。"""
    offenders: list[str] = []
    for path in sorted(RECORDED_DIR.glob("*")):
        if path.is_file() and SECRET_PATTERN.search(path.read_text(encoding="utf-8", errors="replace")):
            offenders.append(path.name)

    assert offenders == [], f"以下录制样本疑似包含密钥: {offenders}"


def test_meta_records_the_environment() -> None:
    meta = read_json("meta.json")
    assert meta["model"] == "deepseek-flash"
    assert meta["base_url"] == "https://api.deepseek.com"
    assert "recorded_at" in meta
    # meta 只记录环境信息，不应出现任何凭据字段
    assert not any("key" in field.lower() for field in meta)


# --------------------------------------------------------------------------- #
# 流式回放：通过真实客户端跑完整链路
# --------------------------------------------------------------------------- #


async def test_real_stream_replays_through_client() -> None:
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=read_text("stream_normal.sse").encode("utf-8"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LlmClient(
        base_url_getter=lambda: "https://api.deepseek.com",
        api_key_getter=lambda: "sk-test-000000000000",
        http_client=http,
    )
    try:
        events = [
            event
            async for event in client.stream_chat(
                model="deepseek-flash",
                messages=[{"role": "user", "content": "用两句话解释什么是贝叶斯定理。"}],
                thinking=True,
                reasoning_effort="high",
                max_tokens=8192,
                purpose=LlmPurpose.TEACH,
            )
        ]
    finally:
        await http.aclose()

    # 远端请求体是我们期望的形态
    assert captured[0]["thinking"] == {"type": "enabled"}
    assert captured[0]["reasoning_effort"] == "high"
    assert captured[0]["stream"] is True

    text = "".join(event.content or "" for event in events)
    reasoning = "".join(event.reasoning or "" for event in events)

    assert "贝叶斯定理" in text
    assert "P(A|B)" in text or "P(B|A)" in text
    # 思考已开启，思维链必然非空
    assert reasoning.strip() != ""

    # 空串增量必须被丢弃，否则前端会收到大量空事件
    assert all(event.content != "" for event in events)
    assert all(event.reasoning != "" for event in events)

    usage = events[-1].usage
    assert usage is not None
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.reasoning_tokens > 0
    assert usage.cached_prompt_tokens <= usage.prompt_tokens
    assert usage.completion_tokens > 100


async def test_real_stream_yields_usage_only_event_last() -> None:
    """用量只出现在最后一个数据块，且必须被单独产出为收尾事件。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=read_text("stream_normal.sse").encode("utf-8"))

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = LlmClient(
        base_url_getter=lambda: "https://api.deepseek.com",
        api_key_getter=lambda: "sk-test-000000000000",
        http_client=http,
    )
    try:
        events = [
            event
            async for event in client.stream_chat(model="deepseek-flash", messages=[{"role": "user", "content": "x"}])
        ]
    finally:
        await http.aclose()

    usage_events = [event for event in events if event.is_usage_only]
    assert len(usage_events) == 1
    assert events[-1] is usage_events[0]


# --------------------------------------------------------------------------- #
# 非流式与 JSON Output 回放
# --------------------------------------------------------------------------- #


def test_real_json_object_response_is_valid_json() -> None:
    """JSON Output 的真实返回：content 是可直接解析的 JSON（FR-716 的前提）。"""
    result = parse_blocking_response(read_json("response_json_object.json"))

    payload = json.loads(result.content)
    assert isinstance(payload, dict)
    assert payload["answer"]
    # 本次录制关闭了思考，思维链应为空
    assert result.reasoning is None
    assert result.usage is not None
    assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens


def test_real_blocking_response_parses() -> None:
    result = parse_blocking_response(read_json("response_blocking.json"))

    assert "贝叶斯定理" in result.content
    assert result.usage is not None
    assert result.usage.completion_tokens > 0
    assert result.usage.cached_prompt_tokens == 0


# --------------------------------------------------------------------------- #
# 模型列表与余额（FR-702、FR-710）
# --------------------------------------------------------------------------- #


def test_real_models_list_matches_official_docs() -> None:
    models = parse_models_response(read_json("models.json"))
    by_id: dict[str, ModelInfo] = {model.id: model for model in models}

    assert set(by_id) == {"deepseek-flash", "deepseek-v4-pro"}

    flash = by_id["deepseek-flash"]
    assert flash.name == "DeepSeek-V4.1-Flash"
    assert flash.context_window == 1048576
    assert flash.max_output_tokens == 393216
    assert flash.effort_levels == ("low", "high", "max")
    assert flash.default_effort == "high"


def test_balance_amounts_are_strings() -> None:
    """FR-710 要求按币种展示：金额是字符串，若按数字解析会丢精度或直接报错。"""
    balance = read_json("balance.json")

    assert isinstance(balance["is_available"], bool)
    infos = balance["balance_infos"]
    assert infos and isinstance(infos, list)

    first = infos[0]
    assert first["currency"] in {"CNY", "USD"}
    assert isinstance(first["total_balance"], str)
    assert float(first["total_balance"]) >= 0


# --------------------------------------------------------------------------- #
# 样本完整性
# --------------------------------------------------------------------------- #


def test_recorded_samples_are_all_success_shapes() -> None:
    """录制脚本只应在成功时落盘；若出现 error 结构说明录制逻辑被改坏。"""
    for name in ("response_blocking.json", "response_json_object.json"):
        body = read_json(name)
        assert "error" not in body
        assert body["choices"]


def test_replay_does_not_need_keepalive_lines() -> None:
    """本次录制未出现保活注释（响应在保活触发前就开始了）。

    保活行为依据官方文档，由 tests/fixtures/llm/stream_keepalive_and_broken.sse 覆盖；
    这里只确认"没有保活行时解析同样正确"，两种形态都要能跑。
    """
    lines = read_text("stream_normal.sse").splitlines()
    assert not any(line.startswith(": ") for line in lines)

    from app.infra.llm.sse import SseEventParser

    parser = SseEventParser()
    payloads: list[str] = []
    for line in lines:
        payloads.extend(parser.feed(line))
        if parser.finished:
            break

    assert parser.finished is True
    # 每个数据块都产出且没有空载荷
    assert len(payloads) == sum(1 for line in lines if line.startswith("data:")) - 1
    assert all(payload for payload in payloads)
