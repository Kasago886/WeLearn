"""运维可见性接口测试（T3.5/T3.6、FR-703、FR-705、FR-715、FR-718、FR-909）。

覆盖三件事：设置接口把"实际会用什么"摊开（含按用途覆盖的生效性）、用量接口给出按用途
与按主题的聚合、健康检查把存储可写性与日志目录状态报出来。
"""

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.contracts.schemas import LlmPurpose
from app.infra.llm.protocol import TokenUsage
from app.infra.settings import AppSettings
from app.main import create_app
from app.main_context import AppContext, build_context
from app.services.learning.repository import DomainRepositories
from app.services.llm.usage import UsageRecorder

PURPOSES = {"assess", "plan", "teach", "judge", "compress"}
ALL_INHERIT = {"model": "inherit", "thinking": "inherit", "reasoning_effort": "inherit"}


@pytest.fixture
def api_client(env: dict[str, str]) -> Iterator[TestClient]:
    app = create_app(env)
    with TestClient(app) as client:
        yield client


def context_of(client: TestClient) -> AppContext:
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    assert isinstance(ctx, AppContext)
    return ctx


def settings_of(client: TestClient) -> dict:
    return client.get("/api/settings").json()["settings"]


# --------------------------------------------------------------------------- #
# 设置接口（FR-703、FR-715、FR-718）
# --------------------------------------------------------------------------- #


def test_settings_expose_new_fields_with_defaults(api_client: TestClient) -> None:
    body = settings_of(api_client)

    assert body["temperature"] == 1.0
    assert body["top_p"] == 1.0
    assert body["audit_enabled"] is False
    assert body["use_recommended_overrides"] is False
    assert set(body["purpose_overrides"]) == PURPOSES
    assert body["purpose_overrides"]["judge"] == ALL_INHERIT
    # model_plan 是只读的"实际会用什么"：默认全部继承全局设置
    assert set(body["model_plan"]) == PURPOSES
    assert body["model_plan"]["judge"] == {
        "model": "deepseek-flash",
        "thinking": True,
        "reasoning_effort": "high",
    }
    assert body["read_only_fields"] == ["data_dir"]


def test_settings_round_trip_preserves_sampling_and_overrides(api_client: TestClient) -> None:
    body = settings_of(api_client)
    body["temperature"] = 0.5
    body["top_p"] = 0.8
    body["audit_enabled"] = True
    body["purpose_overrides"]["compress"] = {"model": "pro", "thinking": "off", "reasoning_effort": "low"}

    saved = api_client.put("/api/settings", json=body).json()["settings"]

    assert saved["temperature"] == 0.5
    assert saved["audit_enabled"] is True
    assert saved["purpose_overrides"]["compress"] == {
        "model": "pro",
        "thinking": "off",
        "reasoning_effort": "low",
    }
    assert saved["model_plan"]["compress"] == {"model": "deepseek-v4-pro", "thinking": False, "reasoning_effort": "low"}

    reloaded = settings_of(api_client)
    assert reloaded["top_p"] == 0.8
    assert reloaded["purpose_overrides"]["compress"] == saved["purpose_overrides"]["compress"]


def test_recommended_overrides_take_effect_without_restart(api_client: TestClient) -> None:
    """保存即生效（FR-703）：响应里的 model_plan 与运行期策略必须同时变。"""
    body = settings_of(api_client)
    body["use_recommended_overrides"] = True

    saved = api_client.put("/api/settings", json=body).json()["settings"]

    assert saved["model_plan"]["judge"] == {
        "model": "deepseek-v4-pro",
        "thinking": True,
        "reasoning_effort": "high",
    }
    assert saved["model_plan"]["plan"]["model"] == "deepseek-v4-pro"
    assert saved["model_plan"]["compress"]["thinking"] is False
    assert saved["model_plan"]["teach"]["model"] == "deepseek-flash"

    stack = api_client.app.state.stack  # type: ignore[attr-defined]
    assert stack.policy.resolve(LlmPurpose.JUDGE).model == "deepseek-v4-pro"


def test_explicit_override_beats_recommended(api_client: TestClient) -> None:
    body = settings_of(api_client)
    body["use_recommended_overrides"] = True
    body["purpose_overrides"]["judge"] = {"model": "default", "thinking": "off", "reasoning_effort": "low"}

    saved = api_client.put("/api/settings", json=body).json()["settings"]

    assert saved["model_plan"]["judge"] == {
        "model": "deepseek-flash",
        "thinking": False,
        "reasoning_effort": "low",
    }
    # 未被显式覆盖的用途仍拿推荐项
    assert saved["model_plan"]["plan"]["model"] == "deepseek-v4-pro"


def test_resetting_overrides_takes_effect_immediately(api_client: TestClient) -> None:
    """把开关关回去、覆盖项改回继承后，必须真的回到默认——否则就是"界面上关了，实际没关"。"""
    body = settings_of(api_client)
    body["use_recommended_overrides"] = True
    body["purpose_overrides"]["teach"] = {"model": "pro", "thinking": "on", "reasoning_effort": "max"}
    api_client.put("/api/settings", json=body)

    reset = settings_of(api_client)
    reset["use_recommended_overrides"] = False
    reset["purpose_overrides"]["teach"] = ALL_INHERIT
    saved = api_client.put("/api/settings", json=reset).json()["settings"]

    assert saved["model_plan"]["judge"]["model"] == "deepseek-flash"
    assert saved["model_plan"]["teach"] == {
        "model": "deepseek-flash",
        "thinking": True,
        "reasoning_effort": "high",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": 2.5},
        {"temperature": -0.1},
        {"top_p": 0.0},
        {"top_p": 1.5},
    ],
)
def test_out_of_range_sampling_returns_422(api_client: TestClient, changes: dict[str, float]) -> None:
    body = {**settings_of(api_client), **changes}

    assert api_client.put("/api/settings", json=body).status_code == 422


def test_unknown_purpose_in_overrides_returns_422(api_client: TestClient) -> None:
    body = settings_of(api_client)
    body["purpose_overrides"]["research"] = ALL_INHERIT

    assert api_client.put("/api/settings", json=body).status_code == 422


def test_unknown_override_value_returns_422(api_client: TestClient) -> None:
    body = settings_of(api_client)
    body["purpose_overrides"]["judge"] = {"model": "gpt-4", "thinking": "inherit", "reasoning_effort": "inherit"}

    assert api_client.put("/api/settings", json=body).status_code == 422


# --------------------------------------------------------------------------- #
# 用量接口（FR-705、FR-720）
# --------------------------------------------------------------------------- #


def seed_usage(client: TestClient, title: str = "贝叶斯定理") -> str:
    ctx = context_of(client)
    subject = DomainRepositories(ctx.db).subjects.create(title)
    recorder = UsageRecorder(ctx.db)
    recorder.record(
        LlmPurpose.TEACH,
        "deepseek-flash",
        TokenUsage(prompt_tokens=100, cache_miss_tokens=100, total_tokens=100),
        subject_id=subject.id,
        duration_ms=120,
    )
    recorder.record(
        LlmPurpose.JUDGE,
        "deepseek-v4-pro",
        TokenUsage(prompt_tokens=50, cached_prompt_tokens=50, total_tokens=50),
        subject_id=subject.id,
        duration_ms=80,
    )
    return subject.id


def test_usage_endpoint_keeps_existing_fields_and_adds_aggregations(api_client: TestClient) -> None:
    subject_id = seed_usage(api_client)

    body = api_client.get("/api/usage").json()

    # 既有字段不变（向后兼容）
    for key in (
        "calls",
        "prompt_tokens",
        "cached_prompt_tokens",
        "cache_miss_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cache_hit_rate",
    ):
        assert key in body

    assert body["calls"] == 2
    assert body["total_tokens"] == 150

    assert set(body["by_purpose"]) == {"teach", "judge"}
    assert body["by_purpose"]["teach"]["calls"] == 1
    assert body["by_purpose"]["teach"]["total_tokens"] == 100
    assert body["by_purpose"]["teach"]["cache_hit_rate"] == 0.0
    assert body["by_purpose"]["teach"]["duration_ms"] == 120
    assert body["by_purpose"]["judge"]["cache_hit_rate"] == 1.0

    assert body["by_subject"] == [
        {
            "subject_id": subject_id,
            "title": "贝叶斯定理",
            "calls": 2,
            "total_tokens": 150,
            "cache_hit_rate": 0.3333,
            "duration_ms": 200,
        }
    ]
    assert body["cache_hint"] is None  # 有命中，不该报"命中率长期为 0"


def test_cache_hint_is_reported_when_hits_are_zero(api_client: TestClient) -> None:
    """FR-705：calls > 0 且命中率为 0 时给出可能原因，而不是让用户自己猜。"""
    ctx = context_of(api_client)
    UsageRecorder(ctx.db).record(
        LlmPurpose.TEACH,
        "deepseek-flash",
        TokenUsage(prompt_tokens=200, cache_miss_tokens=200, total_tokens=200),
    )

    body = api_client.get("/api/usage").json()

    assert body["calls"] == 1
    assert body["cache_hit_rate"] == 0.0
    hint = body["cache_hint"]
    assert isinstance(hint, str)
    assert "缓存" in hint


def test_cache_hint_is_null_without_calls(api_client: TestClient) -> None:
    body = api_client.get("/api/usage").json()

    assert body["calls"] == 0
    assert body["cache_hint"] is None
    assert body["by_purpose"] == {}
    assert body["by_subject"] == []


def test_by_subject_keeps_only_top_ten(api_client: TestClient) -> None:
    ctx = context_of(api_client)
    recorder = UsageRecorder(ctx.db)
    for index in range(12):
        recorder.record(LlmPurpose.TEACH, "m", TokenUsage(total_tokens=index), subject_id=f"ghost-{index}")

    items = api_client.get("/api/usage").json()["by_subject"]

    assert len(items) == 10
    # 主题不存在时标题退回 id，而不是留空让前端显示成空白行
    assert items[0] == {
        "subject_id": "ghost-11",
        "title": "ghost-11",
        "calls": 1,
        "total_tokens": 11,
        "cache_hit_rate": 0.0,
        "duration_ms": 0,
    }


# --------------------------------------------------------------------------- #
# 健康检查（FR-909）
# --------------------------------------------------------------------------- #


def test_health_reports_ops_visibility(api_client: TestClient) -> None:
    body = api_client.get("/api/health").json()

    assert body["status"] == "ok"
    assert body["problems"] == []
    assert body["storage"]["writable"] is True
    assert body["logs"]["writable"] is True
    assert body["logs"]["level"] == "INFO"
    assert body["logs"]["dir"] == str(Path(body["storage"]["data_dir"]) / "logs")
    assert body["llm"] == {"audit_enabled": False, "configured": False}
    assert body["usage"] == {"calls": 0, "total_tokens": 0, "cache_hit_rate": 0.0}
    assert body["features"] == {
        "use_recommended_overrides": False,
        "show_reasoning": False,
        "thinking_enabled": True,
    }
    # 写测试的探针文件必须被清理掉，否则升级/备份会把垃圾文件一起带走
    assert list(Path(body["storage"]["data_dir"]).glob("*health-write-probe*")) == []


def test_health_reflects_current_settings_and_usage(api_client: TestClient) -> None:
    seed_usage(api_client)
    body = settings_of(api_client)
    body["audit_enabled"] = True
    body["use_recommended_overrides"] = True
    body["show_reasoning"] = True
    api_client.put("/api/settings", json=body)

    health = api_client.get("/api/health").json()

    assert health["llm"]["audit_enabled"] is True
    assert health["features"] == {
        "use_recommended_overrides": True,
        "show_reasoning": True,
        "thinking_enabled": True,
    }
    assert health["usage"] == {"calls": 2, "total_tokens": 150, "cache_hit_rate": 0.3333}


def test_health_degrades_when_directories_are_not_writable(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """数据目录与日志目录写不进去时降级而不是崩溃（NFR-107），并把失败项报出来。"""

    def boom(self: Path, *args: object, **kwargs: object) -> int:
        raise OSError("磁盘只读")

    monkeypatch.setattr(Path, "write_text", boom)

    response = api_client.get("/api/health")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert body["storage"]["writable"] is False
    assert body["logs"]["writable"] is False
    assert body["problems"] == ["data_dir", "logs"]
    assert "磁盘只读" in body["storage"]["writable_detail"]


# --------------------------------------------------------------------------- #
# 装配层面的验证（设置 → 请求体 / 审计）
# --------------------------------------------------------------------------- #


async def test_context_wires_settings_into_requests(env: dict[str, str]) -> None:
    """设置页的值真的会进到请求体与审计开关（FR-715、FR-718）。

    这里刻意不用 HTTP 接口：接口层的字段已经由上面的用例覆盖，这一条要证明的是
    ``build_context`` 把 getter 接对了——设置改了，客户端下一次请求就按新值走。
    """
    ctx = build_context(env)
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "好"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 1},
            },
        )

    try:
        ctx.secrets.set_api_key("sk-context-wiring-123456")
        ctx.refresh_settings(AppSettings(temperature=0.4, top_p=0.75, audit_enabled=True))
        assert ctx.llm.audit_enabled() is True

        # 换成桩传输：这里要验的是装配，不是网络
        await ctx.llm.aclose()
        ctx.llm._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await ctx.llm.chat(model="deepseek-flash", messages=[{"role": "user", "content": "hi"}])
    finally:
        await ctx.llm.aclose()
        ctx.close()

    assert captured[0]["temperature"] == 0.4
    assert captured[0]["top_p"] == 0.75

