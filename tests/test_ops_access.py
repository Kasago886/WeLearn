"""日志轮转与可选访问口令（T4.5、FR-903、FR-907）。"""

import os
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.infra.logging.queue_logger import CHANNEL_APP, CHANNEL_LLM, QueueLogger
from app.main import create_app

PROTECTED_READ_ONLY = {"data_dir", "read_only_fields", "api_key", "model_plan", "access_token_enabled"}


@pytest.fixture
def api_client(env: dict[str, str]) -> Iterator[TestClient]:
    app = create_app(env)
    with TestClient(app) as client:
        yield client


def put_settings(client: TestClient, **changes: object) -> dict:
    """按"读-改-写"保存设置：PUT 是整体替换语义，必须先拿全量再改。"""
    current = client.get("/api/settings").json()["settings"]
    payload = {key: value for key, value in current.items() if key not in PROTECTED_READ_ONLY}
    payload.update(changes)
    response = client.put("/api/settings", json=payload)
    assert response.status_code == 200, response.text
    return response.json()["settings"]


def settings_of(client: TestClient) -> dict:
    return client.get("/api/settings").json()["settings"]


# --------------------------------------------------------------------------- #
# 日志轮转与保留（FR-907）
# --------------------------------------------------------------------------- #


def test_logger_rotates_when_file_exceeds_limit(tmp_path: Path) -> None:
    logger = QueueLogger(tmp_path, level="DEBUG", max_bytes=200, retention_days=14)
    try:
        for index in range(30):
            logger.log(CHANNEL_APP, "INFO", f"第 {index} 条日志 " + "x" * 60)
        assert logger.flush(timeout=3.0)
    finally:
        logger.close()

    archived = list(tmp_path.glob(f"{CHANNEL_APP}.*.log"))
    assert archived, "超过阈值后应产生归档文件"
    # 活动文件在轮转后重新从 0 开始写，不会无限增长
    assert (tmp_path / f"{CHANNEL_APP}.log").stat().st_size < 400


def test_prune_removes_only_expired_archives(tmp_path: Path) -> None:
    logger = QueueLogger(tmp_path, level="DEBUG", max_bytes=10**9, retention_days=7)
    try:
        active = tmp_path / f"{CHANNEL_APP}.log"
        active.write_text("active", encoding="utf-8")
        old = tmp_path / f"{CHANNEL_APP}.20200101-000000-000.log"
        old.write_text("old" * 100, encoding="utf-8")
        fresh = tmp_path / f"{CHANNEL_LLM}.20260101-000000-000.log"
        fresh.write_text("fresh", encoding="utf-8")

        old_time = (datetime.now() - timedelta(days=30)).timestamp()
        os.utime(old, (old_time, old_time))

        removed, freed = logger.prune()

        assert removed == 1
        assert freed == 300
        assert not old.exists()
        assert fresh.exists()
        # 活动文件永远不删：删掉它会让日志"凭空消失"
        assert active.exists()
    finally:
        logger.close()


def test_clean_truncates_active_and_deletes_archives(tmp_path: Path) -> None:
    logger = QueueLogger(tmp_path, level="DEBUG", max_bytes=10**9)
    try:
        (tmp_path / f"{CHANNEL_APP}.log").write_text("当前日志" * 10, encoding="utf-8")
        (tmp_path / f"{CHANNEL_APP}.20200101-000000-000.log").write_text("归档", encoding="utf-8")

        removed, freed = logger.clean()

        assert removed == 1
        assert freed > 0
        assert (tmp_path / f"{CHANNEL_APP}.log").read_text(encoding="utf-8") == ""
        assert not (tmp_path / f"{CHANNEL_APP}.20200101-000000-000.log").exists()
    finally:
        logger.close()


def test_log_clean_endpoint(api_client: TestClient) -> None:
    log_dir = Path(api_client.get("/api/logs").json()["log_dir"])
    (log_dir / "app.log").write_text("待清理" * 10, encoding="utf-8")

    stats = api_client.get("/api/logs").json()
    assert stats["total_bytes"] > 0
    assert stats["retention_days"] >= 1

    body = api_client.post("/api/logs/clean").json()
    assert body["freed_bytes"] > 0
    assert api_client.get("/api/logs").json()["total_bytes"] == 0


# --------------------------------------------------------------------------- #
# 可选访问口令（FR-903）
# --------------------------------------------------------------------------- #


def test_access_token_is_optional_by_default(api_client: TestClient) -> None:
    assert api_client.get("/api/subjects").status_code == 200
    verify = api_client.get("/api/auth/verify").json()
    assert verify == {"ok": True, "required": False}


def test_access_token_guards_every_data_endpoint(api_client: TestClient) -> None:
    saved = put_settings(api_client, access_token="s3cret-token")
    assert saved["access_token_enabled"] is True
    # 口令本身永不回传
    assert "access_token_hash" not in saved
    assert "access_token" not in saved

    assert api_client.get("/api/subjects").status_code == 401
    assert api_client.get("/api/subjects", headers={"X-Access-Token": "wrong"}).status_code == 401
    # 健康检查与口令探针必须放行：它们不含学习数据，且是排查"口令错还是服务没起"的入口
    assert api_client.get("/api/health").status_code == 200
    probe = api_client.get("/api/auth/verify")
    assert probe.status_code == 200
    assert probe.json() == {"ok": False, "required": True}
    assert api_client.get("/api/auth/verify", headers={"X-Access-Token": "wrong"}).json()["ok"] is False
    assert api_client.get("/api/auth/verify", headers={"X-Access-Token": "s3cret-token"}).json() == {
        "ok": True,
        "required": True,
    }
    assert api_client.get("/api/subjects", headers={"X-Access-Token": "s3cret-token"}).status_code == 200


def test_access_token_can_be_cleared(api_client: TestClient) -> None:
    put_settings(api_client, access_token="temporary")
    assert api_client.get("/api/subjects").status_code == 401

    headers = {"X-Access-Token": "temporary"}
    current = api_client.get("/api/settings", headers=headers).json()["settings"]
    payload = {key: value for key, value in current.items() if key not in PROTECTED_READ_ONLY}
    payload["access_token"] = ""
    response = api_client.put("/api/settings", json=payload, headers=headers)

    assert response.status_code == 200
    assert response.json()["settings"]["access_token_enabled"] is False
    assert api_client.get("/api/subjects").status_code == 200


def test_omitting_access_token_keeps_existing_one(api_client: TestClient) -> None:
    """三态口令的关键：改别的设置不该顺手把口令清掉。"""
    put_settings(api_client, access_token="persisted")
    headers = {"X-Access-Token": "persisted"}

    current = api_client.get("/api/settings", headers=headers).json()["settings"]
    payload = {key: value for key, value in current.items() if key not in PROTECTED_READ_ONLY}
    payload["log_level"] = "WARNING"  # 只改日志级别，不带 access_token
    response = api_client.put("/api/settings", json=payload, headers=headers)

    assert response.status_code == 200
    assert response.json()["settings"]["access_token_enabled"] is True
    assert api_client.get("/api/subjects").status_code == 401
    assert api_client.get("/api/subjects", headers=headers).status_code == 200


# --------------------------------------------------------------------------- #
# P4 新增设置的往返与用量成本（FR-505、FR-706、FR-714、FR-907）
# --------------------------------------------------------------------------- #


def test_p4_settings_round_trip(api_client: TestClient) -> None:
    body = settings_of(api_client)
    assert body["review_interval_days"] == 7
    assert body["log_max_mb"] == 5
    assert body["log_retention_days"] == 14
    assert body["tools_enabled"] is False
    assert body["access_token_enabled"] is False
    # 价表回的是生效值（内置默认），不是空表
    assert body["price_table"]["deepseek-flash"]["output"] > 0

    saved = put_settings(
        api_client,
        review_interval_days=3,
        log_max_mb=8,
        log_retention_days=30,
        tools_enabled=True,
        price_table={"deepseek-flash": {"cache_hit": 0.1, "cache_miss": 1.5, "output": 2.5}},
    )

    assert saved["review_interval_days"] == 3
    assert saved["log_max_mb"] == 8
    assert saved["log_retention_days"] == 30
    assert saved["tools_enabled"] is True
    assert saved["price_table"]["deepseek-flash"] == {"cache_hit": 0.1, "cache_miss": 1.5, "output": 2.5}
    # 未覆盖的模型保留内置价：深合并而不是整体替换
    assert saved["price_table"]["deepseek-v4-pro"]["output"] > 0

    # 保存后即时生效：日志轮转阈值不再需要重启
    ctx = api_client.app.state.ctx  # type: ignore[attr-defined]
    assert ctx.logger._max_bytes == 8 * 1024 * 1024


def test_usage_reports_cost(api_client: TestClient) -> None:
    from app.contracts.schemas import LlmPurpose
    from app.infra.deepseek.protocol import TokenUsage
    from app.services.llm.usage import UsageRecorder

    ctx = api_client.app.state.ctx  # type: ignore[attr-defined]
    UsageRecorder(ctx.db).record(
        LlmPurpose.TEACH,
        ctx.settings.model_default,
        TokenUsage(prompt_tokens=1000, cached_prompt_tokens=600, cache_miss_tokens=400, completion_tokens=200),
        duration_ms=1200,
    )

    body = api_client.get("/api/usage").json()

    assert body["cost"]["currency"] == "CNY"
    assert body["cost"]["amount"] > 0
    assert body["cost"]["total_tokens"] == 1200
    assert body["cost"]["peak_tokens"] + body["cost"]["off_peak_tokens"] == 1200
