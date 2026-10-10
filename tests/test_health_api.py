"""健康检查接口测试（T0.1 出口条件、FR-708）。"""

import pytest
from fastapi.testclient import TestClient

from app.infra.storage.models import ApiKeySecretRow
from app.main import create_app

KEY = "sk-secret-value-1234567890"


def test_health_reports_storage_and_model_config(env: dict[str, str]) -> None:
    app = create_app(env)
    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["problems"] == []
    assert body["storage"]["ok"] is True
    assert body["storage"]["schema_version"] == 1
    assert body["model"]["default"] == "deepseek-flash"
    assert body["model"]["pro"] == "deepseek-v4-pro"
    assert body["api_key"] == {"configured": False, "source": "none", "masked": None}


def test_health_never_leaks_the_api_key(env: dict[str, str]) -> None:
    app = create_app(env)
    with TestClient(app) as client:
        app.state.ctx.secrets.set_api_key(KEY)
        response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()

    assert body["api_key"]["configured"] is True
    assert body["api_key"]["masked"] == "sk-****7890"
    # 明文与密文都不得出现在响应里
    assert KEY not in response.text

    with app.state.ctx.db.session() as session:
        row = session.get(ApiKeySecretRow, 1)
    assert row is not None
    assert row.cipher not in response.text


def test_health_survives_broken_storage(env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    """存储不可用时服务仍要能响应，便于用户排查（NFR-107）。"""
    app = create_app(env)

    def boom() -> None:
        raise RuntimeError("数据库不可用")

    monkeypatch.setattr(app.state.ctx.db, "session", boom)

    with TestClient(app) as client:
        response = client.get("/api/health")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert body["storage"]["ok"] is False
    assert "数据库不可用" in body["storage"]["detail"]
    # 存储坏了也不能让密钥读取把整个接口带崩（NFR-107）
    assert "storage" in body["problems"]
    assert body["api_key"]["configured"] is False
