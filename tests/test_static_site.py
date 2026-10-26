"""前端静态资源托管（D1）。

交付形态的核心是"一条命令起服务"：后端同时提供 API 与构建好的前端。
这里锁住三件容易出错的事：入口与深链接可用、`/api` 的未知路径**不**回退成 HTML、
静态目录不能把根目录之外的文件读出来。
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.static_site import mount_frontend


def make_site(tmp_path: Path) -> tuple[FastAPI, Path]:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>WeLearn</title>", encoding="utf-8")
    (dist / "assets" / "app-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    # 根目录外的"机密"文件：越界请求绝不能读到它
    (tmp_path / "secret.txt").write_text("TOP-SECRET", encoding="utf-8")

    app = FastAPI()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    assert mount_frontend(app, dist) is True
    return app, dist


def test_missing_dist_is_a_noop(tmp_path: Path) -> None:
    """开发期没有构建产物：不挂载，也不能抛异常。"""
    app = FastAPI()
    assert mount_frontend(app, tmp_path / "not-built") is False


def test_serves_index_and_assets(tmp_path: Path) -> None:
    app, _ = make_site(tmp_path)
    client = TestClient(app)

    root = client.get("/")
    assert root.status_code == 200
    assert "WeLearn" in root.text

    asset = client.get("/assets/app-abc123.js")
    assert asset.status_code == 200
    assert asset.text == "console.log(1)"
    # 带内容哈希的产物可以长缓存
    assert "immutable" in asset.headers["cache-control"]

    # 入口文件必须每次校验，否则发版后用户会一直看到旧页面
    assert "no-cache" in root.headers["cache-control"]


def test_deep_link_falls_back_to_index(tmp_path: Path) -> None:
    """刷新 `/subjects/{id}/teach` 这类前端路由时服务端没有对应文件，必须回 index.html。"""
    app, _ = make_site(tmp_path)
    client = TestClient(app)

    for path in ["/subjects/abc/teach", "/search", "/settings"]:
        response = client.get(path)
        assert response.status_code == 200, path
        assert "WeLearn" in response.text, path


def test_api_paths_never_fall_back_to_html(tmp_path: Path) -> None:
    """接口拼错时要看到 404，而不是一份"看起来加载成功"的页面。"""
    app, _ = make_site(tmp_path)
    client = TestClient(app)

    assert client.get("/api/health").json() == {"status": "ok"}
    missing = client.get("/api/does-not-exist")
    assert missing.status_code == 404
    assert "WeLearn" not in missing.text
    assert missing.json()["detail"] == "接口不存在"


def test_static_mount_cannot_escape_the_dist_directory(tmp_path: Path) -> None:
    app, _ = make_site(tmp_path)
    client = TestClient(app)

    for path in ["/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt"]:
        response = client.get(path)
        assert "TOP-SECRET" not in response.text, path


@pytest.mark.parametrize("missing", ["/subjects/abc"])
def test_unknown_non_api_path_returns_index(tmp_path: Path, missing: str) -> None:
    app, _ = make_site(tmp_path)
    assert TestClient(app).get(missing).status_code == 200


def test_head_requests_are_served_without_a_body(tmp_path: Path) -> None:
    """HEAD 探测（监控、反向代理）要能拿到状态码与缓存头，不能是 405。"""
    app, _ = make_site(tmp_path)
    client = TestClient(app)

    for path in ["/", "/subjects/abc/teach", "/assets/app-abc123.js"]:
        response = client.head(path)
        assert response.status_code == 200, path
        assert response.content == b"", path
        assert response.headers["cache-control"], path


def test_frontend_stays_reachable_when_access_token_is_set(tmp_path: Path) -> None:
    """设置访问口令后，前端页面本身仍必须能加载。

    否则交付形态会死锁：浏览器访问 `/` 先拿到 401 JSON，页面根本渲染不出来，
    用户也就没有地方输入口令了。数据仍然只在 `/api` 后面，前端拿到 401 再弹输入层。
    """
    from app.main import create_app

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>WeLearn</title>", encoding="utf-8")
    env = {"WELEARN_DATA_DIR": str(tmp_path / "data"), "WELEARN_WEB_DIST": str(dist)}

    with TestClient(create_app(env)) as client:
        current = client.get("/api/settings").json()["settings"]
        read_only = {"api_key", "model_plan", "access_token_enabled", "read_only_fields"}
        payload = {key: value for key, value in current.items() if key not in read_only}
        payload["access_token"] = "s3cret-token"
        assert client.put("/api/settings", json=payload).status_code == 200

        assert client.get("/api/subjects").status_code == 401
        for path in ["/", "/settings", "/subjects/abc/teach"]:
            page = client.get(path)
            assert page.status_code == 200, path
            assert "WeLearn" in page.text, path
