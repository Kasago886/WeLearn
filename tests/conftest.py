"""公共测试夹具。

所有测试默认**不访问真实 API**：HTTP 层用 ``httpx.MockTransport`` 替换，
响应样本放在 ``tests/fixtures/llm/``（由 scripts/llm_smoke.py --record 生成真实样本）。
"""

import json
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio

from app.infra.llm.client import LlmClient
from app.main_context import AppContext, build_context

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "llm"

ClientFactory = Callable[..., LlmClient]


def fixture_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def fixture_lines(name: str) -> list[str]:
    return fixture_text(name).splitlines()


def fixture_json(name: str) -> dict[str, Any]:
    loaded = json.loads(fixture_text(name))
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    """把数据目录指向临时目录，避免污染真实工作目录。"""
    return {"WELEARN_DATA_DIR": str(tmp_path / "data")}


@pytest.fixture
def context(env: dict[str, str]) -> Iterator[AppContext]:
    """完整依赖容器（临时数据目录 + 独立 SQLite）。"""
    ctx = build_context(env)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest_asyncio.fixture
async def make_client() -> AsyncIterator[ClientFactory]:
    """构造注入了 MockTransport 的客户端，并在测试结束后关闭底层连接。"""
    opened: list[httpx.AsyncClient] = []

    def _make(handler: Callable[[httpx.Request], httpx.Response], **overrides: Any) -> LlmClient:
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        params: dict[str, Any] = {
            "base_url_getter": lambda: "https://api.deepseek.com",
            "api_key_getter": lambda: "sk-test-abcdef123456",
            "backoff_base_seconds": 0.0,
            "http_client": http,
        }
        params.update(overrides)
        opened.append(http)
        return LlmClient(**params)

    yield _make

    for http in opened:
        await http.aclose()
