"""服务入口（T0.1）。

    python -m app.main
    uvicorn app.main:create_app --factory --reload
"""

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from app.api import health
from app.main_context import AppContext, build_context

ENV_HOST = "WELEARN_HOST"
ENV_PORT = "WELEARN_PORT"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def create_app(env: Mapping[str, str] | None = None) -> FastAPI:
    """创建应用。传入 ``env`` 便于测试在临时目录上装配。"""
    if env is None:
        # .env 仅填充尚未存在的环境变量，不会覆盖真实环境（FR-707 的优先级顺序）
        load_dotenv()

    context = build_context(env)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await app.state.ctx.llm.aclose()
        app.state.ctx.close()

    app = FastAPI(title="WeLearn", version="0.1.0", lifespan=lifespan)
    app.state.ctx = context
    app.include_router(health.router)
    return app


def get_context(app: FastAPI) -> AppContext:
    """从应用上取回依赖容器。"""
    ctx = app.state.ctx
    if not isinstance(ctx, AppContext):
        raise RuntimeError("应用未完成初始化，缺少依赖容器")
    return ctx


def main() -> None:
    """命令行启动。"""
    import uvicorn

    host = (os.environ.get(ENV_HOST) or DEFAULT_HOST).strip()
    port = int(os.environ.get(ENV_PORT) or DEFAULT_PORT)
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
