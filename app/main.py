"""服务入口（T0.1）。

    python -m app.main
    uvicorn app.main:create_app --factory --reload
"""

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import billing, exports, health, ops, search, settings, subjects
from app.api.static_site import mount_frontend
from app.infra.access import token_matches
from app.main_context import AppContext, build_context
from app.services.learning.stack import build_learning_stack

ENV_HOST = "WELEARN_HOST"
ENV_PORT = "WELEARN_PORT"
ENV_WEB_DIST = "WELEARN_WEB_DIST"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
# 仓库内的构建产物默认位置；容器里由 Dockerfile 复制到同一相对位置
DEFAULT_WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"


def create_app(env: Mapping[str, str] | None = None) -> FastAPI:
    """创建应用。传入 ``env`` 便于测试在临时目录上装配。"""
    if env is None:
        # .env 仅填充尚未存在的环境变量，不会覆盖真实环境（FR-707 的优先级顺序）
        load_dotenv()

    context = build_context(env)
    stack = build_learning_stack(context)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await app.state.ctx.llm.aclose()
        app.state.ctx.close()

    app = FastAPI(title="WeLearn", version="0.1.0", lifespan=lifespan)
    app.state.ctx = context
    app.state.stack = stack
    app.include_router(health.router)
    app.include_router(ops.router)
    app.include_router(billing.router)
    app.include_router(exports.router)
    app.include_router(search.router)
    app.include_router(subjects.router)
    app.include_router(settings.router)
    # 必须放在所有 API 路由之后：兜底路由会吃掉未匹配的路径
    mount_frontend(app, _frontend_dir(env))

    @app.middleware("http")
    async def guard_access_token(request: Request, call_next):  # type: ignore[no-untyped-def]
        """可选访问口令（FR-903）。

        **每个请求现读设置**，而不是在启动时决定是否启用：口令可能在运行中被设置或清除，
        缓存下来就会出现"设置了口令却仍能匿名访问"这种最不该出现的窗口。

        跳过健康检查与口令探针：前者不含任何学习数据、且是"服务起没起来"的唯一入口；
        后者需要能被调用来判断"是我的口令错了，还是服务没开"。
        """
        ctx: AppContext = request.app.state.ctx
        stored = ctx.settings.access_token_hash
        path = request.url.path
        exempt = path == "/api/health" or path == "/api/auth/verify" or not path.startswith("/api")
        if stored and not exempt and request.method != "OPTIONS":
            token = request.headers.get("x-access-token") or request.cookies.get("access_token")
            if not token_matches(token, stored):
                return JSONResponse(status_code=401, content={"detail": "需要访问口令：请先在设置页填入口令。"})
        return await call_next(request)

    return app


def _frontend_dir(env: Mapping[str, str] | None) -> Path:
    """前端产物目录。允许用环境变量覆盖，便于容器把静态资源放到别处。"""
    source = env if env is not None else os.environ
    configured = (source.get(ENV_WEB_DIST) or "").strip()
    return Path(configured) if configured else DEFAULT_WEB_DIST


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
