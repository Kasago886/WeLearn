"""健康检查（T0.1 冒烟端点）。

当前只用于验证"服务起得来、数据能落盘、配置读得到"。T3.6 会在保持字段兼容的前提下扩展
（例如补充 LLM 连通性、日志目录状态）。响应中**永不出现 API Key 明文或密文**（FR-708）。
"""

from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.main_context import AppContext

router = APIRouter(tags=["system"])


@router.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    """返回服务、存储与当前模型配置概览。

    这里是"降级而非崩溃"的示范点（NFR-107）：存储或密钥读取失败时，
    接口仍返回 200 并标记 ``status = degraded``，同时给出可读原因，便于用户排查。
    """
    context: AppContext = request.app.state.ctx
    problems: list[str] = []

    storage_ok = True
    storage_detail = "ok"
    try:
        with context.db.session() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:
        storage_ok = False
        storage_detail = f"{type(exc).__name__}: {exc}"
        problems.append("storage")

    settings = context.settings
    api_key: dict[str, Any] = {"configured": False, "source": "none", "masked": None}
    try:
        api_key = {
            "configured": context.secrets.has_api_key(),
            "source": context.secrets.source(),
            "masked": context.secrets.masked_hint(),
        }
    except Exception as exc:
        api_key["detail"] = f"{type(exc).__name__}: {exc}"
        problems.append("api_key")

    return {
        "status": "ok" if not problems else "degraded",
        "problems": problems,
        "storage": {
            "ok": storage_ok,
            "detail": storage_detail,
            "data_dir": str(context.paths.root),
            "schema_version": context.schema_version,
        },
        "model": {
            "base_url": settings.base_url,
            "default": settings.model_default,
            "pro": settings.model_pro,
            "thinking_enabled": settings.thinking_enabled,
            "reasoning_effort": settings.reasoning_effort.value,
            "max_tokens": settings.max_tokens,
            "timeout_seconds": settings.timeout_seconds,
        },
        "api_key": api_key,
        "logging": {
            "level": settings.log_level,
            "log_dir": str(context.logger.log_dir),
        },
    }
