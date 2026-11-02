"""健康检查（T0.1 冒烟端点、T3.6 运维可见性）。

用途是回答"现在这个服务能不能用、用的是哪套配置"：存储可读可写、日志目录可写、
模型配置（不含 Key）、当前用量与生效的功能开关。响应中**永不出现 API Key 明文或密文**（FR-708）。
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.infra.llm.providers import profile_for
from app.main_context import AppContext
from app.services.llm.usage import UsageRecorder

router = APIRouter(tags=["system"])

# 写测试用的探针文件名：固定名字而不是随机名，万一进程被杀也只会留下一个可覆盖的文件
WRITE_PROBE_FILENAME = ".health-write-probe"


@router.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    """返回服务、存储、日志、模型与用量的当前状态。

    这里是"降级而非崩溃"的示范点（NFR-107）：任何一项检查失败都不抛异常，
    接口仍返回 200 并标记 ``status = degraded``，同时把失败项写进 ``problems``，便于用户排查。
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

    # 真写一个文件再删掉：目录权限、磁盘只读、句柄耗尽都只有真写才看得出来（FR-909）
    writable, write_detail = _probe_writable(context.paths.root)
    if not writable:
        problems.append("data_dir")

    log_writable, log_detail = _probe_writable(context.logger.log_dir)
    if not log_writable:
        problems.append("logs")

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

    # 用量是只读聚合，不外呼模型；它坏掉（例如库被锁）不该把健康检查带崩
    usage: dict[str, Any] = {"calls": 0, "total_tokens": 0, "cache_hit_rate": 0.0}
    try:
        summary = UsageRecorder(context.db).summary()
        usage = {
            "calls": summary.calls,
            "total_tokens": summary.total_tokens,
            "cache_hit_rate": round(summary.cache_hit_rate, 4),
        }
    except Exception as exc:
        usage["detail"] = f"{type(exc).__name__}: {exc}"
        problems.append("usage")

    storage: dict[str, Any] = {
        "ok": storage_ok,
        "detail": storage_detail,
        "data_dir": str(context.paths.root),
        "schema_version": context.schema_version,
        "writable": writable,
    }
    if not writable:
        storage["writable_detail"] = write_detail

    logs: dict[str, Any] = {
        "dir": str(context.logger.log_dir),
        "level": settings.log_level,
        "writable": log_writable,
    }
    if not log_writable:
        logs["writable_detail"] = log_detail

    return {
        "status": "ok" if not problems else "degraded",
        "problems": problems,
        "storage": storage,
        "model": {
            "provider": settings.provider.value,
            "base_url": settings.base_url,
            "default": settings.model_default,
            "pro": settings.model_pro,
            "thinking_enabled": settings.thinking_enabled,
            "reasoning_effort": settings.reasoning_effort.value,
            "max_tokens": settings.max_tokens,
            "timeout_seconds": settings.timeout_seconds,
            "capabilities": profile_for(settings.provider).capabilities(),
        },
        "api_key": api_key,
        "logging": {
            "level": settings.log_level,
            "log_dir": str(context.logger.log_dir),
        },
        "logs": logs,
        "llm": {
            "audit_enabled": settings.audit_enabled,
            # 复用一次 api_key 判定即可：为"看起来更真实"多打一次外呼只会让健康检查变慢变脆
            "configured": bool(api_key["configured"]),
        },
        "usage": usage,
        "features": {
            "use_recommended_overrides": settings.use_recommended_overrides,
            "show_reasoning": settings.show_reasoning,
            "thinking_enabled": settings.thinking_enabled,
        },
    }


def _probe_writable(directory: Path) -> tuple[bool, str]:
    """真实写入并清理，返回（是否可写, 说明）。"""
    probe = directory / WRITE_PROBE_FILENAME
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"
