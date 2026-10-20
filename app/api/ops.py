"""运维与访问口令接口（FR-903、FR-907）。

口令校验本身在中间件里（每个请求都要过），这里只提供两件前端需要的东西：

- ``GET /api/auth/verify``：确认当前口令是否有效、以及是否启用了口令。
  中间件会先拦掉错误的请求，所以走到这里就说明通过（或未启用）。
- 日志清理与统计：给设置页一个"清理日志"入口，并让用户看得到日志占了多少空间。
"""

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_context
from app.infra.access import token_matches
from app.main_context import AppContext

router = APIRouter(prefix="/api", tags=["ops"])


@router.get("/auth/verify")
def verify(request: Request, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """口令探针：**永远 200**，用 ``ok`` 表达凭据是否正确。

    特意不让中间件把它拦成 401：前端需要区分"口令错"与"服务没起/口令根本没启用"，
    而这两种情况在 401 里长得一模一样。
    """
    stored = ctx.settings.access_token_hash
    token = request.headers.get("x-access-token") or request.cookies.get("access_token")
    return {"ok": token_matches(token, stored), "required": bool(stored)}


@router.get("/logs")
def log_stats(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """日志目录概览（大小、文件数、保留策略），用于设置页展示。"""
    logger = ctx.logger
    files = []
    total = 0
    for path in sorted(logger.log_dir.glob("*.log")):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        total += size
        files.append({"name": path.name, "size": size})
    return {
        "log_dir": str(logger.log_dir),
        "files": files,
        "total_bytes": total,
        "max_bytes": ctx.settings.log_max_mb * 1024 * 1024,
        "retention_days": ctx.settings.log_retention_days,
    }


@router.post("/logs/clean")
def clean_logs(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """清理日志：截断当前通道文件、删除归档。

    先 flush 再清理，避免队列里待写的日志在清理之后又落进文件，让用户以为"没清干净"。
    """
    ctx.logger.flush(timeout=2.0)
    removed, freed = ctx.logger.clean()
    return {"removed_files": removed, "freed_bytes": freed}
