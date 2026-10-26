"""前端静态资源托管（D1：一条命令起服务，前后端同一进程）。

开发期前端跑 vite（有 HMR 与代理），交付期用 `web/dist` 由后端直接提供——
这样用户不需要同时起两个进程，也不必理解"前后端分离"这件事。

**SPA 回退**：`/subjects/{id}/teach` 这类深链接在服务端不存在对应文件，
必须回 `index.html` 交给前端路由，否则刷新就 404。
`/api/*` 的未知路径**不能**回退成 HTML：那会让"接口不存在"表现为
"前端页面莫名加载成功"，排查时极具误导性。
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

# 带内容哈希的构建产物可以长缓存；入口文件必须每次校验，否则更新后用户会一直看到旧版本
IMMUTABLE_SECONDS = 31536000
INDEX_HEADERS = {"Cache-Control": "no-cache, must-revalidate"}
ASSET_HEADERS = {"Cache-Control": f"public, max-age={IMMUTABLE_SECONDS}, immutable"}


def mount_frontend(app: FastAPI, dist_dir: Path) -> bool:
    """把构建好的前端挂到应用上；目录不存在时返回 False（开发期即如此）。

    刻意**不用** ``StaticFiles`` 挂载：它无法给单个响应加缓存头，而"哈希产物长缓存、
    index 不缓存"正是发版后用户不必手动清缓存的关键。这里统一走一个兜底路由。
    """
    root = Path(dist_dir).resolve()
    index = root / "index.html"
    if not index.is_file():
        return False

    # 同时响应 HEAD：监控脚本与部分 CDN/反向代理用 HEAD 探测可用性，
    # 只注册 GET 会让它们拿到 405，看起来像"页面不存在"
    @app.api_route("/{full_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        if full_path == "api" or full_path.startswith("api/"):
            # 前端路由不会占用 /api，走到这里的只能是拼错的接口
            raise HTTPException(status_code=404, detail="接口不存在")

        candidate = (root / full_path).resolve()
        # 越界与目录都要挡掉：`../` 或指向目录的请求会落到下面的 index 回退
        if full_path and candidate.is_file() and candidate.is_relative_to(root):
            headers = ASSET_HEADERS if full_path.startswith("assets/") else None
            return FileResponse(candidate, headers=headers)
        return FileResponse(index, headers=INDEX_HEADERS)

    return True
