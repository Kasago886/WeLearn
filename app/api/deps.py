"""路由层依赖（p2-api-sse）。"""

from fastapi import HTTPException, Request, status

from app.main_context import AppContext
from app.services.learning.stack import LearningStack


def get_context(request: Request) -> AppContext:
    ctx = getattr(request.app.state, "ctx", None)
    if not isinstance(ctx, AppContext):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="服务未完成初始化")
    return ctx


def get_stack(request: Request) -> LearningStack:
    stack = getattr(request.app.state, "stack", None)
    if not isinstance(stack, LearningStack):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="服务未完成初始化")
    return stack


def require_subject(stack: LearningStack, subject_id: str) -> None:
    """主题不存在时抛 404。

    在建立 SSE 流**之前**做这类校验：流一旦开始就无法再改 HTTP 状态码，
    而"主题不存在"是请求级错误，应该让前端拿到 404 而不是一条 error 事件。
    """
    if stack.repos.subjects.get(subject_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="主题不存在")
