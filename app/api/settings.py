"""设置、API Key 与模型列表接口（p2-api-sse，FR-707~FR-709、FR-901、FR-902）。

安全约定（FR-708）：响应里**只出现掩码**，明文与密文都不返回；日志同理。
"""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_context, get_stack
from app.infra.settings import AppSettings
from app.main_context import AppContext
from app.services.learning.stack import LearningStack

router = APIRouter(prefix="/api", tags=["settings"])


class SettingsRequest(BaseModel):
    """可保存的设置项。``data_dir`` 不在其中：它只由环境变量决定（改了会导致找不到数据库）。"""

    base_url: str = Field(min_length=1)
    model_default: str = Field(min_length=1)
    model_pro: str = Field(min_length=1)
    thinking_enabled: bool = True
    reasoning_effort: str = "high"
    max_tokens: int = Field(default=8192, ge=256, le=393216)
    timeout_seconds: int = Field(default=60, ge=30, le=600)
    max_retries: int = Field(default=2, ge=0, le=5)
    assess_max_turns: int = Field(default=6, ge=3, le=12)
    path_min_nodes: int = Field(default=3, ge=3, le=15)
    path_max_nodes: int = Field(default=15, ge=3, le=15)
    node_max_retries: int = Field(default=2, ge=0, le=5)
    compress_threshold_rounds: int = Field(default=40, ge=4)
    compress_round_count: int = Field(default=10, ge=1)
    show_reasoning: bool = False
    log_level: str = "INFO"


class ApiKeyRequest(BaseModel):
    api_key: str = ""


class TestConnectionRequest(BaseModel):
    model: str | None = None


def _settings_payload(settings: AppSettings, ctx: AppContext) -> dict[str, Any]:
    return {
        "base_url": settings.base_url,
        "model_default": settings.model_default,
        "model_pro": settings.model_pro,
        "thinking_enabled": settings.thinking_enabled,
        "reasoning_effort": settings.reasoning_effort.value,
        "max_tokens": settings.max_tokens,
        "timeout_seconds": settings.timeout_seconds,
        "max_retries": settings.max_retries,
        "assess_max_turns": settings.assess_max_turns,
        "path_min_nodes": settings.path_min_nodes,
        "path_max_nodes": settings.path_max_nodes,
        "node_max_retries": settings.node_max_retries,
        "compress_threshold_rounds": settings.compress_threshold_rounds,
        "compress_round_count": settings.compress_round_count,
        "show_reasoning": settings.show_reasoning,
        "log_level": settings.log_level,
        "data_dir": settings.data_dir,
        # 数据目录只读；这两项告诉前端"哪些东西改了不生效"（FR-718）
        "read_only_fields": ["data_dir"],
        "api_key": {
            "configured": ctx.secrets.has_api_key(),
            "source": ctx.secrets.source(),
            "masked": ctx.secrets.masked_hint(),
        },
    }


@router.get("/settings")
def get_settings(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    return {"settings": _settings_payload(ctx.settings, ctx)}


@router.put("/settings")
def put_settings(
    payload: SettingsRequest,
    ctx: AppContext = Depends(get_context),
    stack: LearningStack = Depends(get_stack),
) -> dict[str, Any]:
    """保存设置。非法值由 pydantic 拦下；区间颠倒由服务层回退默认并记录日志（FR-902）。"""
    from pydantic import ValidationError

    current = ctx.settings
    try:
        candidate = AppSettings(
            **payload.model_dump(),
            data_dir=current.data_dir,
            price_table=current.price_table,
        )
    except ValidationError as exc:
        from fastapi import HTTPException, status

        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()[:3]) from exc

    saved = ctx.settings_service.save(candidate)
    ctx.refresh_settings(saved)
    stack.refresh_settings(saved)
    return {"settings": _settings_payload(saved, ctx)}


@router.put("/settings/api-key")
def put_api_key(payload: ApiKeyRequest, ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """保存或清除 API Key；只返回掩码（FR-708）。"""
    masked = ctx.secrets.set_api_key(payload.api_key)
    return {
        "configured": ctx.secrets.has_api_key(),
        "source": ctx.secrets.source(),
        "masked": masked,
    }


@router.post("/settings/test-connection")
async def test_connection(
    payload: TestConnectionRequest,
    ctx: AppContext = Depends(get_context),
) -> dict[str, Any]:
    """测试连接（FR-709）。返回可分类的失败原因，便于给出可操作提示。"""
    model = payload.model or ctx.settings.model_default
    result = await ctx.llm.test_connection(model=model)
    return {
        "ok": result.ok,
        "message": result.message,
        "category": result.category,
        "latency_ms": result.latency_ms,
        "model": model,
    }


@router.get("/models")
async def list_models(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """可用模型与元数据（FR-702）：避免把模型名写死导致官方改版后失效。"""
    from app.infra.deepseek.errors import LlmError

    try:
        models = await ctx.llm.list_models()
    except LlmError as exc:
        return {"ok": False, "message": exc.user_message, "category": exc.category.value, "items": []}

    return {
        "ok": True,
        "message": "",
        "items": [
            {
                "id": model.id,
                "name": model.name,
                "context_window": model.context_window,
                "max_output_tokens": model.max_output_tokens,
                "effort_levels": list(model.effort_levels),
                "default_effort": model.default_effort,
            }
            for model in models
        ],
    }


@router.get("/usage")
def usage_summary(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """全局用量与缓存命中率（FR-705）。命中率长期为 0 往往意味着前缀被改写。"""
    from app.services.llm.usage import UsageRecorder

    summary = UsageRecorder(ctx.db).summary()
    return {
        "calls": summary.calls,
        "prompt_tokens": summary.prompt_tokens,
        "cached_prompt_tokens": summary.cached_prompt_tokens,
        "cache_miss_tokens": summary.cache_miss_tokens,
        "completion_tokens": summary.completion_tokens,
        "reasoning_tokens": summary.reasoning_tokens,
        "total_tokens": summary.total_tokens,
        "cache_hit_rate": round(summary.cache_hit_rate, 4),
    }
