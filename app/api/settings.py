"""设置、API Key 与模型列表接口（p2-api-sse，FR-707~FR-709、FR-901、FR-902）。

安全约定（FR-708）：响应里**只出现掩码**，明文与密文都不返回；日志同理。
"""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_context, get_stack
from app.contracts.schemas import LlmPurpose
from app.infra.access import hash_token
from app.infra.llm.providers import DEFAULT_PROVIDER, ProviderKind, profile_for, provider_options
from app.infra.settings import AppSettings, PurposeOverride, purpose_overrides_payload
from app.main_context import AppContext
from app.services.learning.stack import LearningStack
from app.services.llm.cost import effective_price_table
from app.services.llm.policy import ModelPolicy, describe_model_plan
from app.services.llm.usage import UsageSummary

router = APIRouter(prefix="/api", tags=["settings"])


class SettingsRequest(BaseModel):
    """可保存的设置项。``data_dir`` 不在其中：它只由环境变量决定（改了会导致找不到数据库）。"""

    # 接口类型（P6）：openai = 通用 OpenAI 兼容；deepseek = 额外开启思考开关/缓存统计/余额
    provider: ProviderKind = DEFAULT_PROVIDER
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
    # 区间越界在这里被 pydantic 拦成 422（FR-718、AC-30：提交不报错但不生效的说法只针对
    # "思考模式下参数被忽略"，取值越界仍属于非法输入）
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.01, le=1.0)
    audit_enabled: bool = False
    use_recommended_overrides: bool = False
    purpose_overrides: dict[LlmPurpose, PurposeOverride] = Field(default_factory=dict)
    # P4：复习排期阈值、日志轮转与保留、工具调用
    review_interval_days: int = Field(default=7, ge=1, le=365)
    log_max_mb: int = Field(default=5, ge=1, le=100)
    log_retention_days: int = Field(default=14, ge=1, le=365)
    tools_enabled: bool = False
    # 单价表（FR-706）：逐模型覆盖，留空则使用内置默认价目；单位「元 / 百万 token」
    price_table: dict[str, dict[str, float]] = Field(default_factory=dict)
    # 访问口令（FR-903）：None = 保持现状，"" = 清除，非空 = 设为新口令。
    # 用三态而不是布尔，是为了让"只想改别的设置"的请求不会顺手把口令清掉。
    access_token: str | None = Field(default=None, max_length=200)


class ApiKeyRequest(BaseModel):
    api_key: str = ""


class TestConnectionRequest(BaseModel):
    model: str | None = None


def _settings_payload(settings: AppSettings, ctx: AppContext, policy: ModelPolicy) -> dict[str, Any]:
    profile = profile_for(settings.provider)
    return {
        "provider": settings.provider.value,
        # 只读：当前接口类型支持哪些能力，前端据此灰掉不生效的控件（P6）
        "provider_capabilities": profile.capabilities(),
        "provider_options": provider_options(),
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
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "audit_enabled": settings.audit_enabled,
        "use_recommended_overrides": settings.use_recommended_overrides,
        "review_interval_days": settings.review_interval_days,
        "log_max_mb": settings.log_max_mb,
        "log_retention_days": settings.log_retention_days,
        "tools_enabled": settings.tools_enabled,
        # 单价表回的是**生效值**（内置默认 + 用户覆盖），否则界面上一片空白，
        # 用户没法判断"这次估算到底按什么价在算"（FR-706）
        "price_table": effective_price_table(settings.price_table),
        # 口令只报告"是否已设置"，本身永不回传（FR-903、FR-708 同一原则）
        "access_token_enabled": bool(settings.access_token_hash),
        # 按用途覆盖：五个用途全部补齐，"继承"显式写出来，前端不必自己补默认值（FR-703）
        "purpose_overrides": purpose_overrides_payload(settings),
        # 只读：每个用途**实际**会用的参数。覆盖项写进去不等于生效（可能被推荐项或"继承"
        # 改变结果），看不出来用户就会以为自己的设置没生效（FR-703、FR-718）
        "model_plan": describe_model_plan(policy),
        # 数据目录只读；这两项告诉前端"哪些东西改了不生效"（FR-718）
        "read_only_fields": ["data_dir"],
        # 被环境变量覆盖的字段：界面上的修改不会生效。不暴露它的话，
        # 用户会以为"选了 OpenAI 兼容却还是 DeepSeek 的地址"是程序坏了
        "env_overridden": ctx.settings_service.diagnostics().env_overridden,
        "api_key": {
            "configured": ctx.secrets.has_api_key(),
            "source": ctx.secrets.source(),
            "masked": ctx.secrets.masked_hint(),
        },
    }


@router.get("/settings")
def get_settings(
    ctx: AppContext = Depends(get_context),
    stack: LearningStack = Depends(get_stack),
) -> dict[str, Any]:
    return {"settings": _settings_payload(ctx.settings, ctx, stack.policy)}


@router.put("/settings")
def put_settings(
    payload: SettingsRequest,
    ctx: AppContext = Depends(get_context),
    stack: LearningStack = Depends(get_stack),
) -> dict[str, Any]:
    """保存设置。非法值由 pydantic 拦下；区间颠倒由服务层回退默认并记录日志（FR-902）。"""
    from pydantic import ValidationError

    current = ctx.settings
    values = payload.model_dump()
    token = values.pop("access_token", None)
    # 三态口令：None 保持现状、"" 清除、非空设为新口令（只存哈希）
    token_hash = current.access_token_hash if token is None else hash_token(token)
    try:
        candidate = AppSettings(
            **values,
            data_dir=current.data_dir,
            access_token_hash=token_hash,
        )
    except ValidationError as exc:
        from fastapi import HTTPException, status

        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()[:3]) from exc

    saved = ctx.settings_service.save(candidate)
    ctx.refresh_settings(saved)
    stack.refresh_settings(saved)
    return {"settings": _settings_payload(saved, ctx, stack.policy)}


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
    from app.infra.llm.errors import LlmError

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


def _summary_payload(summary: UsageSummary) -> dict[str, Any]:
    """一份聚合结果的对外形状：总量、按用途、按主题共用，避免三处口径写歪。"""
    return {
        "calls": summary.calls,
        "prompt_tokens": summary.prompt_tokens,
        "cached_prompt_tokens": summary.cached_prompt_tokens,
        "cache_miss_tokens": summary.cache_miss_tokens,
        "completion_tokens": summary.completion_tokens,
        "reasoning_tokens": summary.reasoning_tokens,
        "total_tokens": summary.total_tokens,
        "cache_hit_rate": round(summary.cache_hit_rate, 4),
        "duration_ms": summary.duration_ms,
    }


def _cost_payload(cost: Any) -> dict[str, Any]:
    """费用估算的对外形状（FR-706）。金额保留 6 位：单次调用常常只有几分钱。"""
    return {
        "amount": round(cost.amount, 6),
        "currency": cost.currency,
        "peak_tokens": cost.peak_tokens,
        "off_peak_tokens": cost.off_peak_tokens,
        "total_tokens": cost.total_tokens,
    }


@router.get("/usage")
def usage_summary(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """全局用量、按用途/按主题聚合与缓存命中提示（FR-705）。

    既有字段保持不变（向后兼容），新增字段只做加法。
    """
    from app.services.llm.usage import UsageRecorder, cache_hint

    recorder = UsageRecorder(ctx.db)
    summary = recorder.summary()
    cost = recorder.cost_summary(
        price_table=ctx.settings.price_table,
        # 空闲折扣只有 DeepSeek 有；其他接口类型按固定价算，否则金额会少一半（P6）
        off_peak_multiplier=profile_for(ctx.settings.provider).off_peak_multiplier,
    )
    return {
        "calls": summary.calls,
        "prompt_tokens": summary.prompt_tokens,
        "cached_prompt_tokens": summary.cached_prompt_tokens,
        "cache_miss_tokens": summary.cache_miss_tokens,
        "completion_tokens": summary.completion_tokens,
        "reasoning_tokens": summary.reasoning_tokens,
        "total_tokens": summary.total_tokens,
        "cache_hit_rate": round(summary.cache_hit_rate, 4),
        "duration_ms": summary.duration_ms,
        # 逐行按"那条记录发生的时刻"结算高峰/空闲价，而不是拿当前时刻一刀切（FR-706）
        "cost": _cost_payload(cost),
        "by_purpose": {purpose: _summary_payload(item) for purpose, item in recorder.by_purpose().items()},
        "by_subject": [
            {
                "subject_id": item.subject_id,
                "title": item.title,
                "calls": item.summary.calls,
                "total_tokens": item.summary.total_tokens,
                "cache_hit_rate": round(item.summary.cache_hit_rate, 4),
                "duration_ms": item.summary.duration_ms,
            }
            for item in recorder.by_subject()
        ],
        "cache_hint": cache_hint(summary),
    }
