"""账户余额查询接口（FR-710）。

余额是**辅助信息**：查不到不能挡住任何主流程，所以失败一律是 ``200 + ok=false``，
而不是 4xx/5xx——前端据此决定"显示余额"还是"显示原因"，不必为可选能力写错误分支。

官方金额是**字符串**（如 ``"17.39"``），这里原样透传：``float`` 会在往返中丢精度，
而金额恰恰是最不能出错的地方（``docs/接口契约.md`` §8 记录的真实响应）。
"""

from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Depends

from app.api.deps import get_context
from app.infra.deepseek.errors import LlmError
from app.infra.logging.queue_logger import CHANNEL_APP
from app.main_context import AppContext

router = APIRouter(prefix="/api", tags=["billing"])

FALLBACK_CURRENCY = "CNY"
PROTOCOL_MESSAGE = "余额接口返回的结构与预期不符，暂时无法显示余额。"
UNEXPECTED_MESSAGE = "查询余额时发生未知错误，可稍后重试。"


def _as_amount(value: Any) -> str:
    """金额保持原样；只有官方改了类型时才转字符串（保持字段类型稳定，前端不必判类型）。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _first_balance_info(body: Mapping[str, Any]) -> Mapping[str, Any] | None:
    infos = body.get("balance_infos")
    if not isinstance(infos, list):
        return None
    for item in infos:
        if isinstance(item, Mapping):
            return item
    return None


def _ok_payload(body: Mapping[str, Any], info: Mapping[str, Any]) -> dict[str, Any]:
    currency = info.get("currency")
    available = body.get("is_available")
    return {
        "ok": True,
        "currency": currency if isinstance(currency, str) and currency else FALLBACK_CURRENCY,
        "total_balance": _as_amount(info.get("total_balance")),
        "granted_balance": _as_amount(info.get("granted_balance")),
        "topped_up_balance": _as_amount(info.get("topped_up_balance")),
        "is_available": available if isinstance(available, bool) else None,
        "message": "",
        "category": None,
    }


def _failure_payload(category: str, message: str) -> dict[str, Any]:
    """失败响应也把字段补齐：前端只按一套形状取值，不必区分成功/失败两种结构。"""
    return {
        "ok": False,
        "currency": None,
        "total_balance": "",
        "granted_balance": "",
        "topped_up_balance": "",
        "is_available": None,
        "message": message,
        "category": category,
    }


@router.get("/balance")
async def get_balance(ctx: AppContext = Depends(get_context)) -> dict[str, Any]:
    """查询账户余额（FR-710）。任何失败都返回 200，绝不影响主流程。"""
    try:
        body = await ctx.llm.get_balance()
        info = _first_balance_info(body)
        if info is None:
            raise LlmError.protocol(PROTOCOL_MESSAGE)
    except LlmError as exc:
        return _failure_payload(exc.category.value, exc.user_message)
    except Exception as exc:
        # 记下原始异常供排障，但不回给前端（可能是桩实现或意料外的客户端故障）
        ctx.logger.log(CHANNEL_APP, "WARNING", f"查询余额失败：{type(exc).__name__}: {exc}")
        return _failure_payload("unknown", UNEXPECTED_MESSAGE)

    return _ok_payload(body, info)
