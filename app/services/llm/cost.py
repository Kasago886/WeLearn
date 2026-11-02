"""成本估算（FR-706）。

三条必须守住的规则，都是官方定价的口径：

1. **单价可配置**：官方价格会调整，写死在代码里的数字迟早会错。内置表只是参考值，
   设置页可整体覆盖。
2. **高峰 / 空闲双价**：北京时间周一至周五 9:00–12:00、14:00–18:00 为高峰，
   其余时段（含周末与法定节假日）为空闲，**空闲价是高峰价的一半**。
   因此表里存的是高峰价，结算时按请求实际发生的时刻打折。
   空闲折扣是 DeepSeek 的计价规则；其他接口类型按固定价结算（系数 1.0），
   由 :mod:`app.infra.llm.providers` 的档案给出。
3. **按请求实际发生的时段计价**：不能拿"当前时刻"去估算历史用量，
   否则同一批记录在晚上和白天的估算结果会不一样。

单位统一为「元 / 百万 token」，与官方定价页一致，避免换算时的小数点错位。
"""

from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, timezone

BEIJING = timezone(timedelta(hours=8))

# 中国不实行夏令时，固定 +08:00 即可；用 zoneinfo 会引入对系统时区库的依赖
PEAK_WINDOWS: tuple[tuple[time, time], ...] = ((time(9, 0), time(12, 0)), (time(14, 0), time(18, 0)))

DEFAULT_PRICE_TABLE: dict[str, dict[str, float]] = {
    # 参考价（元 / 百万 token，高峰价）。以官方定价页为准 —— 这里只是让"没配置"时
    # 也能给出数量级正确的估算，而不是显示 0 元。
    "deepseek-flash": {"cache_hit": 0.2, "cache_miss": 2.0, "output": 3.0},
    "deepseek-v4-pro": {"cache_hit": 1.0, "cache_miss": 4.0, "output": 12.0},
}

OFF_PEAK_MULTIPLIER = 0.5

PRICE_KEYS = ("cache_hit", "cache_miss", "output")


@dataclass(frozen=True)
class CostEstimate:
    """一次或一组用量的费用估算。"""

    amount: float
    currency: str = "CNY"
    peak_tokens: int = 0
    off_peak_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.peak_tokens + self.off_peak_tokens


def effective_price_table(overrides: dict[str, dict[str, float]] | None = None) -> dict[str, dict[str, float]]:
    """内置默认价 + 用户覆盖（逐模型深合并）。

    深合并而不是整体替换：用户只想调一个模型的输出价时，不该被迫把另外两个价也抄一遍
    ——抄漏的那项会变成 0，估算就偏了。
    """
    table = {model: dict(prices) for model, prices in DEFAULT_PRICE_TABLE.items()}
    for model, prices in (overrides or {}).items():
        if not isinstance(prices, dict):
            continue
        merged = table.setdefault(str(model), {"cache_hit": 0.0, "cache_miss": 0.0, "output": 0.0})
        for key, value in prices.items():
            if key in PRICE_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = float(value)
    return table


def is_peak(moment: datetime) -> bool:
    """是否处于高峰时段（按北京时间，周末一律空闲）。"""
    local = to_beijing(moment)
    if local.weekday() >= 5:
        return False
    current = local.time()
    return any(start <= current < end for start, end in PEAK_WINDOWS)


def to_beijing(moment: datetime) -> datetime:
    """把任意时间转成北京时间；naive 时间按 UTC 解释（与库里的存储约定一致）。"""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(BEIJING)


def multiplier_at(moment: datetime) -> float:
    return 1.0 if is_peak(moment) else OFF_PEAK_MULTIPLIER


def estimate_cost(
    *,
    model: str,
    cached_prompt_tokens: int = 0,
    cache_miss_tokens: int = 0,
    output_tokens: int = 0,
    at: datetime | None = None,
    price_table: dict[str, dict[str, float]] | None = None,
    off_peak_multiplier: float = OFF_PEAK_MULTIPLIER,
) -> CostEstimate:
    """按"这一批 token 发生在 ``at`` 时刻"估算费用。

    同一个模型的命中/未命中输入分开计价：混在一起按未命中价算会高估，
    反过来会低估，两种都会让"这个主题花了多少钱"变得不可信。

    ``off_peak_multiplier`` 由接口类型决定：DeepSeek 空闲半价（0.5），
    其他供应商按固定价（1.0）——给 OpenAI 兼容服务套上夜间五折只会把金额算少一半。
    """
    table = effective_price_table(price_table)
    prices = table.get(model)
    if prices is None:
        # 未知模型：用保守的默认模型价，避免显示 0 元让人误以为免费
        fallback = next(iter(table.values()), {"cache_hit": 0.0, "cache_miss": 0.0, "output": 0.0})
        prices = fallback

    moment = at or datetime.now(UTC)
    peak = is_peak(moment)
    factor = 1.0 if peak else max(0.0, off_peak_multiplier)

    tokens = max(0, cached_prompt_tokens) + max(0, cache_miss_tokens) + max(0, output_tokens)
    amount = (
        max(0, cached_prompt_tokens) / 1_000_000 * prices["cache_hit"]
        + max(0, cache_miss_tokens) / 1_000_000 * prices["cache_miss"]
        + max(0, output_tokens) / 1_000_000 * prices["output"]
    ) * factor

    if peak:
        return CostEstimate(amount=amount, peak_tokens=tokens, off_peak_tokens=0)
    return CostEstimate(amount=amount, peak_tokens=0, off_peak_tokens=tokens)


def combine(estimates: list[CostEstimate]) -> CostEstimate:
    """把多条估算相加（保留高峰/空闲 token 的分项，用于展示"为什么便宜了"）。"""
    return CostEstimate(
        amount=sum(item.amount for item in estimates),
        peak_tokens=sum(item.peak_tokens for item in estimates),
        off_peak_tokens=sum(item.off_peak_tokens for item in estimates),
    )
