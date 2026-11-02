"""用量落库与聚合（FR-705、FR-720）。

``prompt_cache_hit_tokens`` 与 ``prompt_cache_miss_tokens`` 都记录：
命中率异常偏低往往意味着上下文前缀被改写（例如压缩重排），是排查费用问题的关键线索。

聚合口径只有一处（:func:`_aggregate_columns`）：总量、按用途、按主题都从它派生，
否则"总量 100、按用途加起来 90"这种对不上账的情况迟早会出现。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.contracts.schemas import LlmPurpose
from app.infra.llm.protocol import TokenUsage
from app.infra.storage.db import Database
from app.infra.storage.models import SubjectRow, UsageRecordRow
from app.services.llm.cost import (
    OFF_PEAK_MULTIPLIER,
    CostEstimate,
    combine,
    effective_price_table,
    estimate_cost,
)

# 聚合金额的保留位数：只用来抹掉浮点累加的尾巴（0.00035700000000000004），
# 展示精度由前端决定，这里不做四舍五入到分。
COST_PRECISION = 6


@dataclass(frozen=True)
class UsageSummary:
    """聚合用量。"""

    calls: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_miss_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    # 累计模型调用耗时（毫秒）。FR-417 的完成报告要用它报"累计耗时"。
    duration_ms: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率；无输入时返回 0。"""
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cached_prompt_tokens / self.prompt_tokens


@dataclass(frozen=True)
class SubjectUsage:
    """按主题聚合的用量。``title`` 取不到时用 ``subject_id`` 顶上，不留空给前端。"""

    subject_id: str
    title: str
    summary: UsageSummary


@dataclass(frozen=True)
class SubjectCost:
    """按主题聚合的费用。结构与 :class:`SubjectUsage` 平行，``title`` 取不到时用 id。"""

    subject_id: str
    title: str
    cost: CostEstimate


CACHE_HINT_NO_HIT = (
    "缓存命中率长期为 0。可能原因：历史消息前缀被改写（逐轮重写历史）、"
    "上下文压缩重排了前缀导致缓存冷启动、或单次会话尚未累积到官方缓存单元大小。"
)


def cache_hint(summary: UsageSummary) -> str | None:
    """命中率异常偏低时的提示（FR-705）；无调用或已有命中时返回 None。

    只报"可能有这些原因"，不下断言：从用量数字本身无法区分是哪种原因造成的。
    """
    if summary.calls <= 0 or summary.cached_prompt_tokens > 0:
        return None
    return CACHE_HINT_NO_HIT


def _aggregate_columns() -> list[Any]:
    """8 项聚合表达式（与 :class:`UsageSummary` 字段一一对应）。"""
    return [
        func.count(UsageRecordRow.id),
        func.coalesce(func.sum(UsageRecordRow.prompt_tokens), 0),
        func.coalesce(func.sum(UsageRecordRow.cached_prompt_tokens), 0),
        func.coalesce(func.sum(UsageRecordRow.cache_miss_tokens), 0),
        func.coalesce(func.sum(UsageRecordRow.completion_tokens), 0),
        func.coalesce(func.sum(UsageRecordRow.reasoning_tokens), 0),
        func.coalesce(func.sum(UsageRecordRow.total_tokens), 0),
        func.coalesce(func.sum(UsageRecordRow.duration_ms), 0),
    ]


def _summary_from_values(values: Sequence[Any]) -> UsageSummary:
    """把 :func:`_aggregate_columns` 的结果按序装配成 :class:`UsageSummary`。"""
    return UsageSummary(
        calls=int(values[0]),
        prompt_tokens=int(values[1]),
        cached_prompt_tokens=int(values[2]),
        cache_miss_tokens=int(values[3]),
        completion_tokens=int(values[4]),
        reasoning_tokens=int(values[5]),
        total_tokens=int(values[6]),
        duration_ms=int(values[7] or 0),
    )


def _subject_titles(session: Session, subject_ids: Sequence[str]) -> dict[str, str]:
    if not subject_ids:
        return {}
    rows = session.execute(select(SubjectRow.id, SubjectRow.title).where(SubjectRow.id.in_(subject_ids))).all()
    return {str(row[0]): str(row[1]) for row in rows}


def _cost_columns() -> list[Any]:
    """计费需要的列（模型 + 三个 token 数 + 事件时刻）。

    一次取齐而不是按用途/按主题各查一遍：金额必须逐行算，
    但"逐行算"不需要"逐行查"——那会让一次汇总变成 N 次数据库往返。
    """
    return [
        UsageRecordRow.model,
        UsageRecordRow.cached_prompt_tokens,
        UsageRecordRow.cache_miss_tokens,
        UsageRecordRow.completion_tokens,
        UsageRecordRow.created_at,
    ]


def _cost_of_row(
    values: Sequence[Any],
    price_table: dict[str, dict[str, float]],
    off_peak_multiplier: float,
) -> CostEstimate:
    """按这一行**实际发生时刻**计价。"""
    return estimate_cost(
        model=str(values[0] or ""),
        cached_prompt_tokens=int(values[1] or 0),
        cache_miss_tokens=int(values[2] or 0),
        output_tokens=int(values[3] or 0),
        at=values[4],
        price_table=price_table,
        off_peak_multiplier=off_peak_multiplier,
    )


def _rounded(estimate: CostEstimate) -> CostEstimate:
    return CostEstimate(
        amount=round(estimate.amount, COST_PRECISION),
        currency=estimate.currency,
        peak_tokens=estimate.peak_tokens,
        off_peak_tokens=estimate.off_peak_tokens,
    )


class UsageRecorder:
    """把每次请求的用量写入 ``usage_record``。"""

    def __init__(
        self,
        db: Database,
        *,
        subject_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self._db = db
        self._subject_id = subject_id
        self._conversation_id = conversation_id

    def record(
        self,
        purpose: LlmPurpose,
        model: str,
        usage: TokenUsage | None,
        *,
        conversation_id: str | None = None,
        subject_id: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """写入一条记录；``usage`` 为空时写入全 0 记录，以保留"这次调用发生过的"事实。"""
        payload = usage or TokenUsage()
        with self._db.session() as session:
            session.add(
                UsageRecordRow(
                    subject_id=subject_id if subject_id is not None else self._subject_id,
                    conversation_id=conversation_id or self._conversation_id,
                    purpose=purpose.value,
                    model=model,
                    prompt_tokens=payload.prompt_tokens,
                    cached_prompt_tokens=payload.cached_prompt_tokens,
                    cache_miss_tokens=payload.cache_miss_tokens,
                    completion_tokens=payload.completion_tokens,
                    reasoning_tokens=payload.reasoning_tokens,
                    total_tokens=payload.total_tokens,
                    duration_ms=duration_ms,
                )
            )

    def summary(self, *, subject_id: str | None = None) -> UsageSummary:
        """聚合统计，用于主题详情页展示与全局用量（T3.6）。"""
        target = subject_id if subject_id is not None else self._subject_id
        statement = select(*_aggregate_columns())
        if target is not None:
            statement = statement.where(UsageRecordRow.subject_id == target)

        with self._db.session() as session:
            row = session.execute(statement).one()

        return _summary_from_values(tuple(row))

    def by_purpose(self, *, subject_id: str | None = None) -> dict[str, UsageSummary]:
        """按用途聚合（FR-705）：回答"哪个用途在烧 token"，这是调档位的直接依据。"""
        target = subject_id if subject_id is not None else self._subject_id
        statement = (
            select(UsageRecordRow.purpose, *_aggregate_columns()).group_by(UsageRecordRow.purpose)
        )
        if target is not None:
            statement = statement.where(UsageRecordRow.subject_id == target)

        with self._db.session() as session:
            rows = session.execute(statement).all()

        return {str(row[0]): _summary_from_values(tuple(row)[1:]) for row in rows}

    def by_subject(self, *, limit: int = 10) -> list[SubjectUsage]:
        """按主题聚合并按总 token 降序取前 ``limit`` 个（FR-705）。

        只看有主题归属的记录：没有主题的调用无法归到任何一个主题上，混进来会让"按主题"
        这个口径失去意义（它们的用量仍计入总量与按用途聚合）。
        """
        statement = (
            select(UsageRecordRow.subject_id, *_aggregate_columns())
            .where(UsageRecordRow.subject_id.is_not(None))
            .group_by(UsageRecordRow.subject_id)
        )
        with self._db.session() as session:
            rows = session.execute(statement).all()
            items = [
                SubjectUsage(
                    subject_id=str(row[0]),
                    title=str(row[0]),
                    summary=_summary_from_values(tuple(row)[1:]),
                )
                for row in rows
            ]
            titles = _subject_titles(session, [item.subject_id for item in items])

        resolved = [
            SubjectUsage(
                subject_id=item.subject_id,
                title=titles.get(item.subject_id) or item.subject_id,
                summary=item.summary,
            )
            for item in items
        ]
        resolved.sort(key=lambda item: item.summary.total_tokens, reverse=True)
        return resolved[:limit]

    def cost_summary(
        self,
        *,
        subject_id: str | None = None,
        price_table: dict[str, dict[str, float]] | None = None,
        off_peak_multiplier: float = OFF_PEAK_MULTIPLIER,
    ) -> CostEstimate:
        """总费用估算（FR-706）。

        **逐行计价**而不是"总量 × 现价"：高峰/空闲按每条记录发生时刻取价，
        一刀切会把夜间跑出来的用量按高峰价算，金额凭空翻倍。
        """
        target = subject_id if subject_id is not None else self._subject_id
        table = effective_price_table(price_table)
        statement = select(*_cost_columns())
        if target is not None:
            statement = statement.where(UsageRecordRow.subject_id == target)

        with self._db.session() as session:
            rows = session.execute(statement).all()

        return _rounded(combine([_cost_of_row(tuple(row), table, off_peak_multiplier) for row in rows]))

    def cost_by_purpose(
        self,
        *,
        subject_id: str | None = None,
        price_table: dict[str, dict[str, float]] | None = None,
        off_peak_multiplier: float = OFF_PEAK_MULTIPLIER,
    ) -> dict[str, CostEstimate]:
        """按用途聚合费用，与 :meth:`by_purpose` 平行（FR-705、FR-706）。"""
        target = subject_id if subject_id is not None else self._subject_id
        table = effective_price_table(price_table)
        statement = select(UsageRecordRow.purpose, *_cost_columns())
        if target is not None:
            statement = statement.where(UsageRecordRow.subject_id == target)

        with self._db.session() as session:
            rows = session.execute(statement).all()

        grouped: dict[str, list[CostEstimate]] = {}
        for row in rows:
            values = tuple(row)
            grouped.setdefault(str(values[0]), []).append(_cost_of_row(values[1:], table, off_peak_multiplier))
        return {purpose: _rounded(combine(items)) for purpose, items in grouped.items()}

    def cost_by_subject(
        self,
        *,
        limit: int = 10,
        price_table: dict[str, dict[str, float]] | None = None,
        off_peak_multiplier: float = OFF_PEAK_MULTIPLIER,
    ) -> list[SubjectCost]:
        """按主题聚合费用并按金额降序取前 ``limit`` 个（FR-706）。

        与 :meth:`by_subject` 同一口径：只统计有主题归属的记录，
        没有归属的调用无法算到任何一个主题上（它们仍计入总费用与按用途聚合）。
        """
        table = effective_price_table(price_table)
        statement = select(UsageRecordRow.subject_id, *_cost_columns()).where(
            UsageRecordRow.subject_id.is_not(None)
        )

        with self._db.session() as session:
            rows = session.execute(statement).all()
            grouped: dict[str, list[CostEstimate]] = {}
            for row in rows:
                values = tuple(row)
                grouped.setdefault(str(values[0]), []).append(_cost_of_row(values[1:], table, off_peak_multiplier))
            titles = _subject_titles(session, list(grouped))

        items = [
            SubjectCost(
                subject_id=subject,
                title=titles.get(subject) or subject,
                cost=_rounded(combine(estimates)),
            )
            for subject, estimates in grouped.items()
        ]
        items.sort(key=lambda item: (-item.cost.amount, item.subject_id))
        return items[:limit]

    @staticmethod
    def from_mapping(purpose: LlmPurpose, model: str, usage: Mapping[str, object]) -> UsageRecordRow:
        """从原始字典构造行（测试与导入场景使用）。"""
        parsed = TokenUsage.from_usage_dict(usage)
        payload = parsed or TokenUsage()
        return UsageRecordRow(
            purpose=purpose.value,
            model=model,
            prompt_tokens=payload.prompt_tokens,
            cached_prompt_tokens=payload.cached_prompt_tokens,
            cache_miss_tokens=payload.cache_miss_tokens,
            completion_tokens=payload.completion_tokens,
            reasoning_tokens=payload.reasoning_tokens,
            total_tokens=payload.total_tokens,
        )
