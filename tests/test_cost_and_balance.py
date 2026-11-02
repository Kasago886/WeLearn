"""成本估算与余额查询（T4.2、FR-706、FR-710）。

两条被测的"不能错"的规则：

1. 费用按**每条记录实际发生的时段**计价——高峰/空闲差一倍，按总量一刀切会让夜间
   跑出来的用量凭空翻倍；
2. 余额查询失败**只能降级**，不能把设置页一起弄坏。

高峰/空闲全部显式传 ``at=``：依赖真实当前时间的测试会在周末或夜里突然变红。
"""

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import billing
from app.contracts.schemas import LlmPurpose
from app.infra.llm.errors import LlmError
from app.infra.storage.models import UsageRecordRow
from app.main import create_app
from app.main_context import AppContext
from app.services.llm.cost import BEIJING, DEFAULT_PRICE_TABLE, effective_price_table, estimate_cost
from app.services.llm.usage import UsageRecorder

MILLION = 1_000_000

# 2026-10-19 是周一，2026-10-17 是周六（周末一律空闲）
MONDAY_MORNING = datetime(2026, 10, 19, 10, 0, tzinfo=BEIJING)
MONDAY_LUNCH_BREAK = datetime(2026, 10, 19, 13, 0, tzinfo=BEIJING)
SATURDAY_MORNING = datetime(2026, 10, 17, 10, 0, tzinfo=BEIJING)

FLASH = DEFAULT_PRICE_TABLE["deepseek-flash"]
PRO = DEFAULT_PRICE_TABLE["deepseek-v4-pro"]

BALANCE_FIXTURE: dict[str, Any] = {
    "is_available": True,
    "balance_infos": [
        {
            "currency": "CNY",
            "total_balance": "17.39",
            "granted_balance": "0.00",
            "topped_up_balance": "17.39",
        }
    ],
}


def utc_naive(moment: datetime) -> datetime:
    """转到库里存的形式：naive UTC（SQLite 不保留时区）。"""
    return moment.astimezone(UTC).replace(tzinfo=None)


def add_usage(
    ctx: AppContext,
    *,
    at: datetime,
    purpose: LlmPurpose = LlmPurpose.TEACH,
    model: str = "deepseek-flash",
    cached: int = 0,
    miss: int = 0,
    output: int = 0,
    subject_id: str | None = None,
) -> None:
    """直接写一行 usage_record：``created_at`` 必须可控，否则测不了高峰/空闲。"""
    with ctx.db.session() as session:
        session.add(
            UsageRecordRow(
                subject_id=subject_id,
                purpose=purpose.value,
                model=model,
                prompt_tokens=cached + miss,
                cached_prompt_tokens=cached,
                cache_miss_tokens=miss,
                completion_tokens=output,
                total_tokens=cached + miss + output,
                created_at=utc_naive(at),
            )
        )


# --------------------------------------------------------------------------- #
# 计价（FR-706）
# --------------------------------------------------------------------------- #


def test_peak_hours_charge_full_price() -> None:
    estimate = estimate_cost(
        model="deepseek-flash",
        cached_prompt_tokens=MILLION,
        cache_miss_tokens=MILLION,
        output_tokens=MILLION,
        at=MONDAY_MORNING,
    )

    assert estimate.amount == pytest.approx(FLASH["cache_hit"] + FLASH["cache_miss"] + FLASH["output"])
    assert estimate.peak_tokens == 3 * MILLION
    assert estimate.off_peak_tokens == 0


def test_weekend_is_off_peak_and_costs_half() -> None:
    peak = estimate_cost(model="deepseek-flash", cache_miss_tokens=MILLION, at=MONDAY_MORNING)
    weekend = estimate_cost(model="deepseek-flash", cache_miss_tokens=MILLION, at=SATURDAY_MORNING)

    assert weekend.amount == pytest.approx(peak.amount / 2)
    assert weekend.off_peak_tokens == MILLION
    assert weekend.peak_tokens == 0


def test_weekday_lunch_break_is_off_peak() -> None:
    """12:00–14:00 落在两个高峰窗口之间，必须按空闲价算。"""
    estimate = estimate_cost(model="deepseek-flash", cache_miss_tokens=MILLION, at=MONDAY_LUNCH_BREAK)

    assert estimate.amount == pytest.approx(FLASH["cache_miss"] / 2)
    assert estimate.off_peak_tokens == MILLION


def test_cache_hit_miss_and_output_are_priced_separately() -> None:
    estimate = estimate_cost(
        model="deepseek-flash",
        cached_prompt_tokens=500_000,
        cache_miss_tokens=250_000,
        output_tokens=2 * MILLION,
        at=MONDAY_MORNING,
    )

    expected = 0.5 * FLASH["cache_hit"] + 0.25 * FLASH["cache_miss"] + 2 * FLASH["output"]
    assert estimate.amount == pytest.approx(expected)


def test_unknown_model_falls_back_to_default_prices_not_zero() -> None:
    """显示 0 元会被误读成"免费"，比估高更糟。"""
    estimate = estimate_cost(model="someone-new-model", cache_miss_tokens=MILLION, at=MONDAY_MORNING)

    assert estimate.amount == pytest.approx(FLASH["cache_miss"])
    assert estimate.amount > 0


def test_price_override_deep_merges_with_builtin() -> None:
    """只覆盖输出价时，命中/未命中价必须保持内置值，而不是被抹成 0。"""
    table = effective_price_table({"deepseek-flash": {"output": 10.0}})

    assert table["deepseek-flash"] == {
        "cache_hit": FLASH["cache_hit"],
        "cache_miss": FLASH["cache_miss"],
        "output": 10.0,
    }
    assert table["deepseek-v4-pro"] == PRO

    estimate = estimate_cost(
        model="deepseek-flash",
        output_tokens=MILLION,
        at=MONDAY_MORNING,
        price_table={"deepseek-flash": {"output": 10.0}},
    )
    assert estimate.amount == pytest.approx(10.0)


def test_override_can_introduce_a_new_model() -> None:
    table = effective_price_table({"local-model": {"cache_miss": 1.0}})

    assert table["local-model"] == {"cache_hit": 0.0, "cache_miss": 1.0, "output": 0.0}


# --------------------------------------------------------------------------- #
# 聚合（FR-706）
# --------------------------------------------------------------------------- #


def test_cost_summary_prices_each_record_at_its_own_moment(context: AppContext) -> None:
    """一峰一谷：总额必须等于两条分别计价之和，而不是按现价乘总量。"""
    add_usage(context, at=MONDAY_MORNING, miss=MILLION)
    add_usage(context, at=SATURDAY_MORNING, miss=MILLION)

    summary = UsageRecorder(context.db).cost_summary()

    expected = estimate_cost(model="deepseek-flash", cache_miss_tokens=MILLION, at=MONDAY_MORNING).amount
    expected += estimate_cost(model="deepseek-flash", cache_miss_tokens=MILLION, at=SATURDAY_MORNING).amount
    assert summary.amount == pytest.approx(expected)
    assert summary.peak_tokens == MILLION
    assert summary.off_peak_tokens == MILLION
    assert summary.total_tokens == 2 * MILLION


def test_cost_summary_without_records_is_zero(context: AppContext) -> None:
    recorder = UsageRecorder(context.db)

    summary = recorder.cost_summary()

    assert summary.amount == 0.0
    assert summary.total_tokens == 0
    assert recorder.cost_by_purpose() == {}
    assert recorder.cost_by_subject() == []


def test_cost_summary_and_costs_group_by_purpose_and_subject(context: AppContext) -> None:
    add_usage(context, at=MONDAY_MORNING, purpose=LlmPurpose.TEACH, miss=MILLION, subject_id="s1")
    add_usage(
        context,
        at=SATURDAY_MORNING,
        purpose=LlmPurpose.JUDGE,
        model="deepseek-v4-pro",
        miss=2 * MILLION,
        subject_id="s2",
    )
    add_usage(context, at=MONDAY_MORNING, purpose=LlmPurpose.JUDGE, miss=MILLION)  # 无主题归属

    recorder = UsageRecorder(context.db)
    by_purpose = recorder.cost_by_purpose()
    by_subject = recorder.cost_by_subject()

    teach_cost = FLASH["cache_miss"]
    s2_cost = PRO["cache_miss"]  # 2M token × 周末半价
    orphan_cost = FLASH["cache_miss"]  # 高峰 × flash
    assert set(by_purpose) == {"teach", "judge"}
    assert by_purpose["teach"].amount == pytest.approx(teach_cost)
    assert by_purpose["judge"].amount == pytest.approx(s2_cost + orphan_cost)

    # 未归属主题的调用不进"按主题"，但仍进总量与按用途；按金额降序
    assert [item.subject_id for item in by_subject] == ["s2", "s1"]
    assert by_subject[0].cost.amount == pytest.approx(s2_cost)
    assert by_subject[0].title == "s2"  # 主题不存在时退回 id，不留空
    assert by_subject[1].cost.amount == pytest.approx(teach_cost)
    assert recorder.cost_summary().amount == pytest.approx(teach_cost + s2_cost + orphan_cost)


def test_cost_summary_filters_by_subject(context: AppContext) -> None:
    add_usage(context, at=MONDAY_MORNING, miss=MILLION, subject_id="s1")
    add_usage(context, at=MONDAY_MORNING, miss=MILLION, subject_id="s2")

    scoped = UsageRecorder(context.db).cost_summary(subject_id="s1")

    assert scoped.amount == pytest.approx(FLASH["cache_miss"])
    assert scoped.total_tokens == MILLION


def test_cost_summary_honours_price_table_override(context: AppContext) -> None:
    add_usage(context, at=MONDAY_MORNING, output=MILLION)

    summary = UsageRecorder(context.db).cost_summary(price_table={"deepseek-flash": {"output": 10.0}})

    assert summary.amount == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# 余额接口（FR-710）
# --------------------------------------------------------------------------- #


class BalanceStub:
    """只实现余额查询的客户端桩；``**kwargs`` 宽容便于接口层将来加参数。"""

    def __init__(self, *, payload: Mapping[str, Any] | None = None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self._delegate: Any = None

    async def get_balance(self, **_: Any) -> Mapping[str, Any]:
        if self.error is not None:
            raise self.error
        assert self.payload is not None
        return self.payload

    async def aclose(self) -> None:
        # 桩替换掉了真实客户端，关闭连接池的责任仍要交给真实客户端，否则连接泄漏
        if self._delegate is not None:
            await self._delegate.aclose()


@pytest.fixture
def balance_client(env: dict[str, str]) -> Iterator[tuple[TestClient, BalanceStub]]:
    app = create_app(env)
    # 路由由 app.main 统一 include；此处兜底，避免"还没接上就测不了"
    if not any(getattr(route, "path", None) == "/api/balance" for route in app.routes):
        app.include_router(billing.router)

    stub = BalanceStub()
    stub._delegate = app.state.ctx.llm
    app.state.ctx.llm = stub  # type: ignore[assignment]

    with TestClient(app) as client:
        yield client, stub


def test_balance_returns_official_string_amounts(balance_client: tuple[TestClient, BalanceStub]) -> None:
    client, stub = balance_client
    stub.payload = BALANCE_FIXTURE

    response = client.get("/api/balance")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["currency"] == "CNY"
    # 金额是字符串原样透传，不能变成 17.39 的浮点
    assert body["total_balance"] == "17.39"
    assert body["granted_balance"] == "0.00"
    assert body["topped_up_balance"] == "17.39"
    assert body["message"] == ""


def test_balance_falls_back_to_cny_when_currency_missing(balance_client: tuple[TestClient, BalanceStub]) -> None:
    client, stub = balance_client
    stub.payload = {"is_available": True, "balance_infos": [{"total_balance": "1.00"}]}

    body = client.get("/api/balance").json()

    assert body["ok"] is True
    assert body["currency"] == "CNY"


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (LlmError.config("尚未配置 API Key，请在设置页填写。"), "config"),
        (LlmError.network("ConnectError"), "network"),
        (LlmError.timeout("ReadTimeout"), "timeout"),
    ],
)
def test_balance_failure_is_still_200(
    balance_client: tuple[TestClient, BalanceStub], error: LlmError, category: str
) -> None:
    """FR-710：余额查不到不能影响主流程，因此不入 4xx/5xx。"""
    client, stub = balance_client
    stub.error = error

    response = client.get("/api/balance")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["currency"] is None
    assert body["category"] == category
    assert isinstance(body["message"], str)
    assert body["message"]


def test_balance_unknown_shape_is_reported_as_protocol_error(balance_client: tuple[TestClient, BalanceStub]) -> None:
    client, stub = balance_client
    stub.payload = {"is_available": True, "balance_infos": []}

    body = client.get("/api/balance").json()

    assert body["ok"] is False
    assert body["category"] == "protocol"
    assert body["message"]


def test_balance_unexpected_exception_does_not_break_the_response(
    balance_client: tuple[TestClient, BalanceStub],
) -> None:
    client, stub = balance_client
    stub.error = AttributeError("client has no get_balance")

    response = client.get("/api/balance")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["category"] == "unknown"
    assert body["message"]
