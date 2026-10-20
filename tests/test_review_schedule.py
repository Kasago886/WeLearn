"""间隔复习排期与久未复习提醒（T4.1，FR-505、FR-506）。

分三层验证：

- **纯函数**：阶梯推进、封顶、失败重置、到期与久未复习判断——排期的全部规则都在这里，
  断言起来不需要数据库；
- **服务层**：掌握节点补齐排期、复习通过后到期日推后、复习失败后回到 1 天，
  以及"懒补齐连调两次不改变结果"这条幂等性；
- **接口层**：``/review`` 暴露 ``due[]`` / ``due_count``；主题列表的 ``needs_review``
  在节点变老之后为真。
"""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.contracts.schemas import NodeState, WeakPointReason
from app.infra.storage.models import utcnow
from app.main import create_app
from app.main_context import AppContext
from app.services.learning.planning import PlanningService
from app.services.learning.review import ReviewService
from app.services.learning.schedule import (
    MAX_BOX,
    REVIEW_INTERVALS_DAYS,
    interval_days,
    is_due,
    schedule_after_fail,
    schedule_after_mastery,
    schedule_after_pass,
)
from tests.learning_harness import LearningHarness, make_harness
from tests.stubs import json_response
from tests.test_teaching_service import enter_teaching, verdict_json

NOW = datetime(2026, 3, 1, 9, 0, 0)


# --------------------------------------------------------------------------- #
# 纯函数：阶梯
# --------------------------------------------------------------------------- #


def test_mastery_starts_at_the_first_box() -> None:
    box, due = schedule_after_mastery(NOW)

    assert box == 1
    assert due == NOW + timedelta(days=1)


def test_pass_moves_up_one_box_and_widens_the_gap() -> None:
    box = 0
    seen: list[tuple[int, int]] = []
    for _ in REVIEW_INTERVALS_DAYS:
        box, due = schedule_after_pass(box, NOW)
        seen.append((box, (due - NOW).days))

    assert seen == [(1, 1), (2, 3), (3, 7), (4, 21), (5, 60)]


def test_pass_caps_at_the_top_box() -> None:
    box, due = schedule_after_pass(MAX_BOX, NOW)

    assert box == MAX_BOX
    assert due == NOW + timedelta(days=REVIEW_INTERVALS_DAYS[-1])


def test_fail_resets_to_the_first_box() -> None:
    box, due = schedule_after_fail(NOW)

    assert box == 1
    assert due == NOW + timedelta(days=1)


def test_interval_days_clamps_out_of_range_boxes() -> None:
    assert interval_days(None) == 1
    assert interval_days(0) == 1
    assert interval_days(MAX_BOX + 10) == REVIEW_INTERVALS_DAYS[-1]


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (None, False),
        (timedelta(seconds=-1), True),
        (timedelta(0), True),
        (timedelta(days=1), False),
    ],
)
def test_is_due_compares_against_now(offset: timedelta | None, expected: bool) -> None:
    due_at = None if offset is None else NOW + offset

    assert is_due(due_at, NOW) is expected


# --------------------------------------------------------------------------- #
# 服务层：排期写入与懒补齐
# --------------------------------------------------------------------------- #


def script(harness: LearningHarness, *, blocking: list[str] | None = None) -> None:
    harness.client.blocking_script = [json_response(item) for item in (blocking or [])]
    harness.client.stream_script = []
    harness.client.blocking_calls = 0
    harness.client.stream_calls = 0


def make_review(harness: LearningHarness, planning: PlanningService) -> ReviewService:
    return ReviewService(harness.repos, harness.state, harness.sessions, planning)


def first_node(harness: LearningHarness, subject_id: str):
    path = harness.repos.paths.get_active(subject_id)
    assert path is not None
    return path.ordered_nodes()[0]


def mastered_long_ago(harness: LearningHarness, subject_id: str, *, days: int = 30):
    """造一个"功能上线前就已掌握"的节点：有掌握时间，但没有排期。"""
    node = first_node(harness, subject_id)
    harness.repos.paths.update_node(
        node.id,
        state=NodeState.MASTERED,
        mastered_at=utcnow() - timedelta(days=days),
    )
    return node


async def test_mastered_node_without_schedule_appears_as_due(context: AppContext) -> None:
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = mastered_long_ago(harness, subject_id)

    overview = reviews.overview(subject_id)

    assert overview.due_count == 1
    item = overview.due[0]
    assert item.node.id == node.id
    assert item.node_index == 1
    assert item.node_total == 2
    assert item.interval_days == 1
    # 补齐要落库：否则每次读取都要重新猜一遍，且列表页拿不到这个数字
    stored = harness.repos.paths.node(node.id)
    assert stored is not None
    assert stored.review_box == 1
    assert stored.review_due_at is not None
    # 以掌握时间为锚：几十天前学会的节点应该"现在就到期"，而不是明天
    assert stored.review_due_at < utcnow()


async def test_backfill_is_idempotent(context: AppContext) -> None:
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = mastered_long_ago(harness, subject_id)

    first = reviews.overview(subject_id)
    snapshot_after_first = harness.repos.paths.node(node.id)
    second = reviews.overview(subject_id)
    snapshot_after_second = harness.repos.paths.node(node.id)

    assert snapshot_after_first is not None and snapshot_after_second is not None
    assert (snapshot_after_first.review_box, snapshot_after_first.review_due_at) == (
        snapshot_after_second.review_box,
        snapshot_after_second.review_due_at,
    )
    assert second.due_count == first.due_count == 1


async def test_pass_review_pushes_the_due_date_back(context: AppContext) -> None:
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = first_node(harness, subject_id)
    harness.repos.paths.update_node(
        node.id,
        state=NodeState.PARTIAL,
        review_box=2,
        review_due_at=utcnow() - timedelta(days=2),
    )
    harness.repos.weak_points.record(subject_id, node.id, WeakPointReason.SHOW_ANSWER)
    reviews.start(subject_id, node.id)

    script(harness, blocking=[verdict_json("pass")])
    turn = await reviews.answer(subject_id, "这次说清楚了")

    assert turn.ok and turn.phase == "passed"
    refreshed = harness.repos.paths.node(node.id)
    assert refreshed is not None
    # 第 2 档通过 → 第 3 档（7 天）
    assert refreshed.review_box == 3
    assert refreshed.review_due_at is not None
    assert refreshed.review_due_at > utcnow() + timedelta(days=6)
    assert refreshed.last_reviewed_at is not None
    assert refreshed.mastered_at is not None
    # 推后之后不再出现在到期清单里
    assert reviews.overview(subject_id).due_count == 0


async def test_failed_review_resets_the_schedule_to_one_day(context: AppContext) -> None:
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = first_node(harness, subject_id)
    harness.repos.paths.update_node(
        node.id,
        state=NodeState.PARTIAL,
        review_box=3,
        review_due_at=utcnow() - timedelta(days=1),
    )
    harness.repos.weak_points.record(subject_id, node.id, WeakPointReason.SHOW_ANSWER)
    reviews.start(subject_id, node.id)

    script(harness, blocking=[verdict_json("fail")])
    turn = await reviews.answer(subject_id, "还是没讲清")

    assert turn.ok and turn.phase == "failed"
    refreshed = harness.repos.paths.node(node.id)
    assert refreshed is not None
    assert refreshed.review_box == 1
    assert refreshed.review_due_at is not None
    assert utcnow() + timedelta(hours=23) < refreshed.review_due_at < utcnow() + timedelta(days=1, hours=1)


# --------------------------------------------------------------------------- #
# 接口层
# --------------------------------------------------------------------------- #


@dataclass
class Api:
    client: TestClient
    app: Any


@pytest.fixture
def api(env: dict[str, str]) -> Iterator[Api]:
    """完整应用（临时数据目录）；这里的接口都不调用模型，无需替换客户端。"""
    app = create_app(env)
    with TestClient(app) as client:
        yield Api(client=client, app=app)


def node_payload(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "objective": f"能解释 {title}",
        "prerequisites": [],
        "key_points": ["要点一"],
        "checkpoints": [
            {
                "question": f"{title} 的检查点？",
                "expected_points": ["判定要点"],
                "reference_answer": "参考答案",
            }
        ],
        "is_quick_review": False,
    }


def seed_mastered(api: Api, *, age_days: int) -> tuple[str, str]:
    """建主题 + 一条含两个节点的路径，并把第一个节点标成若干天前已掌握。"""
    response = api.client.post("/api/subjects", json={"title": "间隔复习"})
    assert response.status_code == 201, response.text
    subject_id = response.json()["subject"]["id"]

    stack = api.app.state.stack
    stack.repos.paths.create(
        subject_id,
        rationale="测试路径",
        model=None,
        nodes=[node_payload("节点 0"), node_payload("节点 1")],
    )
    path = stack.repos.paths.get_active(subject_id)
    assert path is not None
    node = path.ordered_nodes()[0]
    stack.repos.paths.update_node(
        node.id,
        state=NodeState.MASTERED,
        mastered_at=utcnow() - timedelta(days=age_days),
    )
    return subject_id, node.id


def test_review_endpoint_exposes_due_items(api: Api) -> None:
    subject_id, node_id = seed_mastered(api, age_days=30)

    body = api.client.get(f"/api/subjects/{subject_id}/review").json()

    assert body["due_count"] == 1
    assert len(body["due"]) == 1
    due = body["due"][0]
    assert due["node_id"] == node_id
    assert due["node_title"] == "节点 0"
    assert due["node_index"] == 1
    assert due["node_total"] == 2
    assert due["interval_days"] == 1
    assert isinstance(due["due_at"], str) and "T" in due["due_at"]
    # 原有字段一个不少（向后兼容）
    assert body["open_count"] == 0
    assert body["node_total"] == 2
    assert body["mastered_count"] == 1
    assert body["current"] is None


def test_weak_points_endpoint_keeps_fields_and_adds_due_count(api: Api) -> None:
    subject_id, _ = seed_mastered(api, age_days=30)

    body = api.client.get(f"/api/subjects/{subject_id}/weak-points").json()

    assert body["items"] == []
    assert body["node_total"] == 2
    assert body["mastered_count"] == 1
    assert body["due_count"] == 1


def test_subject_list_flags_need_review_after_the_threshold(api: Api) -> None:
    subject_id, node_id = seed_mastered(api, age_days=1)

    def entry() -> dict[str, Any]:
        items = api.client.get("/api/subjects").json()["items"]
        return next(item for item in items if item["id"] == subject_id)

    # 昨天刚掌握、阈值 7 天：还不需要复习
    assert entry()["needs_review"] is False
    assert entry()["due_reviews"] == 0

    # 把掌握时间改老到超过阈值（模拟"久未复习"）
    api.app.state.stack.repos.paths.update_node(
        node_id, mastered_at=utcnow() - timedelta(days=30)
    )

    assert entry()["needs_review"] is True
    assert entry()["due_reviews"] == 1

    # 打开一次复习页会补上排期：节点从"久未复习"变成"已到期"，但不该被算两次
    assert api.client.get(f"/api/subjects/{subject_id}/review").json()["due_count"] == 1
    assert entry()["due_reviews"] == 1
