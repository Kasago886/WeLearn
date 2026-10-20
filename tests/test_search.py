"""全局检索（T4.4，FR-803）。"""

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.contracts.schemas import ConversationKind
from app.main import create_app
from app.main_context import AppContext
from app.services.learning.repository import DomainRepositories
from app.services.learning.search import MAX_LIMIT, make_snippet, search
from app.services.llm.memory import (
    Memory,
    assistant_message,
    summary_message,
    system_message,
    user_message,
)

KEYWORD = "Bayes"
NOTE_TEXT = "Bayes 定理的核心是先验与似然。"
ANSWER_QUESTION = "Bayes 公式怎么写？"
MESSAGE_TEXT = "请解释 Bayes 定理"


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #


def make_path(repos: DomainRepositories, subject_id: str, titles: list[str]) -> list[str]:
    path = repos.paths.create(
        subject_id,
        rationale="从基础推进",
        model=None,
        nodes=[
            {
                "title": title,
                "objective": f"理解 {title}",
                "prerequisites": [],
                "key_points": ["要点一", "要点二", "要点三"],
                "checkpoints": [],
            }
            for title in titles
        ],
    )
    return [node.id for node in path.ordered_nodes()]


def seed_two_subjects(context: AppContext) -> tuple[DomainRepositories, str, str]:
    """主题 A 含三类命中，主题 B 只有一条笔记命中（用于验证 subject_id 过滤）。"""
    repos = DomainRepositories(context.db)
    subject_a = repos.subjects.create("贝叶斯定理")
    subject_b = repos.subjects.create("极大似然估计")

    nodes_a = make_path(repos, subject_a.id, ["先验", "似然", "后验"])
    nodes_b = make_path(repos, subject_b.id, ["似然函数"])

    repos.notes.upsert(subject_a.id, nodes_a[0], NOTE_TEXT, source="user")
    repos.notes.upsert(subject_a.id, nodes_a[1], "似然函数描述数据出现的概率。", source="auto")
    repos.notes.upsert(subject_a.id, nodes_a[2], "后验 ∝ 先验 × 似然。", source="user")
    repos.notes.upsert(subject_b.id, nodes_b[0], "Bayes 与频率学派对立。", source="user")

    repos.answers.add(
        subject_id=subject_a.id,
        node_id=nodes_a[0],
        checkpoint_id=None,
        question=ANSWER_QUESTION,
        user_answer="后验正比于先验乘似然",
        verdict="pass",
        confidence=0.9,
        feedback="回答完整",
        hit_points=["先验"],
        missing_points=[],
        source="teach",
    )

    conversation_id = context.memories.ensure_conversation(ConversationKind.TEACH, subject_id=subject_a.id)
    memory = Memory()
    memory.add_round().add(system_message("你是 Bayes 专家，负责讲解。"))
    first = memory.add_round()
    first.add(user_message(MESSAGE_TEXT))
    first.add(assistant_message("它用新证据更新信念。"))
    summary_round = memory.add_round()
    summary_round.add(summary_message("早前聊过 Bayes 定理"), is_summary=True)
    context.memories.save(conversation_id, memory)
    return repos, subject_a.id, subject_b.id


# --------------------------------------------------------------------------- #
# 服务层
# --------------------------------------------------------------------------- #


def test_search_hits_notes_answers_and_messages(context: AppContext) -> None:
    _, subject_a, subject_b = seed_two_subjects(context)

    results = search(context.db, query=KEYWORD)

    assert len(results.notes) == 2
    assert len(results.answers) == 1
    # system 与摘要消息不应出现在检索结果里
    assert len(results.messages) == 1
    assert results.total == 4
    assert {hit.subject_id for hit in results.notes} == {subject_a, subject_b}
    assert results.answers[0].subject_id == subject_a
    assert results.messages[0].subject_id == subject_a
    assert results.messages[0].role == "user"
    assert results.messages[0].conversation_kind == "teach"


def test_search_is_case_insensitive(context: AppContext) -> None:
    seed_two_subjects(context)

    lower = search(context.db, query="bayes")
    upper = search(context.db, query="BAYES")

    assert lower.total == upper.total == 4
    assert "Bayes" in lower.notes[0].snippet


def test_search_snippet_contains_keyword(context: AppContext) -> None:
    seed_two_subjects(context)

    results = search(context.db, query=KEYWORD)

    assert all(KEYWORD.lower() in hit.snippet.lower() for hit in results.notes)
    assert KEYWORD.lower() in results.answers[0].snippet.lower()
    assert KEYWORD.lower() in results.messages[0].snippet.lower()


def test_search_filters_by_subject(context: AppContext) -> None:
    _, subject_a, subject_b = seed_two_subjects(context)

    only_a = search(context.db, query=KEYWORD, subject_id=subject_a)
    only_b = search(context.db, query=KEYWORD, subject_id=subject_b)

    assert len(only_a.notes) == 1
    assert len(only_a.answers) == 1
    assert len(only_a.messages) == 1
    assert [note.subject_id for note in only_b.notes] == [subject_b]
    assert only_b.answers == []
    assert only_b.messages == []


def test_search_empty_query_returns_empty_results(context: AppContext) -> None:
    seed_two_subjects(context)

    for query in ("", "   ", "\t\n"):
        results = search(context.db, query=query)
        assert results.total == 0
        assert results.notes == []
        assert results.answers == []
        assert results.messages == []


def test_search_respects_limit_per_category(context: AppContext) -> None:
    seed_two_subjects(context)

    results = search(context.db, query="似然", limit=2)

    # limit 是"每类最多几条"：笔记截到 2 条，作答仍有 1 条
    assert len(results.notes) == 2
    assert len(results.answers) == 1
    assert results.total == 3


def test_search_clamps_limit_to_maximum(context: AppContext) -> None:
    seed_two_subjects(context)

    results = search(context.db, query="似然", limit=MAX_LIMIT * 10)

    assert len(results.notes) == 3


def test_search_treats_wildcards_literally(context: AppContext) -> None:
    seed_two_subjects(context)

    results = search(context.db, query="%")

    assert results.total == 0


def test_make_snippet_marks_truncation() -> None:
    text = "前" * 100 + KEYWORD + "后" * 100

    snippet = make_snippet(text, KEYWORD)

    assert snippet.startswith("…")
    assert snippet.endswith("…")
    assert KEYWORD in snippet


# --------------------------------------------------------------------------- #
# HTTP 接口
# --------------------------------------------------------------------------- #


@dataclass
class ApiHarness:
    client: TestClient
    ctx: AppContext


def ensure_routers(app: FastAPI) -> None:
    """路由由主装配统一 include；单独跑本文件时自己补上。"""
    paths = {getattr(route, "path", "") for route in app.routes}
    if "/api/search" not in paths:
        from app.api.search import router as search_router

        app.include_router(search_router)


@pytest.fixture
def api(env: dict[str, str]) -> Iterator[ApiHarness]:
    app = create_app(env)
    ensure_routers(app)
    with TestClient(app) as client:
        yield ApiHarness(client=client, ctx=app.state.ctx)


def test_search_endpoint_shape(api: ApiHarness) -> None:
    _, subject_a, subject_b = seed_two_subjects(api.ctx)

    response = api.client.get("/api/search", params={"q": "bayes"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["query"] == "bayes"
    assert body["total"] == 4
    assert set(body) == {"query", "total", "notes", "answers", "messages"}
    assert set(body["notes"][0]) == {
        "subject_id",
        "subject_title",
        "node_id",
        "node_title",
        "snippet",
        "updated_at",
    }
    assert set(body["answers"][0]) == {
        "subject_id",
        "subject_title",
        "node_id",
        "node_title",
        "verdict",
        "snippet",
        "created_at",
        "source",
    }
    assert set(body["messages"][0]) == {
        "subject_id",
        "subject_title",
        "conversation_kind",
        "role",
        "snippet",
        "created_at",
    }
    assert body["notes"][0]["node_title"] is not None
    assert {note["subject_id"] for note in body["notes"]} == {subject_a, subject_b}
    assert body["answers"][0]["subject_id"] == subject_a
    assert body["messages"][0]["conversation_kind"] == "teach"


def test_search_endpoint_defaults_and_limits(api: ApiHarness) -> None:
    _, subject_a, _ = seed_two_subjects(api.ctx)

    empty = api.client.get("/api/search")
    limited = api.client.get("/api/search", params={"q": "似然", "limit": 1})
    filtered = api.client.get("/api/search", params={"q": "bayes", "subject_id": subject_a})

    assert empty.status_code == 200
    assert empty.json()["total"] == 0
    assert len(limited.json()["notes"]) == 1
    assert limited.json()["total"] == 2
    assert filtered.json()["total"] == 3
