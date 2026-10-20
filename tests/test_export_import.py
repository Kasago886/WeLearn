"""学习报告导出、记忆与档案导入导出（T4.3，FR-805、FR-613）。"""

import json
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.contracts.schemas import ConversationKind, NodeState, WeakPointReason
from app.main import create_app
from app.main_context import AppContext
from app.services.learning.export import (
    ARCHIVE_FORMAT,
    ARCHIVE_VERSION,
    ExportError,
    build_archive_json,
    import_archive,
    render_markdown,
)
from app.services.learning.repository import DomainRepositories
from app.services.learning.stack import LearningStack, build_learning_stack
from app.services.llm.memory import Memory, assistant_message, summary_message, system_message, user_message
from tests.learning_harness import make_harness
from tests.test_teaching_service import enter_teaching

NOTE_CONTENT = "我的笔记：先验很重要"
GENERAL_NOTE = "主题级备忘：先看定义再看例子"
ANSWER_QUESTION = "什么是先验？"
SUBJECT_TITLE = "贝叶斯定理"


# --------------------------------------------------------------------------- #
# 领域层夹具
# --------------------------------------------------------------------------- #


def make_stack(context: AppContext) -> LearningStack:
    return build_learning_stack(context)


def seed_subject(context: AppContext, *, nodes: int = 2) -> tuple[LearningStack, str]:
    """建一个带路径、笔记、薄弱点与作答记录的主题。"""
    harness = make_harness(context)
    _, _, _, subject_id = enter_teaching(harness, node_count=nodes)
    stack = make_stack(context)

    path = stack.repos.paths.get_active(subject_id)
    assert path is not None
    first = path.ordered_nodes()[0]
    stack.repos.paths.update_node(first.id, state=NodeState.PARTIAL)
    stack.repos.notes.upsert(subject_id, first.id, NOTE_CONTENT, source="user")
    stack.repos.notes.upsert(subject_id, None, GENERAL_NOTE, source="auto")
    stack.repos.weak_points.record(subject_id, first.id, WeakPointReason.SHOW_ANSWER)
    stack.repos.answers.add(
        subject_id=subject_id,
        node_id=first.id,
        checkpoint_id=first.checkpoints[0].id,
        question=ANSWER_QUESTION,
        user_answer="初始信念",
        verdict="fail",
        confidence=0.2,
        feedback="缺少似然说明",
        hit_points=[],
        missing_points=["似然"],
        source="teach",
    )
    return stack, subject_id


def seed_memory(context: AppContext, subject_id: str) -> None:
    """写入一路教学记忆：含 system、正文与摘要，用于验证导出时的取舍。"""
    conversation_id = context.memories.ensure_conversation(ConversationKind.TEACH, subject_id=subject_id)
    memory = Memory()
    memory.add_round().add(system_message("你是耐心的导师"))
    first = memory.add_round()
    first.add(user_message("什么是贝叶斯定理？"))
    first.add(assistant_message("它是用新证据更新信念的方法。", reasoning="先给定义。"))
    summary_round = memory.add_round()
    summary_round.add(summary_message("早前聊过定义"), is_summary=True)
    context.memories.save(conversation_id, memory)


# --------------------------------------------------------------------------- #
# Markdown 报告
# --------------------------------------------------------------------------- #


def test_markdown_contains_nodes_notes_and_weak_points(context: AppContext) -> None:
    stack, subject_id = seed_subject(context)
    path = stack.repos.paths.get_active(subject_id)
    assert path is not None

    markdown = render_markdown(stack, subject_id)

    assert f"# 学习报告：{SUBJECT_TITLE}" in markdown
    assert "## 学习路径" in markdown
    assert "从基础开始逐点推进。" in markdown
    for node in path.ordered_nodes():
        assert node.title in markdown
    assert NOTE_CONTENT in markdown
    assert "用户笔记" in markdown
    assert GENERAL_NOTE in markdown
    assert "## 薄弱点" in markdown
    assert "直接查看答案" in markdown
    assert "## 完成情况" in markdown
    assert "已掌握：" in markdown
    assert "部分掌握：" in markdown
    assert "累计 token：" in markdown


def test_markdown_marks_auto_notes(context: AppContext) -> None:
    stack, subject_id = seed_subject(context)

    markdown = render_markdown(stack, subject_id)

    assert "自动整理" in markdown


# --------------------------------------------------------------------------- #
# JSON 档案
# --------------------------------------------------------------------------- #


def test_archive_ids_match_database(context: AppContext) -> None:
    stack, subject_id = seed_subject(context)
    path = stack.repos.paths.get_active(subject_id)
    assert path is not None
    notes = stack.repos.notes.list_for_subject(subject_id)
    answers = stack.repos.answers.list_for_subject(subject_id)

    archive = build_archive_json(stack, subject_id)

    assert archive["format"] == ARCHIVE_FORMAT
    assert archive["version"] == ARCHIVE_VERSION
    assert archive["subject"]["id"] == subject_id
    assert archive["path"]["id"] == path.id
    assert [node["id"] for node in archive["path"]["nodes"]] == [node.id for node in path.ordered_nodes()]
    assert {note["id"] for note in archive["notes"]} == {note.id for note in notes}
    assert [record["id"] for record in archive["answers"]] == [record.id for record in answers]
    assert [point["node_id"] for point in archive["weak_points"]] == [path.ordered_nodes()[0].id]
    assert archive["answers"][0]["source"] == "teach"


def test_archive_memory_excludes_system_and_summary(context: AppContext) -> None:
    stack, subject_id = seed_subject(context)
    seed_memory(context, subject_id)

    archive = build_archive_json(stack, subject_id)
    teach = archive["memory"]["teach"]

    assert [message["role"] for message in teach] == ["user", "assistant"]
    assert all(message["role"] != "system" for message in teach)
    assert all("早前聊过定义" not in message["content"] for message in teach)
    assert teach[1]["reasoning_content"] == "先给定义。"
    # 直接读表才能拿到时间戳；退回可见正文会丢时间
    assert teach[0]["created_at"] is not None
    assert archive["memory"]["assess"] == []
    assert archive["memory"]["review"] == []


# --------------------------------------------------------------------------- #
# 导入
# --------------------------------------------------------------------------- #


def test_import_creates_new_subject_with_remapped_ids(context: AppContext) -> None:
    stack, subject_id = seed_subject(context)
    seed_memory(context, subject_id)
    original = stack.repos.paths.get_active(subject_id)
    assert original is not None
    original_node_ids = {node.id for node in original.ordered_nodes()}
    original_checkpoint_ids = {
        checkpoint.id for node in original.ordered_nodes() for checkpoint in node.checkpoints
    }
    archive = build_archive_json(stack, subject_id)

    first = import_archive(context.db, archive)
    second = import_archive(context.db, archive)

    assert len({subject_id, first, second}) == 3
    for new_id in (first, second):
        subject = stack.repos.subjects.get(new_id)
        assert subject is not None
        assert subject.title.endswith("（导入）")
        assert subject.goal == stack.repos.subjects.get(subject_id).goal  # type: ignore[union-attr]

        path = stack.repos.paths.get_active(new_id)
        assert path is not None
        assert path.version == 1
        assert path.node_count == original.node_count
        assert {node.id for node in path.ordered_nodes()}.isdisjoint(original_node_ids)
        assert path.rationale == original.rationale

        note_contents = {note.content for note in stack.repos.notes.list_for_subject(new_id)}
        assert note_contents == {NOTE_CONTENT, GENERAL_NOTE}

        answers = stack.repos.answers.list_for_subject(new_id)
        assert len(answers) == 1
        assert answers[0].source == "teach"
        assert answers[0].node_id in {node.id for node in path.ordered_nodes()}
        assert answers[0].checkpoint_id not in original_checkpoint_ids

        weak_points = stack.repos.weak_points.list_open(new_id)
        assert [point.reason for point in weak_points] == [WeakPointReason.SHOW_ANSWER]

        conversation_id = context.memories.conversation_id_for(new_id, ConversationKind.TEACH)
        assert conversation_id is not None
        restored = context.memories.load(conversation_id).outbound_messages()
        assert [message["role"] for message in restored] == ["user", "assistant"]
        assert restored[1]["reasoning_content"] == "先给定义。"


def test_import_without_path_is_allowed(context: AppContext) -> None:
    stack, subject_id = seed_subject(context)
    archive = build_archive_json(stack, subject_id)
    archive["path"] = None
    archive["notes"] = []
    archive["answers"] = []
    archive["weak_points"] = []

    new_id = import_archive(context.db, archive)

    assert stack.repos.subjects.get(new_id) is not None
    assert stack.repos.paths.get_active(new_id) is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"format": "other", "version": 1, "subject": {"title": "x"}},
        {"format": ARCHIVE_FORMAT, "version": 99, "subject": {"title": "x"}},
        {"format": ARCHIVE_FORMAT, "version": 1},
        {"format": ARCHIVE_FORMAT, "version": 1, "subject": {"title": ""}},
        {"format": ARCHIVE_FORMAT, "version": 1, "subject": {"title": "x"}, "path": {"nodes": "oops"}},
    ],
)
def test_import_rejects_invalid_payload(context: AppContext, payload: dict) -> None:
    with pytest.raises(ExportError):
        import_archive(context.db, payload)


# --------------------------------------------------------------------------- #
# HTTP 接口
# --------------------------------------------------------------------------- #


@dataclass
class ApiHarness:
    client: TestClient
    ctx: AppContext
    stack: LearningStack


def ensure_routers(app: FastAPI) -> None:
    """路由由主装配统一 include；单独跑本文件时自己补上，避免依赖并行改动的 app/main.py。"""
    paths = {getattr(route, "path", "") for route in app.routes}
    if "/api/subjects/{subject_id}/export" not in paths:
        from app.api.exports import router as exports_router

        app.include_router(exports_router)


@pytest.fixture
def api(env: dict[str, str]) -> Iterator[ApiHarness]:
    app = create_app(env)
    ensure_routers(app)
    with TestClient(app) as client:
        yield ApiHarness(client=client, ctx=app.state.ctx, stack=app.state.stack)


def create_subject(harness: ApiHarness, title: str = SUBJECT_TITLE) -> str:
    response = harness.client.post("/api/subjects", json={"title": title, "goal": "通过面试"})
    assert response.status_code == 201, response.text
    return response.json()["subject"]["id"]


def seed_path(harness: ApiHarness, subject_id: str) -> None:
    path = harness.stack.repos.paths.create(
        subject_id,
        rationale="先打基础",
        model="deepseek-flash",
        nodes=[
            {
                "title": "先验",
                "objective": "理解先验",
                "prerequisites": [],
                "key_points": ["要点一", "要点二", "要点三"],
                "checkpoints": [
                    {"question": "先验是什么？", "expected_points": ["初始信念"], "reference_answer": "初始信念"}
                ],
            }
        ],
    )
    node = path.ordered_nodes()[0]
    harness.stack.repos.notes.upsert(subject_id, node.id, NOTE_CONTENT, source="user")


def test_export_endpoint_returns_markdown(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    seed_path(api, subject_id)

    response = api.client.get(f"/api/subjects/{subject_id}/export", params={"format": "markdown"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["format"] == "markdown"
    assert body["filename"].endswith(".md")
    assert SUBJECT_TITLE in body["content"]
    assert NOTE_CONTENT in body["content"]


def test_export_endpoint_returns_json_archive(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    seed_path(api, subject_id)

    response = api.client.get(f"/api/subjects/{subject_id}/export", params={"format": "json"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["format"] == "json"
    assert body["filename"].endswith(".json")
    archive = json.loads(body["content"])
    assert archive["format"] == ARCHIVE_FORMAT
    assert archive["subject"]["id"] == subject_id


def test_export_endpoint_rejects_unknown_format_and_subject(api: ApiHarness) -> None:
    subject_id = create_subject(api)

    bad_format = api.client.get(f"/api/subjects/{subject_id}/export", params={"format": "xml"})
    missing = api.client.get("/api/subjects/not-exists/export", params={"format": "markdown"})

    assert bad_format.status_code == 422
    assert missing.status_code == 404


def test_import_endpoint_round_trip_twice(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    seed_path(api, subject_id)
    archive = json.loads(
        api.client.get(f"/api/subjects/{subject_id}/export", params={"format": "json"}).json()["content"]
    )

    first = api.client.post("/api/subjects/import", json=archive)
    second = api.client.post("/api/subjects/import", json=archive)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["subject_id"] != second.json()["subject_id"]
    assert first.json()["title"].endswith("（导入）")

    detail = api.client.get(f"/api/subjects/{first.json()['subject_id']}")
    assert detail.status_code == 200
    assert detail.json()["subject"]["title"].endswith("（导入）")
    assert detail.json()["progress"]["total_nodes"] == 1


def test_import_endpoint_rejects_invalid_payload(api: ApiHarness) -> None:
    response = api.client.post(
        "/api/subjects/import",
        json={"format": ARCHIVE_FORMAT, "version": ARCHIVE_VERSION},
    )

    assert response.status_code == 422


def test_import_uses_repository_identity_for_notes(context: AppContext) -> None:
    """导入后的笔记必须挂在**新**节点上，否则档案页会指向不存在的节点。"""
    stack, subject_id = seed_subject(context)
    archive = build_archive_json(stack, subject_id)

    new_id = import_archive(context.db, archive)
    path = stack.repos.paths.get_active(new_id)
    assert path is not None
    node_ids = {node.id for node in path.ordered_nodes()}

    notes = [note for note in stack.repos.notes.list_for_subject(new_id) if note.node_id is not None]
    assert len(notes) == 1
    assert notes[0].node_id in node_ids
    assert notes[0].node_id not in {node.id for node in stack.repos.paths.get_active(subject_id).nodes}  # type: ignore[union-attr]


def test_repository_helper_matches_archive_subject(context: AppContext) -> None:
    """``DomainRepositories`` 与导出用的 stack 看到的是同一份数据。"""
    stack, subject_id = seed_subject(context)
    repos = DomainRepositories(context.db)

    archive = build_archive_json(stack, subject_id)

    assert repos.subjects.get(subject_id) is not None
    assert archive["subject"]["title"] == repos.subjects.get(subject_id).title  # type: ignore[union-attr]
