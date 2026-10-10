"""表结构增量升级测试（T2.2）。

背景：``create_all`` **不会**修改已存在的表。若只升 ``SCHEMA_VERSION`` 而不补列，
已有数据库会在访问新列时报 "no such column"——这正是 ``session_state.awaiting_decision``
落地时必须处理的问题。这里用"先手工建一张缺列的旧表"来复现该场景。
"""

from pathlib import Path

from sqlalchemy import text

from app.infra.storage.db import SCHEMA_VERSION, Database

OLD_SESSION_STATE_DDL = """
CREATE TABLE session_state (
    subject_id VARCHAR(36) NOT NULL PRIMARY KEY,
    state VARCHAR(24) NOT NULL,
    current_node_id VARCHAR(36),
    pending_checkpoint_id VARCHAR(36),
    pending_question TEXT,
    node_attempts INTEGER NOT NULL DEFAULT 0,
    taught_node_id VARCHAR(36),
    updated_at DATETIME NOT NULL
)
"""


def make_old_database(path: Path) -> Database:
    """构造一个"v2 时代"的库：session_state 存在但缺 awaiting_decision。"""
    db = Database.from_path(path)
    with db.engine.begin() as connection:
        connection.exec_driver_sql(OLD_SESSION_STATE_DDL)
    return db


def column_names(db: Database, table: str) -> set[str]:
    with db.engine.connect() as connection:
        rows = connection.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return {str(row[1]) for row in rows}


def test_missing_column_is_added_on_init(tmp_path: Path) -> None:
    db = make_old_database(tmp_path / "welearn.db")
    try:
        assert "awaiting_decision" not in column_names(db, "session_state")

        db.init()

        assert "awaiting_decision" in column_names(db, "session_state")
    finally:
        db.dispose()


def test_existing_rows_get_the_default_value(tmp_path: Path) -> None:
    """升级不能破坏已有数据：缺列补上后，老行应取默认值 False。"""
    db = make_old_database(tmp_path / "welearn.db")
    try:
        with db.engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO session_state (subject_id, state, node_attempts, updated_at) "
                "VALUES ('s1', 'teaching', 3, '2026-10-10 00:00:00')"
            )

        db.init()

        with db.engine.connect() as connection:
            row = connection.execute(
                text("SELECT subject_id, node_attempts, awaiting_decision FROM session_state")
            ).one()

        assert row[0] == "s1"
        assert row[1] == 3  # 原有数据未被改动
        assert row[2] == 0  # 新列取默认值
    finally:
        db.dispose()


def test_init_is_idempotent(tmp_path: Path) -> None:
    """每次启动都会跑升级流程，重复执行必须无害。"""
    db = Database.from_path(tmp_path / "fresh.db")
    try:
        assert db.init() == SCHEMA_VERSION
        assert db.init() == SCHEMA_VERSION

        assert "awaiting_decision" in column_names(db, "session_state")
    finally:
        db.dispose()


def test_fresh_database_has_all_domain_tables(tmp_path: Path) -> None:
    db = Database.from_path(tmp_path / "fresh.db")
    try:
        db.init()
        with db.engine.connect() as connection:
            rows = connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
            ).fetchall()
        tables = {str(row[0]) for row in rows}
    finally:
        db.dispose()

    expected = {
        "answer_record",
        "api_key_secret",
        "app_settings",
        "assessment",
        "assessment_level",
        "chat_round",
        "checkpoint",
        "conversation",
        "knowledge_point",
        "learning_path",
        "message",
        "schema_version",
        "session_state",
        "study_note",
        "subject",
        "usage_record",
        "weak_point",
    }
    assert expected <= tables


def test_upgrade_then_repository_round_trip(context: object) -> None:
    """升级后的列必须能被仓储正常读写（否则升级只是"看起来成功"）。"""
    from app.contracts.schemas import SubjectState
    from app.main_context import AppContext
    from app.services.learning.repository import DomainRepositories

    assert isinstance(context, AppContext)
    repos = DomainRepositories(context.db)
    subject = repos.subjects.create("走一遍升级后的列")
    repos.sessions.create(subject.id, state=SubjectState.TEACHING)

    assert repos.sessions.get(subject.id).needs_user_decision is False  # type: ignore[union-attr]

    repos.sessions.update(subject.id, awaiting_decision=True)

    snapshot = repos.sessions.get(subject.id)
    assert snapshot is not None
    assert snapshot.needs_user_decision is True
    assert snapshot.state is SubjectState.TEACHING
