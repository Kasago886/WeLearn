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

# v5 之前的样子：作答记录没有 source，用量记录没有 duration_ms
OLD_ANSWER_RECORD_DDL = """
CREATE TABLE answer_record (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    subject_id VARCHAR(36) NOT NULL,
    node_id VARCHAR(36) NOT NULL,
    checkpoint_id VARCHAR(36),
    question TEXT NOT NULL DEFAULT '',
    user_answer TEXT NOT NULL DEFAULT '',
    verdict VARCHAR(16) NOT NULL,
    confidence FLOAT,
    feedback TEXT NOT NULL DEFAULT '',
    hit_points_json TEXT,
    missing_points_json TEXT,
    attempt_index INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL
)
"""

OLD_USAGE_RECORD_DDL = """
CREATE TABLE usage_record (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    subject_id VARCHAR(36),
    conversation_id VARCHAR(36),
    purpose VARCHAR(16) NOT NULL,
    model VARCHAR(120) NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    cached_prompt_tokens INTEGER NOT NULL DEFAULT 0,
    cache_miss_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL
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


def test_v5_columns_are_added_and_keep_existing_rows(tmp_path: Path) -> None:
    """v5 的两个新列必须能被补到老库上，且老数据不丢。

    ``answer_record.source`` 与 ``usage_record.duration_ms`` 是 P3B 才加的：
    前者让档案页区分"首次学会"与"复习通过"，后者供完成报告统计累计耗时。
    只升版本号不补列的话，已有库一访问就会 "no such column"。
    """
    db = Database.from_path(tmp_path / "welearn.db")
    try:
        with db.engine.begin() as connection:
            connection.exec_driver_sql(OLD_ANSWER_RECORD_DDL)
            connection.exec_driver_sql(OLD_USAGE_RECORD_DDL)
            connection.exec_driver_sql(
                "INSERT INTO answer_record "
                "(subject_id, node_id, verdict, user_answer, created_at) "
                "VALUES ('s1', 'n1', 'pass', '老作答', '2026-10-10 00:00:00')"
            )
            connection.exec_driver_sql(
                "INSERT INTO usage_record (purpose, model, total_tokens, created_at) "
                "VALUES ('teach', 'deepseek-flash', 100, '2026-10-10 00:00:00')"
            )
        assert "source" not in column_names(db, "answer_record")
        assert "duration_ms" not in column_names(db, "usage_record")

        db.init()

        assert "source" in column_names(db, "answer_record")
        assert "duration_ms" in column_names(db, "usage_record")

        with db.engine.connect() as connection:
            answer = connection.execute(
                text("SELECT verdict, user_answer, source FROM answer_record")
            ).one()
            usage = connection.execute(text("SELECT total_tokens, duration_ms FROM usage_record")).one()

        assert answer[0] == "pass"
        assert answer[1] == "老作答"  # 老数据原样保留
        assert answer[2] == "teach"  # 新列取默认值：老记录都来自教学
        assert usage[0] == 100
        assert usage[1] is None  # 老记录没有耗时可言，留空而不是编一个 0
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
