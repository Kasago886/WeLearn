"""数据备份与恢复演练（T5.5、NFR-106、D3）。

覆盖五件事，每件都对应一条"出事时会疼"的路径：

1. **在线快照一致性**：写入第一批 → 快照 → 再写第二批，快照里只能有第一批。
   测试同时断言 WAL 文件非空：若只复制主库文件，第一批数据根本不在副本里，
   这正是必须用 :meth:`sqlite3.Connection.backup` 的原因。
2. **恢复**：用快照覆盖当前库，原库改名留存，数据回到快照时刻。
3. **崩溃恢复（NFR-106）**：独立子进程写入并 ``os._exit(1)``（连接不关闭、
   WAL 不 checkpoint），随后重开同一个库，已提交数据必须仍在——这才接近
   "kill -9" 的语义，而不是同进程里的优雅关闭。
4. **升级迁移（v6 → v7）**：删掉 v7 新增列造出老库，跑 ``Database.init()``，
   断言数据一行不少、列补齐、已掌握节点的 ``review_due_at`` 由读取路径幂等补齐。
5. **备份不含明文 API Key**：直接读快照二进制找明文。

全部离线、确定性、不依赖网络与真实额度。
"""

from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from sqlalchemy import text

from app.contracts.schemas import NodeState
from app.infra.paths import DataPaths
from app.infra.secrets import SecretStore
from app.infra.storage.db import Database
from app.main_context import DATABASE_FILENAME
from app.services.learning.entities import LearningPath
from app.services.learning.repository import DomainRepositories
from app.services.learning.review import ReviewOverview, ReviewService

ROOT = Path(__file__).resolve().parents[1]
BACKUP_SCRIPT = ROOT / "scripts" / "backup_db.py"
NOW = datetime(2026, 10, 21, 9, 0, 0)

V7_SETTINGS_COLUMNS = (
    "review_interval_days",
    "log_max_mb",
    "log_retention_days",
    "access_token_hash",
    "tools_enabled",
)
V7_NODE_COLUMNS = ("review_box", "review_due_at")


def _load_backup_tool() -> ModuleType:
    """按路径加载脚本（``scripts/`` 不是包，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location("welearn_backup_db", BACKUP_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


backup_db: Any = _load_backup_tool()


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #


def open_database(path: Path) -> Database:
    db = Database.from_path(path)
    db.init()
    return db


def seed(db: Database, title: str) -> str:
    """写入一个主题 + 路径 + 节点 + 一次作答，返回主题 id。"""
    repos = DomainRepositories(db)
    subject = repos.subjects.create(title)
    path = repos.paths.create(
        subject.id,
        rationale="备份演练用的路径",
        model=None,
        nodes=[
            {
                "title": f"{title}的节点",
                "objective": "验证备份与恢复",
                "prerequisites": [],
                "key_points": ["要点一"],
                "checkpoints": [
                    {
                        "question": "这一步在验证什么？",
                        "expected_points": ["数据完整性"],
                        "reference_answer": "验证数据完整性",
                    }
                ],
            }
        ],
    )
    node = path.ordered_nodes()[0]
    repos.answers.add(
        subject_id=subject.id,
        node_id=node.id,
        checkpoint_id=None,
        question="这一步在验证什么？",
        user_answer="数据完整性",
        verdict="pass",
        confidence=0.9,
        feedback="答对了",
    )
    return subject.id


def raw_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(path))


def subject_titles(path: Path) -> list[str]:
    connection = raw_connection(path)
    try:
        rows = connection.execute("SELECT title FROM subject").fetchall()
    finally:
        connection.close()
    return sorted(str(row[0]) for row in rows)


def count_rows(path: Path, table: str) -> int:
    connection = raw_connection(path)
    try:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


def column_names(db: Database, table: str) -> set[str]:
    with db.engine.connect() as connection:
        rows = connection.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return {str(row[1]) for row in rows}


# --------------------------------------------------------------------------- #
# 1. 在线备份一致性
# --------------------------------------------------------------------------- #


def test_online_snapshot_is_a_consistent_point_in_time(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    db = open_database(data_dir / DATABASE_FILENAME)
    try:
        seed(db, "第一批主题")

        # 已提交但尚未 checkpoint 的事务只存在于 WAL 里——直接复制主库文件会漏掉它
        wal_path = data_dir / f"{DATABASE_FILENAME}-wal"
        assert wal_path.exists()
        assert wal_path.stat().st_size > 0

        info = backup_db.create_backup(data_dir, now=NOW)

        seed(db, "第二批主题")

        assert info.path.name == "welearn-20261021-090000.db"
        assert info.path.parent == data_dir / "backups"
        assert info.size_bytes == info.path.stat().st_size
        # 快照停在第一批之后：后续写入不会"渗"进已经生成的副本
        assert subject_titles(info.path) == ["第一批主题"]
        assert backup_db.read_integrity(info.path) == "ok"
        assert info.integrity == "ok"
    finally:
        db.dispose()


# --------------------------------------------------------------------------- #
# 2. 恢复
# --------------------------------------------------------------------------- #


def test_restore_rolls_back_to_snapshot_and_keeps_the_original(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    db = open_database(data_dir / DATABASE_FILENAME)
    try:
        seed(db, "第一批主题")
        info = backup_db.create_backup(data_dir, now=NOW)
        seed(db, "第二批主题")
    finally:
        # 必须先释放连接：Windows 上被别的连接打开的文件无法改名
        db.dispose()

    result = backup_db.restore_database(info.path, data_dir, now=NOW + timedelta(hours=1))

    assert result.retained is not None
    assert result.retained.name == "welearn.db.before-restore-20261021-100000"
    assert result.retained.exists()
    assert result.target == data_dir / DATABASE_FILENAME
    assert subject_titles(result.target) == ["第一批主题"]
    # 原库被完整留存，而不是被删除或覆盖
    assert subject_titles(result.retained) == ["第一批主题", "第二批主题"]
    assert count_rows(result.target, "answer_record") == 1
    assert backup_db.read_integrity(result.target) == "ok"

    # 恢复出来的库要能直接被应用打开（升级流程幂等，重跑无副作用）
    restored = Database.from_path(result.target)
    try:
        assert restored.init() == 7
        assert [subject.title for subject in DomainRepositories(restored).subjects.list_all()] == ["第一批主题"]
    finally:
        restored.dispose()

    # 全新目录（没有旧库）上恢复：不需要留存原库，数据同样来自快照时刻
    fresh_dir = tmp_path / "fresh-install"
    second = backup_db.restore_database(info.path, fresh_dir)
    assert second.retained is None
    assert subject_titles(second.target) == ["第一批主题"]
    assert backup_db.read_integrity(second.target) == "ok"


def test_restore_refuses_a_corrupt_backup_without_touching_the_live_database(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    db = open_database(data_dir / DATABASE_FILENAME)
    try:
        seed(db, "现有主题")
    finally:
        db.dispose()

    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not a sqlite database" * 8)

    with pytest.raises(backup_db.BackupError):
        backup_db.restore_database(broken, data_dir)

    # 校验在动原库之前完成：既没有改名留存，也没有覆盖
    assert subject_titles(data_dir / DATABASE_FILENAME) == ["现有主题"]
    assert list(data_dir.glob("*.before-restore-*")) == []


# --------------------------------------------------------------------------- #
# 3. 崩溃恢复（NFR-106）
# --------------------------------------------------------------------------- #

# 用独立进程写库后直接 os._exit(1)：不给 close() 与 WAL checkpoint 任何机会。
# 同进程里把连接关掉再重开只能验证"正常关闭"，验证不了进程被杀的那一瞬间。
CRASH_WRITER = """
import os
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1])
connection.execute("PRAGMA journal_mode = WAL")
connection.execute("PRAGMA busy_timeout = 5000")
connection.execute(
    "INSERT INTO subject (id, title, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
    ("crash-subject", "崩溃前提交的主题", "teaching", "2026-10-21 00:00:00", "2026-10-21 00:00:00"),
)
connection.commit()
os._exit(1)
"""


def test_committed_data_survives_an_abrupt_exit(tmp_path: Path) -> None:
    db_path = tmp_path / DATABASE_FILENAME
    open_database(db_path).dispose()

    script_path = tmp_path / "crash_writer.py"
    script_path.write_text(CRASH_WRITER, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script_path), str(db_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 1, completed.stderr

    reopened = Database.from_path(db_path)
    try:
        # 崩溃后重启：升级流程与正常启动一致，必须先能跑通
        assert reopened.init() == 7
        titles = [subject.title for subject in DomainRepositories(reopened).subjects.list_all()]
    finally:
        reopened.dispose()

    assert titles == ["崩溃前提交的主题"]
    assert backup_db.read_integrity(db_path) == "ok"


# --------------------------------------------------------------------------- #
# 4. 升级迁移演练（v6 → v7）
# --------------------------------------------------------------------------- #


def downgrade_to_v6(db: Database) -> None:
    """在当前 schema 上删掉 v7 新增列，得到一份"v6 时代"的库。"""
    with db.engine.begin() as connection:
        # review_due_at 上有索引，SQLite 不允许直接删除被索引引用的列
        connection.exec_driver_sql("DROP INDEX IF EXISTS ix_knowledge_point_review_due_at")
        for column in V7_NODE_COLUMNS:
            connection.exec_driver_sql(f"ALTER TABLE knowledge_point DROP COLUMN {column}")
        for column in V7_SETTINGS_COLUMNS:
            connection.exec_driver_sql(f"ALTER TABLE app_settings DROP COLUMN {column}")
        connection.exec_driver_sql("DELETE FROM schema_version WHERE version = 7")
        connection.exec_driver_sql(
            "INSERT INTO schema_version (version, applied_at) VALUES (6, '2026-09-01 00:00:00')"
        )


class _StubPlanning:
    """``overview()`` 只读 ``current_path``，不必为一次读取装配整套服务。"""

    def __init__(self, repos: DomainRepositories) -> None:
        self._repos = repos

    def current_path(self, subject_id: str) -> LearningPath | None:
        return self._repos.paths.get_active(subject_id)


def review_overview(db: Database, subject_id: str) -> ReviewOverview:
    repos = DomainRepositories(db)
    service = ReviewService(repos, None, None, _StubPlanning(repos))  # type: ignore[arg-type]
    return service.overview(subject_id, now=NOW)


def test_v6_database_upgrades_to_v7_without_losing_data(tmp_path: Path) -> None:
    db_path = tmp_path / DATABASE_FILENAME
    db = open_database(db_path)
    try:
        subject_id = seed(db, "升级演练主题")
        repos = DomainRepositories(db)
        path = repos.paths.get_active(subject_id)
        assert path is not None
        node_id = path.ordered_nodes()[0].id
        repos.paths.update_node(
            node_id,
            state=NodeState.MASTERED,
            mastered_at=NOW - timedelta(days=5),
            attempts=2,
        )

        downgrade_to_v6(db)
        assert set(V7_NODE_COLUMNS).isdisjoint(column_names(db, "knowledge_point"))
        assert set(V7_SETTINGS_COLUMNS).isdisjoint(column_names(db, "app_settings"))
    finally:
        db.dispose()

    upgraded = Database.from_path(db_path)
    try:
        # 相当于"装了新版本后第一次启动"
        assert upgraded.init() == 7
        assert set(V7_NODE_COLUMNS) <= column_names(upgraded, "knowledge_point")
        assert set(V7_SETTINGS_COLUMNS) <= column_names(upgraded, "app_settings")
        with upgraded.engine.connect() as connection:
            assert connection.execute(text("SELECT MAX(version) FROM schema_version")).scalar_one() == 7

        repos = DomainRepositories(upgraded)
        assert [subject.title for subject in repos.subjects.list_all()] == ["升级演练主题"]
        assert repos.paths.get_active(subject_id) is not None
        node = repos.paths.node(node_id)
        assert node is not None
        assert node.title == "升级演练主题的节点"
        assert node.state is NodeState.MASTERED
        assert node.attempts == 2
        assert repos.answers.count_for_node(node_id) == 1
        # 补列不会替老数据编一个到期日：留空，交给读取路径按 mastered_at 补齐
        assert node.review_due_at is None

        overview = review_overview(upgraded, subject_id)
        assert overview.mastered_count == 1
        assert overview.due_count == 1
        first_due = overview.due[0].due_at
        assert first_due == NOW - timedelta(days=4)  # mastered_at + 1 天

        # 幂等：再读一次不会把到期日往后推
        assert review_overview(upgraded, subject_id).due[0].due_at == first_due
    finally:
        upgraded.dispose()


# --------------------------------------------------------------------------- #
# 5. 备份不含明文 API Key
# --------------------------------------------------------------------------- #


def test_backup_contains_no_plaintext_api_key(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = open_database(data_dir / DATABASE_FILENAME)
    try:
        # env={} 隔离真实环境变量，避免用真实 Key 做测试
        store = SecretStore(db, DataPaths(data_dir), env={})
        fake_key = "sk-fake-0123456789abcdef0123456789abcdef"
        store.set_api_key(fake_key)
        assert store.get_api_key() == fake_key

        info = backup_db.create_backup(data_dir, now=NOW)

        raw = info.path.read_bytes()
        assert fake_key.encode("ascii") not in raw
        assert b"sk-fake" not in raw
        # 密文确实落库了，否则"找不到明文"只是因为什么都没存
        assert count_rows(info.path, "api_key_secret") == 1
    finally:
        db.dispose()


# --------------------------------------------------------------------------- #
# CLI 退出码
# --------------------------------------------------------------------------- #


def test_verify_exits_non_zero_on_a_corrupt_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not a sqlite database" * 8)

    assert backup_db.main(["verify", "--data-dir", str(tmp_path), "--file", str(broken)]) == 1
    assert "异常" in capsys.readouterr().out

    db = open_database(tmp_path / "data" / DATABASE_FILENAME)
    try:
        info = backup_db.create_backup(tmp_path / "data", now=NOW)
    finally:
        db.dispose()

    assert backup_db.main(["verify", "--data-dir", str(tmp_path / "data"), "--file", str(info.path)]) == 0
