"""数据库引擎与会话（T0.3）。

设计要点：
- SQLite + WAL：崩溃或异常退出后已提交数据不丢（NFR-106）；
- 事务边界由 :meth:`Database.session` 统一管理，异常自动回滚；
- 单实例单用户场景下连接数极少，因此使用同步引擎，避免异步驱动带来的额外复杂度。
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, sessionmaker

from app.infra.storage.models import Base, SchemaVersionRow, utcnow

SCHEMA_VERSION = 7
"""当前表结构版本。

- v1：设置、密钥、会话记忆、用量（P0/P1）。
- v2：学习领域表（主题、会话状态、测评、路径、知识点、检查点、作答、薄弱点、笔记）。
- v3：``session_state.awaiting_decision``（"正在等用户三选一"必须跨重启可见）。
- v4（P3）：
  - ``message.kind``：消息用途标记，用于把"自由提问"（FR-411）与检查点问答区分开；
  - ``session_state.last_action``：最近一次教学动作，供"重新生成上一条回复"（FR-413）
    知道该重跑哪一个动作——不知道上一步做了什么就没法重跑。
- v5（P3B）：
  - ``answer_record.source``：这次作答来自教学还是复习。没有它，档案页无法区分
    "首次学会"与"复习通过"，复习记录（FR-507）也就无从统计；
  - ``usage_record.duration_ms``：单次模型调用耗时，完成报告要统计累计耗时（FR-417）；
  - ``session_state.review_mode`` / ``review_node_id`` / ``review_checkpoint_id``：
    复习会话的进行中状态。**必须与教学状态分开存**——复习某个旧节点时如果复用
    ``current_node_id``，用户回到教学页就会发现自己"跳"到了那个旧节点上。
- v6（P3C）：
  - ``app_settings.temperature`` / ``top_p``：采样参数，需要落盘才能跨重启生效（FR-718）；
  - ``app_settings.audit_enabled``：请求审计开关，必须与"环境变量强制开启"分开（FR-715）；
  - ``app_settings.use_recommended_overrides`` / ``purpose_overrides_json``：按用途的
    模型与思考档位覆盖（FR-703）。存 JSON 文本而不是逐用途加列：用途集合会随功能扩展，
    加列意味着每次都要动表结构，而这一项的形状只由设置层解释。
- v7（P4）：
  - ``knowledge_point.review_box`` / ``review_due_at``：间隔复习排期（FR-506）。到期日单独存，
    不用"最近修改 + 间隔"现算——否则改个标题就会把复习计划推迟；
  - ``app_settings.review_interval_days``：久未复习的提醒阈值（FR-505，默认 7 天）；
  - ``app_settings.log_max_mb`` / ``log_retention_days``：日志轮转与保留（FR-907）；
  - ``app_settings.access_token_hash``：可选访问口令（FR-903），**只存哈希**；
  - ``app_settings.tools_enabled``：工具调用开关（FR-714），默认关闭。

升级策略有两条，都是幂等的，因此对"全新库"与"已有库"都安全：

1. ``create_all`` 建缺失的**表**（新增表无需 ALTER）；
2. :data:`_COLUMN_UPGRADES` 逐个补齐缺失的**列**（SQLite 的 ``ADD COLUMN`` 支持默认值）。

之所以要显式做第 2 条：``create_all`` 不会修改已存在的表，若只升版本号，
已有数据库会在访问新列时报 "no such column"。
"""

_COLUMN_UPGRADES: tuple[tuple[str, str, str], ...] = (
    ("session_state", "awaiting_decision", "BOOLEAN NOT NULL DEFAULT 0"),
    ("session_state", "last_action", "VARCHAR(32)"),
    ("message", "kind", "VARCHAR(24)"),
    # 带默认值很关键：老库里已有的作答全部来自教学，补列时必须回填成 'teach'，
    # 否则它们的 source 会是 NULL，档案页据此分类就会漏掉这些记录。
    ("answer_record", "source", "VARCHAR(16) NOT NULL DEFAULT 'teach'"),
    ("usage_record", "duration_ms", "INTEGER"),
    ("session_state", "review_mode", "VARCHAR(16)"),
    ("session_state", "review_node_id", "VARCHAR(36)"),
    ("session_state", "review_checkpoint_id", "VARCHAR(36)"),
    # 记录"为了进入哪个节点已经做过回顾"（FR-414）。没有它会重复提问：
    # 用户刷新后点「继续学习」，就会再被问一遍上一节点。
    ("session_state", "adjacent_review_for", "VARCHAR(36)"),
    # 采样参数留空即可：读取时空值走默认（1.0），不必在 DDL 里编一个默认值。
    ("app_settings", "temperature", "FLOAT"),
    ("app_settings", "top_p", "FLOAT"),
    ("app_settings", "audit_enabled", "BOOLEAN NOT NULL DEFAULT 0"),
    ("app_settings", "use_recommended_overrides", "BOOLEAN NOT NULL DEFAULT 0"),
    ("app_settings", "purpose_overrides_json", "TEXT"),
    # v7（P4）：留空即走默认值，不在 DDL 里编默认值——与 temperature/top_p 同一理由
    ("app_settings", "review_interval_days", "INTEGER"),
    ("app_settings", "log_max_mb", "INTEGER"),
    ("app_settings", "log_retention_days", "INTEGER"),
    ("app_settings", "access_token_hash", "VARCHAR(64)"),
    ("app_settings", "tools_enabled", "BOOLEAN NOT NULL DEFAULT 0"),
    ("knowledge_point", "review_box", "INTEGER NOT NULL DEFAULT 0"),
    ("knowledge_point", "review_due_at", "DATETIME"),
)


class Database:
    """SQLite 数据库封装。"""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self._url = url
        self._engine: Engine = create_engine(
            url,
            echo=echo,
            future=True,
            connect_args={"check_same_thread": False} if url.startswith("sqlite") else {},
        )
        if url.startswith("sqlite"):
            event.listen(self._engine, "connect", self._configure_sqlite)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)

    @classmethod
    def from_path(cls, path: Path | str) -> "Database":
        """按数据库文件路径创建；父目录会自动创建。"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(f"sqlite:///{file_path}")

    @property
    def url(self) -> str:
        return self._url

    @property
    def engine(self) -> Engine:
        return self._engine

    @staticmethod
    def _configure_sqlite(dbapi_connection: sqlite3.Connection, _record: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    def init(self) -> int:
        """建表、补齐缺失的列并记录 schema 版本，返回当前版本号。"""
        Base.metadata.create_all(self._engine)
        self._upgrade_columns()

        with self.session() as session:
            row = session.scalar(select(SchemaVersionRow).where(SchemaVersionRow.version == SCHEMA_VERSION))
            if row is None:
                session.add(SchemaVersionRow(version=SCHEMA_VERSION, applied_at=utcnow()))
        return SCHEMA_VERSION

    def _upgrade_columns(self) -> list[str]:
        """把模型里新增、但已有数据库还没有的列补上；返回实际执行的升级描述。

        幂等：列已存在时什么都不做，因此可以每次启动都跑。
        """
        applied: list[str] = []
        with self._engine.begin() as connection:
            for table, column, ddl in _COLUMN_UPGRADES:
                if not self._table_exists(connection, table):
                    continue
                if column in self._column_names(connection, table):
                    continue
                connection.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                applied.append(f"{table}.{column}")
        return applied

    @staticmethod
    def _table_exists(connection: Connection, table: str) -> bool:
        row = connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).first()
        return row is not None

    @staticmethod
    def _column_names(connection: Connection, table: str) -> set[str]:
        rows = connection.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
        return {str(row[1]) for row in rows}

    @contextmanager
    def session(self) -> Iterator[Session]:
        """事务边界：正常退出提交，异常回滚。"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self._engine.dispose()
