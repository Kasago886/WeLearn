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
from sqlalchemy.orm import Session, sessionmaker

from app.infra.storage.models import Base, SchemaVersionRow, utcnow

SCHEMA_VERSION = 2
"""当前表结构版本。

v1：设置、密钥、会话记忆、用量（P0/P1）。
v2：学习领域表（主题、会话状态、测评、路径、知识点、检查点、作答、薄弱点、笔记）。

升级策略：只做"新增表"，因此 :meth:`Database.init` 里的 ``create_all`` 即可完成迁移
（SQLite 对新增表无需 ALTER）。将来若要改列，再引入真正的迁移步骤（见实施计划待决策项 E2）。
"""


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
        """建表并记录 schema 版本，返回当前版本号。"""
        Base.metadata.create_all(self._engine)

        with self.session() as session:
            row = session.scalar(select(SchemaVersionRow).where(SchemaVersionRow.version == SCHEMA_VERSION))
            if row is None:
                session.add(SchemaVersionRow(version=SCHEMA_VERSION, applied_at=utcnow()))
        return SCHEMA_VERSION

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
