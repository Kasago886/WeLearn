"""SQLAlchemy 表模型（T0.6 数据库契约的代码部分）。

本文件只包含 **P0/P1 需要落库的实体**；学习领域实体（subject / assessment /
learning_path / knowledge_point / checkpoint / answer_record / weak_point / study_note）
在 P2 落地时按需求规格说明书第 6 章补入本文件。

时间戳约定：统一存 **UTC**，展示时再转本地时区（SRS 第 6 章）。
"""

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """当前 UTC 时间（naive，SQLite 不保留时区信息）。"""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """声明基类。"""


class SchemaVersionRow(Base):
    """已应用的 schema 版本（轻量迁移方案，见待决策项 E2）。"""

    __tablename__ = "schema_version"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AppSettingsRow(Base):
    """应用设置（单行，id 恒为 1）。

    所有列可空：读取时缺失/空值/非法值一律回退默认（FR-902）。
    """

    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    base_url: Mapped[str | None] = mapped_column(String(300))
    model_default: Mapped[str | None] = mapped_column(String(120))
    model_pro: Mapped[str | None] = mapped_column(String(120))
    thinking_enabled: Mapped[bool | None] = mapped_column(Boolean)
    reasoning_effort: Mapped[str | None] = mapped_column(String(16))
    max_tokens: Mapped[int | None] = mapped_column(Integer)
    timeout_seconds: Mapped[int | None] = mapped_column(Integer)
    max_retries: Mapped[int | None] = mapped_column(Integer)
    assess_max_turns: Mapped[int | None] = mapped_column(Integer)
    path_min_nodes: Mapped[int | None] = mapped_column(Integer)
    path_max_nodes: Mapped[int | None] = mapped_column(Integer)
    node_max_retries: Mapped[int | None] = mapped_column(Integer)
    compress_threshold_rounds: Mapped[int | None] = mapped_column(Integer)
    compress_round_count: Mapped[int | None] = mapped_column(Integer)
    show_reasoning: Mapped[bool | None] = mapped_column(Boolean)
    log_level: Mapped[str | None] = mapped_column(String(16))
    data_dir: Mapped[str | None] = mapped_column(String(500))
    price_table_json: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class ApiKeySecretRow(Base):
    """API Key 密文（单行，id 恒为 1）。只存密文，明文永不落库（FR-708）。"""

    __tablename__ = "api_key_secret"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cipher: Mapped[str] = mapped_column(Text)
    masked_hint: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(16), default="db")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class ConversationRow(Base):
    """记忆容器：一个主题下的测评 / 教学 / 复习各一条，彼此隔离（FR-601、FR-602）。"""

    __tablename__ = "conversation"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))
    # subject 表在 P2 落地，此处先存字符串外键以免建表顺序耦合
    subject_id: Mapped[str | None] = mapped_column(String(36), index=True)
    round_count: Mapped[int] = mapped_column(Integer, default=0)
    compressed_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ChatRoundRow(Base):
    """一轮对话：一次用户输入及其引发的全部消息（FR-603）。"""

    __tablename__ = "chat_round"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversation.id", ondelete="CASCADE"))
    round_index: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (Index("ix_chat_round_conversation_index", "conversation_id", "round_index", unique=True),)


class MessageRow(Base):
    """协议消息。以原始字段字典形式承载，保留服务端附加字段（如 tool_calls）。"""

    __tablename__ = "message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(ForeignKey("chat_round.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text, default="")
    reasoning_content: Mapped[str | None] = mapped_column(Text)
    tool_calls_json: Mapped[str | None] = mapped_column(Text)
    is_summary: Mapped[bool] = mapped_column(Boolean, default=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class UsageRecordRow(Base):
    """每次 LLM 请求的用量明细（FR-705）。

    缓存命中与未命中都记录，便于定位"命中率异常偏低"（例如压缩改写前缀导致缓存失效）。
    """

    __tablename__ = "usage_record"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[str | None] = mapped_column(String(36), index=True)
    conversation_id: Mapped[str | None] = mapped_column(String(36), index=True)
    purpose: Mapped[str] = mapped_column(String(16))
    model: Mapped[str] = mapped_column(String(120))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_miss_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
