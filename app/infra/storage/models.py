"""SQLAlchemy 表模型（T0.6 数据库契约的代码部分）。

v1：设置、密钥、会话记忆、用量（P0/P1）。
**v2（P2）**：学习领域实体 —— subject / session_state / assessment / assessment_level /
learning_path / knowledge_point / checkpoint / answer_record / weak_point / study_note。

时间戳约定：统一存 **UTC**，展示时再转本地时区（SRS 第 6 章）。
"""

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
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
    kind: Mapped[str | None] = mapped_column(String(24))
    """消息用途标记（v4）。目前用于区分检查点问答与"自由提问"（FR-411）。

    它是**元信息**，不会进入发给模型的协议字典——与 ``is_summary`` 同理。
    """
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


# --------------------------------------------------------------------------- #
# v2：学习领域表（P2）
# --------------------------------------------------------------------------- #


class SubjectRow(Base):
    """学习主题 = 一个独立的学习空间（FR-101~FR-106）。"""

    __tablename__ = "subject"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    goal: Mapped[str | None] = mapped_column(Text)
    background: Mapped[str | None] = mapped_column(Text)
    depth: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(24), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime)


class SessionStateRow(Base):
    """会话状态机（FR-410）。

    ``pending_checkpoint_id`` / ``pending_question`` 是"刷新后恢复到待回答的检查点"的关键字段：
    没有它们就只能恢复到最后一条消息，无法知道"系统正在等用户回答哪一个问题"。

    ``awaiting_decision`` 记录"重试已耗尽、正在等用户三选一"这一事实。它必须落库而不是靠前端记忆：
    用户在这里关掉页面再打开，仍然要看到三个选项，而不是被重新讲一遍（FR-408）。

    ``last_action`` 记录最近一次教学动作（``explain`` / ``reteach:simpler`` / ``hint:2`` …），
    供"重新生成上一条回复"（FR-413）知道该重跑哪一个动作——不知道上一步做了什么就没法重跑。
    """

    __tablename__ = "session_state"

    subject_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    state: Mapped[str] = mapped_column(String(24))
    current_node_id: Mapped[str | None] = mapped_column(String(36))
    pending_checkpoint_id: Mapped[str | None] = mapped_column(String(36))
    pending_question: Mapped[str | None] = mapped_column(Text)
    node_attempts: Mapped[int] = mapped_column(Integer, default=0)
    taught_node_id: Mapped[str | None] = mapped_column(String(36))
    awaiting_decision: Mapped[bool] = mapped_column(Boolean, default=False)
    last_action: Mapped[str | None] = mapped_column(String(32))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class AssessmentRow(Base):
    """知识基础确认（FR-201~FR-208）。"""

    __tablename__ = "assessment"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(36), index=True)
    status: Mapped[str] = mapped_column(String(16), default="running")
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    max_turns: Mapped[int] = mapped_column(Integer, default=6)
    topics_json: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    suggested_start_node: Mapped[str | None] = mapped_column(Text)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AssessmentLevelRow(Base):
    """画像明细：子领域 × 掌握水平（FR-206）。"""

    __tablename__ = "assessment_level"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    assessment_id: Mapped[str] = mapped_column(
        ForeignKey("assessment.id", ondelete="CASCADE"), index=True
    )
    topic: Mapped[str] = mapped_column(String(120))
    level: Mapped[str] = mapped_column(String(16))
    evidence: Mapped[str] = mapped_column(Text)
    user_overridden: Mapped[bool] = mapped_column(Boolean, default=False)


class LearningPathRow(Base):
    """学习路径（FR-301~FR-308）。同一主题可保留多个版本，但只有一个 active。"""

    __tablename__ = "learning_path"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(36), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    rationale: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str | None] = mapped_column(String(120))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class KnowledgePointRow(Base):
    """知识点 / 教学节点（FR-302、FR-309）。"""

    __tablename__ = "knowledge_point"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    path_id: Mapped[str] = mapped_column(ForeignKey("learning_path.id", ondelete="CASCADE"), index=True)
    order_index: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(200))
    objective: Mapped[str] = mapped_column(Text, default="")
    prerequisites_json: Mapped[str | None] = mapped_column(Text)
    key_points_json: Mapped[str | None] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(24), default="not_started", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    is_quick_review: Mapped[bool] = mapped_column(Boolean, default=False)
    mastered_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime)
    summary_note: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index("ix_knowledge_point_path_order", "path_id", "order_index"),)


class CheckpointRow(Base):
    """检查点：必须可判定，因此必须有判定要点与参考答案（FR-302）。"""

    __tablename__ = "checkpoint"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_id: Mapped[str] = mapped_column(ForeignKey("knowledge_point.id", ondelete="CASCADE"), index=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    question: Mapped[str] = mapped_column(Text)
    expected_points_json: Mapped[str | None] = mapped_column(Text)
    reference_answer: Mapped[str] = mapped_column(Text, default="")


class AnswerRecordRow(Base):
    """作答与判定记录（FR-404、FR-802）。"""

    __tablename__ = "answer_record"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[str] = mapped_column(String(36), index=True)
    node_id: Mapped[str] = mapped_column(String(36), index=True)
    checkpoint_id: Mapped[str | None] = mapped_column(String(36))
    question: Mapped[str] = mapped_column(Text, default="")
    user_answer: Mapped[str] = mapped_column(Text, default="")
    verdict: Mapped[str] = mapped_column(String(16))
    confidence: Mapped[float | None] = mapped_column(Float)
    feedback: Mapped[str] = mapped_column(Text, default="")
    hit_points_json: Mapped[str | None] = mapped_column(Text)
    missing_points_json: Mapped[str | None] = mapped_column(Text)
    attempt_index: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class WeakPointRow(Base):
    """薄弱点（FR-502~FR-504）。"""

    __tablename__ = "weak_point"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[str] = mapped_column(String(36), index=True)
    node_id: Mapped[str] = mapped_column(String(36), index=True)
    reason: Mapped[str] = mapped_column(String(32))
    fail_count: Mapped[int] = mapped_column(Integer, default=1)
    last_attempt_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)


class StudyNoteRow(Base):
    """节点笔记（FR-801）。"""

    __tablename__ = "study_note"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(36), index=True)
    node_id: Mapped[str | None] = mapped_column(String(36), index=True)
    content: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(16), default="auto")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
