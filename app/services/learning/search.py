"""全局检索（T4.4，FR-803）。

检索范围固定为三类：节点笔记、作答记录、对话消息正文。三张表各发一条查询，
在 SQL 里完成匹配与排序（**不做 N+1**：主题标题与节点标题都用 JOIN 带回来）。

大小写不敏感靠 ``ilike``（SQLite 编译为 ``lower(a) LIKE lower(b)``，ASCII 生效；
中文本身无大小写，按字面匹配即可）。
"""

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import or_, select

from app.infra.storage.db import Database
from app.infra.storage.models import (
    AnswerRecordRow,
    ChatRoundRow,
    ConversationRow,
    KnowledgePointRow,
    MessageRow,
    StudyNoteRow,
    SubjectRow,
)

DEFAULT_LIMIT = 20
MAX_LIMIT = 100
SNIPPET_RADIUS = 60
"""命中词前后各保留的字符数。"""

LIKE_ESCAPE = "\\"


@dataclass(frozen=True)
class NoteHit:
    subject_id: str
    subject_title: str
    node_id: str | None
    node_title: str | None
    snippet: str
    updated_at: datetime | None


@dataclass(frozen=True)
class AnswerHit:
    subject_id: str
    subject_title: str
    node_id: str
    node_title: str | None
    verdict: str
    snippet: str
    created_at: datetime | None
    source: str


@dataclass(frozen=True)
class MessageHit:
    subject_id: str | None
    subject_title: str | None
    conversation_kind: str | None
    role: str
    snippet: str
    created_at: datetime | None


@dataclass(frozen=True)
class SearchResults:
    query: str
    notes: list[NoteHit] = field(default_factory=list)
    answers: list[AnswerHit] = field(default_factory=list)
    messages: list[MessageHit] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.notes) + len(self.answers) + len(self.messages)


def search(
    db: Database,
    *,
    query: str,
    subject_id: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> SearchResults:
    """跨笔记 / 作答 / 对话消息检索关键词。

    空查询返回空结果而不是报错：前端输入框是"边打字边请求"，空串是常态。
    """
    text = (query or "").strip()
    if not text:
        return SearchResults(query="")

    size = max(1, min(int(limit), MAX_LIMIT))
    pattern = f"%{_escape_like(text)}%"

    return SearchResults(
        query=text,
        notes=_search_notes(db, pattern, text, subject_id, size),
        answers=_search_answers(db, pattern, text, subject_id, size),
        messages=_search_messages(db, pattern, text, subject_id, size),
    )


def make_snippet(text: str, query: str, *, radius: int = SNIPPET_RADIUS) -> str:
    """截取命中词附近的片段；两端截断时用 ``…`` 标注。"""
    compact = " ".join((text or "").split())
    if not compact:
        return ""
    index = compact.lower().find(query.lower())
    if index < 0:
        index = 0
    start = max(0, index - radius)
    end = min(len(compact), index + len(query) + radius)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(compact) else ""
    return f"{prefix}{compact[start:end]}{suffix}"


def _escape_like(text: str) -> str:
    """把用户输入里的 LIKE 通配符转义成字面量，否则输入 ``%`` 会匹配一切。"""
    return text.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2).replace("%", f"{LIKE_ESCAPE}%").replace("_", f"{LIKE_ESCAPE}_")


def _search_notes(
    db: Database,
    pattern: str,
    query: str,
    subject_id: str | None,
    limit: int,
) -> list[NoteHit]:
    statement = (
        select(StudyNoteRow, SubjectRow.title, KnowledgePointRow.title)
        .join(SubjectRow, SubjectRow.id == StudyNoteRow.subject_id)
        .outerjoin(KnowledgePointRow, KnowledgePointRow.id == StudyNoteRow.node_id)
        .where(StudyNoteRow.content.ilike(pattern, escape=LIKE_ESCAPE))
    )
    if subject_id is not None:
        statement = statement.where(StudyNoteRow.subject_id == subject_id)
    # 时间倒序：SQLite 的 NULL 最小，DESC 时自然排在最后
    statement = statement.order_by(StudyNoteRow.updated_at.desc(), StudyNoteRow.id).limit(limit)

    with db.session() as session:
        rows = session.execute(statement).all()

    return [
        NoteHit(
            subject_id=row[0].subject_id,
            subject_title=str(row[1]),
            node_id=row[0].node_id,
            node_title=str(row[2]) if row[2] is not None else None,
            snippet=make_snippet(row[0].content, query),
            updated_at=row[0].updated_at,
        )
        for row in rows
    ]


def _search_answers(
    db: Database,
    pattern: str,
    query: str,
    subject_id: str | None,
    limit: int,
) -> list[AnswerHit]:
    statement = (
        select(AnswerRecordRow, SubjectRow.title, KnowledgePointRow.title)
        .join(SubjectRow, SubjectRow.id == AnswerRecordRow.subject_id)
        .outerjoin(KnowledgePointRow, KnowledgePointRow.id == AnswerRecordRow.node_id)
        .where(
            or_(
                AnswerRecordRow.question.ilike(pattern, escape=LIKE_ESCAPE),
                AnswerRecordRow.user_answer.ilike(pattern, escape=LIKE_ESCAPE),
                AnswerRecordRow.feedback.ilike(pattern, escape=LIKE_ESCAPE),
            )
        )
    )
    if subject_id is not None:
        statement = statement.where(AnswerRecordRow.subject_id == subject_id)
    statement = statement.order_by(AnswerRecordRow.created_at.desc(), AnswerRecordRow.id.desc()).limit(limit)

    with db.session() as session:
        rows = session.execute(statement).all()

    hits: list[AnswerHit] = []
    for row in rows:
        record = row[0]
        snippet = _first_match_snippet(
            (record.question, record.user_answer, record.feedback),
            query,
        )
        hits.append(
            AnswerHit(
                subject_id=record.subject_id,
                subject_title=str(row[1]),
                node_id=record.node_id,
                node_title=str(row[2]) if row[2] is not None else None,
                verdict=record.verdict,
                snippet=snippet,
                created_at=record.created_at,
                source=record.source,
            )
        )
    return hits


def _search_messages(
    db: Database,
    pattern: str,
    query: str,
    subject_id: str | None,
    limit: int,
) -> list[MessageHit]:
    """对话消息可能极多，因此一律按时间倒序取前 ``limit`` 条命中。

    摘要消息不是用户说过的话（只是历史正文的替代物），system 是内部提示词，两者都不该
    出现在检索结果里——用户搜不到"自己没说过的话"才符合直觉。
    """
    statement = (
        select(MessageRow, ConversationRow.kind, ConversationRow.subject_id, SubjectRow.title)
        .join(ChatRoundRow, ChatRoundRow.id == MessageRow.round_id)
        .join(ConversationRow, ConversationRow.id == ChatRoundRow.conversation_id)
        .outerjoin(SubjectRow, SubjectRow.id == ConversationRow.subject_id)
        .where(
            MessageRow.content.ilike(pattern, escape=LIKE_ESCAPE),
            MessageRow.is_summary.is_(False),
            MessageRow.role != "system",
            MessageRow.content != "",
        )
    )
    if subject_id is not None:
        statement = statement.where(ConversationRow.subject_id == subject_id)
    statement = statement.order_by(MessageRow.created_at.desc(), MessageRow.id.desc()).limit(limit)

    with db.session() as session:
        rows = session.execute(statement).all()

    return [
        MessageHit(
            subject_id=row[2],
            subject_title=str(row[3]) if row[3] is not None else None,
            conversation_kind=str(row[1]),
            role=row[0].role,
            snippet=make_snippet(row[0].content, query),
            created_at=row[0].created_at,
        )
        for row in rows
    ]


def _first_match_snippet(candidates: tuple[str, ...], query: str) -> str:
    """多个字段都可能命中：取第一个真正命中的字段做片段，片段才一定包含关键词。"""
    lowered = query.lower()
    for candidate in candidates:
        if candidate and lowered in candidate.lower():
            return make_snippet(candidate, query)
    return make_snippet(candidates[0] if candidates else "", query)
