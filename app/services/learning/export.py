"""学习报告导出、记忆与档案导入导出（T4.3，FR-805、FR-613）。

三种形态刻意分开：

- :func:`render_markdown` 产出**给人看**的学习报告，纯 Markdown，不依赖前端渲染；
- :func:`build_archive_json` 产出**给机器看**的备份档案（含三路会话记忆的正文）；
- :func:`import_archive` 把档案恢复成一个**新主题**。

两个关键取舍：

1. **导出记忆时只保留正文**：system 提示词是内部构造、压缩摘要只是历史正文的替代物，
   两者都无法在导入端复原。把它们带过去只会污染新主题的模型输入。
2. **导入时所有 id 重新生成**：id 是库内主键，同一个档案导入两次（或导入回原本就含它的库）
   必然主键冲突。行间引用靠一张"旧 id → 新 id"的映射表重建，与原 id 无关。
"""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.contracts.schemas import ConversationKind, NodeState, SubjectState, WeakPointReason
from app.infra.logging.queue_logger import CHANNEL_APP, get_logger
from app.infra.storage.db import Database
from app.infra.storage.models import (
    AnswerRecordRow,
    ChatRoundRow,
    CheckpointRow,
    ConversationRow,
    KnowledgePointRow,
    LearningPathRow,
    MessageRow,
    SessionStateRow,
    StudyNoteRow,
    SubjectRow,
    WeakPointRow,
    utcnow,
)
from app.services.learning.entities import (
    AnswerRecord,
    KnowledgePoint,
    LearningPath,
    StudyNote,
    Subject,
)
from app.services.learning.repository import new_id
from app.services.learning.stack import LearningStack

ARCHIVE_FORMAT = "welearn-subject"
ARCHIVE_VERSION = 1
IMPORT_SUFFIX = "（导入）"
MAX_TITLE_LENGTH = 200
ANSWER_SCAN_LIMIT = 100_000
"""导出作答记录时的读取上限：导出是"全量"语义，取一个足够大的上界即可。"""

MEMORY_ROLES_EXCLUDED = {"system"}
"""不导出 system 消息：它是内部提示词，导入端无法复原（也不该复原）。"""

_log = get_logger(CHANNEL_APP)


class ExportError(ValueError):
    """档案不合法或不可导出。消息直接面向用户，由 API 层转成 422。"""


NODE_STATE_LABELS: dict[NodeState, str] = {
    NodeState.NOT_STARTED: "未开始",
    NodeState.LEARNING: "学习中",
    NodeState.MASTERED: "已掌握",
    NodeState.PARTIAL: "部分掌握",
    NodeState.SKIPPED: "已跳过",
    NodeState.REVIEW_NEEDED: "待复习",
}

WEAK_REASON_LABELS: dict[WeakPointReason, str] = {
    WeakPointReason.PARTIAL: "判定部分通过",
    WeakPointReason.SKIPPED: "跳过检查点",
    WeakPointReason.SHOW_ANSWER: "直接查看答案",
    WeakPointReason.ADVANCED: "连续未通过后推进",
}

NOTE_SOURCE_LABELS = {"auto": "自动整理", "user": "用户笔记"}


# --------------------------------------------------------------------------- #
# Markdown 报告
# --------------------------------------------------------------------------- #


def render_markdown(stack: LearningStack, subject_id: str) -> str:
    """渲染面向人的学习报告（FR-805）。"""
    subject = stack.repos.subjects.get(subject_id)
    if subject is None:
        raise ExportError("主题不存在。")

    path = stack.repos.paths.get_active(subject_id)
    notes = stack.repos.notes.list_for_subject(subject_id)
    weak_points = stack.repos.weak_points.list_open(subject_id)
    answers = stack.repos.answers.list_for_subject(subject_id, limit=ANSWER_SCAN_LIMIT)
    progress = stack.repos.subjects.progress(subject_id)

    notes_by_node: dict[str | None, StudyNote] = {note.node_id: note for note in notes}
    answer_counts: dict[str, int] = {}
    for record in answers:
        answer_counts[record.node_id] = answer_counts.get(record.node_id, 0) + 1

    lines: list[str] = [f"# 学习报告：{subject.title}", ""]
    lines += [
        f"- 学习目标：{subject.goal or '（未填写）'}",
        f"- 已有背景：{subject.background or '（未填写）'}",
        f"- 期望深度：{subject.depth or '（未填写）'}",
        f"- 状态：{subject.status.value}",
        f"- 创建时间：{_format_time(subject.created_at)}",
        f"- 最近更新：{_format_time(subject.updated_at)}",
    ]
    if subject.archived:
        lines.append(f"- 归档时间：{_format_time(subject.archived_at)}")

    lines += ["", "## 学习路径", ""]
    if path is None:
        lines.append("尚未生成学习路径。")
    else:
        lines += [f"- 版本：{path.version}", f"- 生成理由：{path.rationale or '（未提供）'}", ""]
        lines += ["| # | 节点 | 状态 | 作答次数 |", "| --- | --- | --- | --- |"]
        for node in path.ordered_nodes():
            lines.append(
                f"| {node.order_index + 1} | {_cell(node.title)} | {_state_label(node.state)} "
                f"| {answer_counts.get(node.id, 0)} |"
            )

        for node in path.ordered_nodes():
            lines += _node_section(node, answer_counts.get(node.id, 0), notes_by_node.get(node.id))

    general_note = notes_by_node.get(None)
    if general_note is not None:
        source_label = NOTE_SOURCE_LABELS.get(general_note.source, general_note.source)
        lines += ["", f"## 主题笔记（{source_label}）", "", *_quote(general_note.content)]

    lines += ["", "## 薄弱点", ""]
    if weak_points:
        node_titles = _node_titles(stack, path, [point.node_id for point in weak_points])
        lines += ["| 节点 | 原因 | 未通过次数 |", "| --- | --- | --- |"]
        for point in weak_points:
            title = node_titles.get(point.node_id, point.node_id)
            lines.append(f"| {_cell(title)} | {_reason_label(point.reason)} | {point.fail_count} |")
    else:
        lines.append("暂无未解决的薄弱点。")

    unfinished = max(
        0,
        progress.total_nodes - progress.mastered_nodes - progress.partial_nodes - progress.skipped_nodes,
    )
    lines += [
        "",
        "## 完成情况",
        "",
        f"- 节点总数：{progress.total_nodes}",
        f"- 已掌握：{progress.mastered_nodes}",
        f"- 部分掌握：{progress.partial_nodes}",
        f"- 已跳过：{progress.skipped_nodes}",
        f"- 未完成：{unfinished}",
        f"- 未解决薄弱点：{progress.weak_points}",
        f"- 累计 token：{progress.total_tokens}",
    ]

    return "\n".join(lines).rstrip() + "\n"


def _node_section(node: KnowledgePoint, answer_count: int, note: StudyNote | None) -> list[str]:
    lines = [
        "",
        f"### {node.order_index + 1}. {node.title}",
        "",
        f"- 目标：{node.objective or '（未填写）'}",
        f"- 状态：{_state_label(node.state)}",
        f"- 作答次数：{answer_count}",
    ]
    if node.attempts:
        lines.append(f"- 讲解重试次数：{node.attempts}")
    if node.prerequisites:
        lines.append(f"- 前置知识：{'、'.join(node.prerequisites)}")
    if node.key_points:
        lines += ["", "**要点**", ""]
        lines += [f"- {point}" for point in node.key_points]
    if note is not None:
        lines += ["", f"**笔记（{NOTE_SOURCE_LABELS.get(note.source, note.source)}）**", ""]
        lines += _quote(note.content)
    return lines


def _node_titles(stack: LearningStack, path: LearningPath | None, node_ids: Sequence[str]) -> dict[str, str]:
    """薄弱点可能指向旧版本的节点，活动路径里查不到时再按 id 补查。"""
    titles: dict[str, str] = {}
    if path is not None:
        titles.update({node.id: node.title for node in path.ordered_nodes()})
    for node_id in node_ids:
        if node_id in titles:
            continue
        stale = stack.repos.paths.node(node_id)
        if stale is not None:
            titles[node_id] = stale.title
    return titles


def _state_label(state: NodeState) -> str:
    return NODE_STATE_LABELS.get(state, state.value)


def _reason_label(reason: WeakPointReason) -> str:
    return WEAK_REASON_LABELS.get(reason, reason.value)


def _cell(text: str) -> str:
    """表格单元格：竖线会破坏列结构，换行会让行断开。"""
    return " ".join(text.split()).replace("|", "\\|")


def _quote(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return ["> （空）"]
    return [f"> {line}" if line.strip() else ">" for line in stripped.splitlines()]


def _format_time(value: datetime | None) -> str:
    if value is None:
        return "（未知）"
    # 库里存的是 naive UTC（见 models.utcnow），这里显式标注时区避免误读
    return value.strftime("%Y-%m-%d %H:%M") + " UTC"


# --------------------------------------------------------------------------- #
# JSON 档案
# --------------------------------------------------------------------------- #


def build_archive_json(
    stack: LearningStack,
    subject_id: str,
    *,
    db: Database | None = None,
) -> dict[str, Any]:
    """构造备份/迁移用的结构化档案（FR-613）。"""
    subject = stack.repos.subjects.get(subject_id)
    if subject is None:
        raise ExportError("主题不存在。")

    path = stack.repos.paths.get_active(subject_id)
    database = _resolve_db(stack, db)

    return {
        "format": ARCHIVE_FORMAT,
        "version": ARCHIVE_VERSION,
        "exported_at": _iso(utcnow()),
        "subject": _subject_archive(subject),
        "path": _path_archive(path) if path is not None else None,
        "notes": [_note_archive(note) for note in stack.repos.notes.list_for_subject(subject_id)],
        "weak_points": _weak_points_archive(database, subject_id),
        "answers": [
            _answer_archive(record)
            for record in stack.repos.answers.list_for_subject(subject_id, limit=ANSWER_SCAN_LIMIT)
        ],
        "memory": _memory_archive(stack, subject_id, database),
    }


def _subject_archive(subject: Subject) -> dict[str, Any]:
    return {
        "id": subject.id,
        "title": subject.title,
        "goal": subject.goal,
        "background": subject.background,
        "depth": subject.depth,
        "status": subject.status.value,
        "archived": subject.archived,
        "created_at": _iso(subject.created_at),
        "updated_at": _iso(subject.updated_at),
        "archived_at": _iso(subject.archived_at),
    }


def _path_archive(path: LearningPath) -> dict[str, Any]:
    return {
        "id": path.id,
        "version": path.version,
        "rationale": path.rationale,
        "model": path.model,
        "is_active": path.is_active,
        "created_at": _iso(path.created_at),
        "nodes": [_node_archive(node) for node in path.ordered_nodes()],
    }


def _node_archive(node: KnowledgePoint) -> dict[str, Any]:
    return {
        "id": node.id,
        "order_index": node.order_index,
        "title": node.title,
        "objective": node.objective,
        "prerequisites": list(node.prerequisites),
        "key_points": list(node.key_points),
        "state": node.state.value,
        "attempts": node.attempts,
        "is_quick_review": node.is_quick_review,
        "mastered_at": _iso(node.mastered_at),
        "last_reviewed_at": _iso(node.last_reviewed_at),
        "summary_note": node.summary_note,
        "checkpoints": [
            {
                "id": checkpoint.id,
                "order_index": checkpoint.order_index,
                "question": checkpoint.question,
                "expected_points": list(checkpoint.expected_points),
                "reference_answer": checkpoint.reference_answer,
            }
            for checkpoint in node.checkpoints
        ],
    }


def _note_archive(note: StudyNote) -> dict[str, Any]:
    return {
        "id": note.id,
        "node_id": note.node_id,
        "content": note.content,
        "source": note.source,
        "updated_at": _iso(note.updated_at),
    }


def _weak_points_archive(db: Database, subject_id: str) -> list[dict[str, Any]]:
    """薄弱点连已解除的一起导出：它是学习历史，丢掉会让迁移后的档案不完整。"""
    statement = (
        select(WeakPointRow)
        .where(WeakPointRow.subject_id == subject_id)
        .order_by(WeakPointRow.last_attempt_at.desc(), WeakPointRow.id.desc())
    )
    with db.session() as session:
        rows = list(session.scalars(statement))
    return [
        {
            "id": row.id,
            "node_id": row.node_id,
            "reason": row.reason,
            "fail_count": row.fail_count,
            "last_attempt_at": _iso(row.last_attempt_at),
            "resolved": row.resolved_at is not None,
            "resolved_at": _iso(row.resolved_at),
        }
        for row in rows
    ]


def _answer_archive(record: AnswerRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "node_id": record.node_id,
        "checkpoint_id": record.checkpoint_id,
        "question": record.question,
        "answer": record.user_answer,
        "verdict": record.verdict,
        "confidence": record.confidence,
        "feedback": record.feedback,
        "hit_points": list(record.hit_points),
        "missing_points": list(record.missing_points),
        "attempt_index": record.attempt_index,
        "source": record.source,
        "created_at": _iso(record.created_at),
    }


def _memory_archive(stack: LearningStack, subject_id: str, db: Database) -> dict[str, list[dict[str, Any]]]:
    return {kind.value: _conversation_messages(stack, subject_id, kind, db) for kind in ConversationKind}


def _conversation_messages(
    stack: LearningStack,
    subject_id: str,
    kind: ConversationKind,
    db: Database,
) -> list[dict[str, Any]]:
    """读某用途会话的正文消息。

    ``stack.transcript`` 只返回"可展示的正文"，拿不到 reasoning 与时间戳；导出要保留这些，
    所以直接读表。会话不存在（还没进入该阶段）时返回空列表，而不是报错。
    """
    conversation_id = stack.memories.conversation_id_for(subject_id, kind)
    if conversation_id is None:
        return []

    statement = (
        select(
            MessageRow.role,
            MessageRow.content,
            MessageRow.reasoning_content,
            MessageRow.kind,
            MessageRow.created_at,
        )
        .join(ChatRoundRow, ChatRoundRow.id == MessageRow.round_id)
        .where(
            ChatRoundRow.conversation_id == conversation_id,
            MessageRow.is_summary.is_(False),
            MessageRow.role.notin_(MEMORY_ROLES_EXCLUDED),
            MessageRow.content != "",
        )
        .order_by(ChatRoundRow.round_index, MessageRow.id)
    )
    try:
        with db.session() as session:
            rows = session.execute(statement).all()
    except Exception as exc:  # pragma: no cover - 表结构异常属于环境问题
        _log.warning(f"读取会话消息失败，改用可见正文导出: {exc}")
        rows = []

    if rows:
        return [
            {
                "role": str(row[0]),
                "content": str(row[1]),
                "reasoning_content": row[2],
                "kind": row[3],
                "created_at": _iso(row[4]),
            }
            for row in rows
        ]

    # 兜底：仓储层读不到时退回可见正文（会丢 reasoning 与时间戳）
    return [
        {
            "role": message.get("role", ""),
            "content": message.get("content", ""),
            "reasoning_content": None,
            "kind": None,
            "created_at": None,
        }
        for message in stack.transcript(subject_id, kind)
    ]


# --------------------------------------------------------------------------- #
# 导入
# --------------------------------------------------------------------------- #


def import_archive(db: Database, payload: object) -> str:
    """把档案恢复成一个新主题，返回新 ``subject_id``。"""
    archive = _validate_archive(payload)
    subject_data = _as_mapping(archive["subject"])
    path_data = archive.get("path")
    path_mapping = _as_mapping(path_data) if path_data is not None else None

    new_subject_id = new_id()
    title = _import_title(str(subject_data.get("title", "")))
    status = _parse_subject_state(subject_data.get("status"))

    node_map: dict[str, str] = {}
    checkpoint_map: dict[str, str] = {}

    with db.session() as session:
        session.add(
            SubjectRow(
                id=new_subject_id,
                title=title,
                goal=_optional_text(subject_data.get("goal")),
                background=_optional_text(subject_data.get("background")),
                depth=_optional_text(subject_data.get("depth")),
                status=status.value,
                created_at=_parse_time(subject_data.get("created_at")) or utcnow(),
                updated_at=_parse_time(subject_data.get("updated_at")) or utcnow(),
                archived_at=_parse_time(subject_data.get("archived_at")),
            )
        )
        # 主题与状态必须同时存在，否则导入后的主题连"继续学习"都点不动
        session.add(SessionStateRow(subject_id=new_subject_id, state=status.value))

        if path_mapping is not None:
            path_id = new_id()
            session.add(
                LearningPathRow(
                    id=path_id,
                    subject_id=new_subject_id,
                    # 导入的路径是这条主题的新起点：版本从 1 开始，且必须是活动版本
                    version=1,
                    rationale=str(path_mapping.get("rationale") or ""),
                    model=_optional_text(path_mapping.get("model")),
                    is_active=True,
                    created_at=_parse_time(path_mapping.get("created_at")) or utcnow(),
                )
            )
            session.flush()
            _insert_nodes(session, path_id, path_mapping, node_map, checkpoint_map)

        _insert_notes(session, new_subject_id, archive.get("notes"), node_map)
        _insert_weak_points(session, new_subject_id, archive.get("weak_points"), node_map)
        _insert_answers(session, new_subject_id, archive.get("answers"), node_map, checkpoint_map)

    _restore_memory(db, new_subject_id, archive.get("memory"))
    _log.info(f"导入学习档案: {title}（id={new_subject_id}）")
    return new_subject_id


def _insert_nodes(
    session: Session,
    path_id: str,
    path_data: Mapping[str, Any],
    node_map: dict[str, str],
    checkpoint_map: dict[str, str],
) -> None:
    pending: list[tuple[str, list[Mapping[str, Any]]]] = []
    nodes = path_data.get("nodes")
    if nodes is None:
        nodes = []
    if not isinstance(nodes, list):
        raise ExportError("档案中的路径缺少 nodes 列表。")

    for position, raw_node in enumerate(nodes):
        node = _as_mapping(raw_node)
        node_title = node.get("title")
        if not isinstance(node_title, str) or not node_title.strip():
            raise ExportError("档案中的节点缺少标题。")

        node_id = new_id()
        old_id = node.get("id")
        if isinstance(old_id, str) and old_id:
            node_map[old_id] = node_id

        order_index = node.get("order_index")
        session.add(
            KnowledgePointRow(
                id=node_id,
                path_id=path_id,
                order_index=order_index if isinstance(order_index, int) else position,
                title=node_title.strip(),
                objective=str(node.get("objective") or ""),
                prerequisites_json=_dump_list(node.get("prerequisites")),
                key_points_json=_dump_list(node.get("key_points")),
                state=_parse_node_state(node.get("state")).value,
                attempts=_as_int(node.get("attempts")),
                is_quick_review=bool(node.get("is_quick_review", False)),
                mastered_at=_parse_time(node.get("mastered_at")),
                last_reviewed_at=_parse_time(node.get("last_reviewed_at")),
                summary_note=_optional_text(node.get("summary_note")),
            )
        )
        pending.append((node_id, _as_mapping_list(node.get("checkpoints"))))

    session.flush()  # checkpoint 引用 node，必须先落盘节点行

    for node_id, checkpoints in pending:
        for order, checkpoint in enumerate(checkpoints):
            question = checkpoint.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ExportError("档案中的检查点缺少问题。")
            checkpoint_id = new_id()
            old_id = checkpoint.get("id")
            if isinstance(old_id, str) and old_id:
                checkpoint_map[old_id] = checkpoint_id
            session.add(
                CheckpointRow(
                    id=checkpoint_id,
                    node_id=node_id,
                    order_index=order,
                    question=question.strip(),
                    expected_points_json=_dump_list(checkpoint.get("expected_points")),
                    reference_answer=str(checkpoint.get("reference_answer") or ""),
                )
            )


def _insert_notes(session: Session, subject_id: str, raw_notes: object, node_map: Mapping[str, str]) -> None:
    for raw_note in _as_mapping_list(raw_notes):
        content = str(raw_note.get("content") or "")
        if not content.strip():
            continue
        old_node_id = raw_note.get("node_id")
        node_id: str | None = None
        if isinstance(old_node_id, str) and old_node_id:
            node_id = node_map.get(old_node_id)
            if node_id is None:
                # 引用不到节点的笔记无法挂载；留成孤儿会在档案页上指向不存在的节点
                _log.warning(f"导入时跳过引用了未知节点的笔记: {old_node_id}")
                continue
        source = str(raw_note.get("source") or "auto")
        session.add(
            StudyNoteRow(
                id=new_id(),
                subject_id=subject_id,
                node_id=node_id,
                content=content,
                source=source if source in NOTE_SOURCE_LABELS else "auto",
                updated_at=_parse_time(raw_note.get("updated_at")) or utcnow(),
            )
        )


def _insert_weak_points(
    session: Session,
    subject_id: str,
    raw_points: object,
    node_map: Mapping[str, str],
) -> None:
    for raw_point in _as_mapping_list(raw_points):
        old_node_id = raw_point.get("node_id")
        node_id = node_map.get(str(old_node_id)) if isinstance(old_node_id, str) else None
        if node_id is None:
            _log.warning(f"导入时跳过引用了未知节点的薄弱点: {old_node_id}")
            continue
        try:
            reason = WeakPointReason(str(raw_point.get("reason") or WeakPointReason.PARTIAL.value))
        except ValueError:
            _log.warning(f"导入时跳过成因非法的薄弱点: {raw_point.get('reason')!r}")
            continue
        session.add(
            WeakPointRow(
                subject_id=subject_id,
                node_id=node_id,
                reason=reason.value,
                fail_count=max(1, _as_int(raw_point.get("fail_count"))),
                last_attempt_at=_parse_time(raw_point.get("last_attempt_at")) or utcnow(),
                resolved_at=_parse_time(raw_point.get("resolved_at")),
            )
        )


def _insert_answers(
    session: Session,
    subject_id: str,
    raw_answers: object,
    node_map: Mapping[str, str],
    checkpoint_map: Mapping[str, str],
) -> None:
    for raw_answer in _as_mapping_list(raw_answers):
        old_node_id = raw_answer.get("node_id")
        node_id = node_map.get(str(old_node_id)) if isinstance(old_node_id, str) else None
        if node_id is None:
            _log.warning(f"导入时跳过引用了未知节点的作答记录: {old_node_id}")
            continue
        old_checkpoint_id = raw_answer.get("checkpoint_id")
        checkpoint_id = (
            checkpoint_map.get(str(old_checkpoint_id)) if isinstance(old_checkpoint_id, str) else None
        )
        confidence = raw_answer.get("confidence")
        session.add(
            AnswerRecordRow(
                subject_id=subject_id,
                node_id=node_id,
                checkpoint_id=checkpoint_id,
                question=str(raw_answer.get("question") or ""),
                user_answer=str(raw_answer.get("answer") or ""),
                verdict=str(raw_answer.get("verdict") or "fail"),
                confidence=float(confidence) if isinstance(confidence, int | float) else None,
                feedback=str(raw_answer.get("feedback") or ""),
                hit_points_json=_dump_list(raw_answer.get("hit_points")),
                missing_points_json=_dump_list(raw_answer.get("missing_points")),
                attempt_index=max(1, _as_int(raw_answer.get("attempt_index"))),
                # 保留来源：复习通过记录是"学过了"的证据，改成 teach 会让统计失真（FR-507）
                source=str(raw_answer.get("source") or "teach"),
                created_at=_parse_time(raw_answer.get("created_at")) or utcnow(),
            )
        )


def _restore_memory(db: Database, subject_id: str, raw_memory: object) -> None:
    """恢复三路会话记忆。

    记忆是"锦上添花"：结构异常时记日志跳过，不能让整个档案导入失败——主题、路径与作答
    都已经落库，此时回滚反而丢掉了用户真正想要的数据。
    """
    if not isinstance(raw_memory, Mapping):
        return

    for kind in ConversationKind:
        raw_messages = raw_memory.get(kind.value)
        messages = [
            message
            for message in _as_mapping_list(raw_messages)
            if isinstance(message.get("role"), str)
            and str(message.get("role")) not in MEMORY_ROLES_EXCLUDED
            and str(message.get("content") or "").strip()
        ]
        if not messages:
            continue
        try:
            _write_conversation(db, subject_id, kind, messages)
        except Exception as exc:
            _log.warning(f"恢复 {kind.value} 会话记忆失败，已跳过: {exc}")


def _write_conversation(
    db: Database,
    subject_id: str,
    kind: ConversationKind,
    messages: Sequence[Mapping[str, Any]],
) -> None:
    with db.session() as session:
        conversation = ConversationRow(id=new_id(), kind=kind.value, subject_id=subject_id, round_count=0)
        session.add(conversation)
        session.flush()

        round_row: ChatRoundRow | None = None
        round_index = 0
        for message in messages:
            role = str(message.get("role"))
            # 一轮以用户输入开头；连续出现的 assistant 消息留在同一轮里
            if round_row is None or role == "user":
                round_row = ChatRoundRow(conversation_id=conversation.id, round_index=round_index)
                session.add(round_row)
                session.flush()
                round_index += 1
            session.add(
                MessageRow(
                    round_id=round_row.id,
                    role=role,
                    content=str(message.get("content") or ""),
                    reasoning_content=_optional_text(message.get("reasoning_content")),
                    is_summary=False,
                    kind=_optional_text(message.get("kind")),
                    created_at=_parse_time(message.get("created_at")) or utcnow(),
                )
            )
        conversation.round_count = round_index
        session.add(conversation)


# --------------------------------------------------------------------------- #
# 校验与转换
# --------------------------------------------------------------------------- #


def _validate_archive(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ExportError("档案内容必须是一个 JSON 对象。")
    archive = dict(payload)

    if archive.get("format") != ARCHIVE_FORMAT:
        raise ExportError("这不是 WeLearn 学习档案（format 字段不匹配）。")

    version = archive.get("version")
    if not isinstance(version, int) or version != ARCHIVE_VERSION:
        raise ExportError(f"不支持的档案版本：{version!r}（当前仅支持 {ARCHIVE_VERSION}）。")

    subject = archive.get("subject")
    if not isinstance(subject, Mapping):
        raise ExportError("档案缺少 subject 字段。")
    title = subject.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ExportError("档案中的主题标题为空。")

    for key in ("notes", "weak_points", "answers"):
        value = archive.get(key)
        if value is not None and not isinstance(value, list):
            raise ExportError(f"档案中的 {key} 字段必须是数组。")

    path = archive.get("path")
    if path is not None:
        if not isinstance(path, Mapping):
            raise ExportError("档案中的 path 字段格式不正确。")
        nodes = path.get("nodes")
        if nodes is not None and not isinstance(nodes, list):
            raise ExportError("档案中的 nodes 字段必须是数组。")

    return archive


def _as_mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExportError("档案结构不正确。")
    return value


def _as_mapping_list(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _parse_subject_state(raw: object) -> SubjectState:
    try:
        return SubjectState(str(raw or SubjectState.DRAFT.value))
    except ValueError as exc:
        raise ExportError(f"档案中的主题状态非法：{raw!r}") from exc


def _parse_node_state(raw: object) -> NodeState:
    try:
        return NodeState(str(raw or NodeState.NOT_STARTED.value))
    except ValueError as exc:
        raise ExportError(f"档案中的节点状态非法：{raw!r}") from exc


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return 0


def _optional_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _dump_list(value: object) -> str | None:
    if isinstance(value, str):
        items = [value.strip()] if value.strip() else []
    elif isinstance(value, list | tuple):
        items = [str(item).strip() for item in value if item is not None and str(item).strip()]
    else:
        items = []
    return json.dumps(items, ensure_ascii=False) if items else None


def _import_title(title: str) -> str:
    """导入的主题加后缀：原主题可能还在库里，同名会让用户无从分辨。"""
    trimmed = title.strip()
    if not trimmed.endswith(IMPORT_SUFFIX):
        trimmed = f"{trimmed}{IMPORT_SUFFIX}"
    return trimmed[:MAX_TITLE_LENGTH]


def _resolve_db(stack: LearningStack, db: Database | None) -> Database:
    """拿到原始表访问入口。

    记忆仓储只暴露"可见正文"，而档案需要 reasoning 与时间戳，因此必须直连数据库。
    API 层会显式传入 ``ctx.db``；直接调用服务时退回仓储持有的同一个连接。
    """
    if db is not None:
        return db
    candidate = getattr(stack.memories, "_db", None)
    if isinstance(candidate, Database):
        return candidate
    raise ExportError("无法访问数据库连接，导出失败。")
