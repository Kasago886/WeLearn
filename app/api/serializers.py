"""领域实体 → 响应字典（p2-api-sse）。

放在一处集中转换，避免每个路由各写一套字段名（那样前端会面对不一致的命名）。
时间统一用 ISO 8601（UTC），由前端负责本地化展示。
"""

from datetime import datetime
from typing import Any

from app.services.learning.entities import (
    AnswerRecord,
    Assessment,
    Checkpoint,
    KnowledgePoint,
    LearningPath,
    ProfileLevel,
    SessionSnapshot,
    StudyNote,
    Subject,
    SubjectSummary,
    WeakPoint,
)


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def subject_payload(subject: Subject) -> dict[str, Any]:
    return {
        "id": subject.id,
        "title": subject.title,
        "goal": subject.goal,
        "background": subject.background,
        "depth": subject.depth,
        "status": subject.status.value,
        "archived": subject.archived,
        "created_at": iso(subject.created_at),
        "updated_at": iso(subject.updated_at),
        "archived_at": iso(subject.archived_at),
    }


def subject_summary_payload(summary: SubjectSummary) -> dict[str, Any]:
    progress = summary.progress
    return {
        **subject_payload(summary.subject),
        "progress": {
            "total_nodes": progress.total_nodes,
            "mastered_nodes": progress.mastered_nodes,
            "partial_nodes": progress.partial_nodes,
            "skipped_nodes": progress.skipped_nodes,
            "weak_points": progress.weak_points,
            "total_tokens": progress.total_tokens,
            "mastery_ratio": round(progress.mastery_ratio, 3),
        },
        "last_studied_at": iso(summary.last_studied_at),
    }


def checkpoint_payload(checkpoint: Checkpoint) -> dict[str, Any]:
    return {
        "id": checkpoint.id,
        "order_index": checkpoint.order_index,
        "question": checkpoint.question,
        "expected_points": list(checkpoint.expected_points),
        "reference_answer": checkpoint.reference_answer,
    }


def node_payload(node: KnowledgePoint) -> dict[str, Any]:
    return {
        "id": node.id,
        # 位置一律从 1 开始，与 node_state 事件的 index 保持同一套语义
        "index": node.order_index + 1,
        "title": node.title,
        "objective": node.objective,
        "prerequisites": list(node.prerequisites),
        "key_points": list(node.key_points),
        "state": node.state.value,
        "attempts": node.attempts,
        "is_quick_review": node.is_quick_review,
        "summary_note": node.summary_note,
        "mastered_at": iso(node.mastered_at),
        "checkpoints": [checkpoint_payload(item) for item in node.checkpoints],
    }


def path_payload(path: LearningPath) -> dict[str, Any]:
    return {
        "id": path.id,
        "version": path.version,
        "rationale": path.rationale,
        "model": path.model,
        "is_active": path.is_active,
        "created_at": iso(path.created_at),
        "nodes": [node_payload(node) for node in path.ordered_nodes()],
    }


def profile_payload(level: ProfileLevel) -> dict[str, Any]:
    return {
        "topic": level.topic,
        "level": level.level.value,
        "evidence": level.evidence,
        "user_overridden": level.user_overridden,
    }


def assessment_payload(assessment: Assessment, levels: list[ProfileLevel]) -> dict[str, Any]:
    return {
        "id": assessment.id,
        "status": assessment.status.value,
        "turn_count": assessment.turn_count,
        "max_turns": assessment.max_turns,
        "turns_left": assessment.turns_left,
        "topics": list(assessment.topics),
        "summary": assessment.summary,
        "suggested_start_node": assessment.suggested_start_node,
        "skipped": assessment.skipped,
        "completed_at": iso(assessment.completed_at),
        "levels": [profile_payload(level) for level in levels],
    }


def session_payload(snapshot: SessionSnapshot, *, next_action: str) -> dict[str, Any]:
    return {
        "state": snapshot.state.value,
        "next_action": next_action,
        "current_node_id": snapshot.current_node_id,
        "pending_checkpoint_id": snapshot.pending_checkpoint_id,
        "pending_question": snapshot.pending_question,
        "node_attempts": snapshot.node_attempts,
        "awaiting_decision": snapshot.needs_user_decision,
        "node_taught": snapshot.node_taught,
        "updated_at": iso(snapshot.updated_at),
    }


def weak_point_payload(point: WeakPoint) -> dict[str, Any]:
    return {
        "id": point.id,
        "node_id": point.node_id,
        "reason": point.reason.value,
        "fail_count": point.fail_count,
        "last_attempt_at": iso(point.last_attempt_at),
        "resolved": point.resolved,
    }


def answer_payload(record: AnswerRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "node_id": record.node_id,
        "question": record.question,
        "answer": record.user_answer,
        "verdict": record.verdict,
        "confidence": record.confidence,
        "feedback": record.feedback,
        "hit_points": list(record.hit_points),
        "missing_points": list(record.missing_points),
        "attempt_index": record.attempt_index,
        "created_at": iso(record.created_at),
    }


def note_payload(note: StudyNote) -> dict[str, Any]:
    return {
        "id": note.id,
        "node_id": note.node_id,
        "content": note.content,
        "source": note.source,
        "updated_at": iso(note.updated_at),
    }
