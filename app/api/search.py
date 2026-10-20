"""全局检索接口（T4.4，FR-803）。

``q`` 未填写或只有空白时返回空结果：前端是边输入边请求，空串属于正常输入而不是错误。
"""

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_context
from app.api.serializers import iso
from app.main_context import AppContext
from app.services.learning.search import (
    DEFAULT_LIMIT,
    AnswerHit,
    MessageHit,
    NoteHit,
    search,
)

router = APIRouter(prefix="/api", tags=["search"])


def _note_payload(hit: NoteHit) -> dict[str, Any]:
    return {
        "subject_id": hit.subject_id,
        "subject_title": hit.subject_title,
        "node_id": hit.node_id,
        "node_title": hit.node_title,
        "snippet": hit.snippet,
        "updated_at": iso(hit.updated_at),
    }


def _answer_payload(hit: AnswerHit) -> dict[str, Any]:
    return {
        "subject_id": hit.subject_id,
        "subject_title": hit.subject_title,
        "node_id": hit.node_id,
        "node_title": hit.node_title,
        "verdict": hit.verdict,
        "snippet": hit.snippet,
        "created_at": iso(hit.created_at),
        "source": hit.source,
    }


def _message_payload(hit: MessageHit) -> dict[str, Any]:
    return {
        "subject_id": hit.subject_id,
        "subject_title": hit.subject_title,
        "conversation_kind": hit.conversation_kind,
        "role": hit.role,
        "snippet": hit.snippet,
        "created_at": iso(hit.created_at),
    }


@router.get("/search")
def search_everything(
    q: str = "",
    subject_id: str | None = None,
    limit: int = Query(default=DEFAULT_LIMIT, ge=1),
    ctx: AppContext = Depends(get_context),
) -> dict[str, Any]:
    """跨笔记、作答与对话消息检索。``limit`` 超过上限时按上限截断，而不是报错。"""
    results = search(ctx.db, query=q, subject_id=subject_id, limit=limit)
    return {
        "query": results.query,
        "total": results.total,
        "notes": [_note_payload(hit) for hit in results.notes],
        "answers": [_answer_payload(hit) for hit in results.answers],
        "messages": [_message_payload(hit) for hit in results.messages],
    }
