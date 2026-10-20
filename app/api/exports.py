"""学习报告导出与档案导入接口（T4.3，FR-805、FR-613）。

**为什么返回 content 而不是文件流**：前端用 Blob + ``<a download>`` 就能落盘，
而 ``StreamingResponse`` 会让"导出内容是否正确"变得难测——测试要读完整个流再断字节。
导出是几十 KB 级别的文本，一次性返回没有内存压力。
"""

import json
import re
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from app.api.deps import get_context, get_stack, require_subject
from app.main_context import AppContext
from app.services.learning.export import (
    ExportError,
    build_archive_json,
    import_archive,
    render_markdown,
)
from app.services.learning.stack import LearningStack

router = APIRouter(prefix="/api/subjects", tags=["exports"])

SUPPORTED_FORMATS = {"markdown", "json"}
EXTENSIONS = {"markdown": "md", "json": "json"}

# 文件名要落到用户的操作系统上：去掉各平台的非法字符与控制字符
_UNSAFE_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")


def _safe_filename(title: str, extension: str) -> str:
    cleaned = _UNSAFE_FILENAME.sub("", title).strip().strip(".")
    cleaned = _WHITESPACE.sub(" ", cleaned)[:60].strip() or "subject"
    # 日期让同名主题的多次导出在下载目录里自然区分开
    return f"{cleaned}_{datetime.now(UTC).strftime('%Y%m%d')}.{extension}"


@router.get("/{subject_id}/export")
def export_subject(
    subject_id: str,
    format: str = Query(default="markdown"),
    stack: LearningStack = Depends(get_stack),
    ctx: AppContext = Depends(get_context),
) -> dict[str, Any]:
    """导出学习报告（markdown）或完整档案（json）。"""
    require_subject(stack, subject_id)
    if format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="format 仅支持 markdown 或 json。",
        )

    subject = stack.subjects.get(subject_id)
    extension = EXTENSIONS[format]
    if format == "json":
        archive = build_archive_json(stack, subject_id, db=ctx.db)
        content = json.dumps(archive, ensure_ascii=False, indent=2)
    else:
        content = render_markdown(stack, subject_id)

    return {
        "filename": _safe_filename(subject.title, extension),
        "content": content,
        "format": format,
    }


@router.post("/import")
def import_subject(
    payload: dict[str, Any] = Body(...),
    stack: LearningStack = Depends(get_stack),
    ctx: AppContext = Depends(get_context),
) -> dict[str, Any]:
    """把导出的 JSON 档案导入成一个新主题。"""
    try:
        subject_id = import_archive(ctx.db, payload)
    except ExportError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return {"subject_id": subject_id, "title": stack.subjects.get(subject_id).title}
