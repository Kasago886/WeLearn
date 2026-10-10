"""主题与学习空间服务（T2.3 前置，FR-101~FR-107）。

职责：输入校验、主题生命周期、进度概览。**不涉及 LLM**——主题的创建与修改都是纯本地操作，
把它们和 AI 流程混在一起会让"没配 API Key 就不能管理已有主题"这种荒谬行为出现。
"""

import re

from app.contracts.schemas import SubjectState
from app.infra.logging.queue_logger import CHANNEL_APP, get_logger
from app.services.learning.entities import Subject, SubjectProgress, SubjectSummary
from app.services.learning.repository import DomainRepositories

TITLE_MIN_LENGTH = 2
TITLE_MAX_LENGTH = 100
GOAL_MAX_LENGTH = 500
BACKGROUND_MAX_LENGTH = 2000

_MEANINGFUL_PATTERN = re.compile(r"[A-Za-z\u4e00-\u9fff]")


class SubjectValidationError(ValueError):
    """主题输入不合法（FR-102）。消息直接面向用户。"""


def normalize_title(raw: str | None) -> str:
    """校验并规范化概念名称（FR-101、FR-102）。"""
    title = (raw or "").strip()

    if not title:
        raise SubjectValidationError("请填写要学习的概念名称。")
    if len(title) < TITLE_MIN_LENGTH:
        raise SubjectValidationError(f"概念名称至少需要 {TITLE_MIN_LENGTH} 个字符。")
    if len(title) > TITLE_MAX_LENGTH:
        raise SubjectValidationError(f"概念名称最多 {TITLE_MAX_LENGTH} 个字符，请精简后再试。")
    if not _MEANINGFUL_PATTERN.search(title):
        # 纯符号/纯数字大概率是误输入；不做联网校验（FR-102）
        raise SubjectValidationError("这个概念名称看起来不像是可以学习的内容，请换成具体概念名（例如「贝叶斯定理」）。")

    return title


def _normalize_optional(raw: str | None, *, field_label: str, limit: int) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    if len(text) > limit:
        raise SubjectValidationError(f"{field_label}最多 {limit} 个字符。")
    return text


class SubjectService:
    """主题生命周期与进度。"""

    def __init__(self, repos: DomainRepositories) -> None:
        self._repos = repos
        self._log = get_logger(CHANNEL_APP)

    # ------------------------------------------------------------------ #
    # 创建与修改
    # ------------------------------------------------------------------ #

    def create(
        self,
        title: str,
        *,
        goal: str | None = None,
        background: str | None = None,
        depth: str | None = None,
    ) -> Subject:
        """创建学习主题并初始化状态机为 ``draft``。"""
        normalized = normalize_title(title)
        subject = self._repos.subjects.create(
            normalized,
            goal=_normalize_optional(goal, field_label="学习目标", limit=GOAL_MAX_LENGTH),
            background=_normalize_optional(background, field_label="已有背景", limit=BACKGROUND_MAX_LENGTH),
            depth=_normalize_optional(depth, field_label="期望深度", limit=32),
        )
        # 主题与状态必须同时存在，否则后续任何流程都要先判空
        self._repos.sessions.ensure(subject.id)
        self._log.info(f"创建学习主题: {subject.title}（id={subject.id}）")
        return subject

    def update(
        self,
        subject_id: str,
        *,
        title: str | None = None,
        goal: str | None = None,
        background: str | None = None,
        depth: str | None = None,
    ) -> Subject:
        """重命名 / 修改目标（FR-104）。"""
        updates: dict[str, str | None] = {}
        if title is not None:
            updates["title"] = normalize_title(title)
        if goal is not None:
            updates["goal"] = _normalize_optional(goal, field_label="学习目标", limit=GOAL_MAX_LENGTH)
        if background is not None:
            updates["background"] = _normalize_optional(
                background, field_label="已有背景", limit=BACKGROUND_MAX_LENGTH
            )
        if depth is not None:
            updates["depth"] = _normalize_optional(depth, field_label="期望深度", limit=32)

        updated = self._repos.subjects.update(subject_id, **updates)
        if updated is None:
            raise KeyError(f"主题不存在: {subject_id}")
        return updated

    def archive(self, subject_id: str, *, archived: bool = True) -> Subject:
        if not self._repos.subjects.archive(subject_id, archived=archived):
            raise KeyError(f"主题不存在: {subject_id}")
        subject = self._repos.subjects.get(subject_id)
        assert subject is not None
        self._log.info(f"{'归档' if archived else '取消归档'}主题: {subject.title}")
        return subject

    def delete(self, subject_id: str) -> None:
        """删除主题及其全部下级数据（FR-104）。二次确认由界面负责。"""
        subject = self._repos.subjects.get(subject_id)
        if subject is None:
            raise KeyError(f"主题不存在: {subject_id}")

        self._repos.subjects.delete(subject_id)
        self._log.info(f"删除学习主题及其全部数据: {subject.title}（id={subject_id}）")

    # ------------------------------------------------------------------ #
    # 读取
    # ------------------------------------------------------------------ #

    def get(self, subject_id: str) -> Subject:
        subject = self._repos.subjects.get(subject_id)
        if subject is None:
            raise KeyError(f"主题不存在: {subject_id}")
        return subject

    def list_summaries(self, *, include_archived: bool = False) -> list[SubjectSummary]:
        """列表页数据（FR-103、FR-106）。"""
        return self._repos.subjects.summaries(include_archived=include_archived)

    def progress(self, subject_id: str) -> SubjectProgress:
        self.get(subject_id)
        return self._repos.subjects.progress(subject_id)

    def has_same_title(self, title: str, *, exclude_id: str | None = None) -> bool:
        """是否存在同名主题（FR-107：提示但允许创建）。"""
        try:
            normalized = normalize_title(title)
        except SubjectValidationError:
            return False
        return any(
            subject.title == normalized and subject.id != exclude_id
            for subject in self._repos.subjects.list_all(include_archived=True)
        )

    def status(self, subject_id: str) -> SubjectState:
        return self.get(subject_id).status

    def mark_teaching(self, subject_id: str) -> None:
        """把主题置为教学态（状态机的状态同步由 SessionStateService 负责，这里只兜底）。"""
        self._repos.subjects.set_status(subject_id, SubjectState.TEACHING)
