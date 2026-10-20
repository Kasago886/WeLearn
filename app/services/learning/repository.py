"""领域仓储（T2.1）。

约定：

- **所有方法自带事务**（``with db.session()``），调用方拿到的都是不可变 dataclass；
- **级联删除在仓储层显式完成**：SQLite 的外键级联依赖 ``PRAGMA foreign_keys=ON``（已开启），
  但显式按序删除更易读，也不依赖每个连接都正确设置了 PRAGMA；
- **JSON 列统一由本模块编解码**，服务层只见到 ``tuple[str, ...]``。
"""

import json
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta

from sqlalchemy import and_, case, delete, func, select

from app.contracts.schemas import (
    AssessmentStatus,
    MasteryLevel,
    NodeState,
    SubjectState,
    WeakPointReason,
)
from app.infra.storage.db import Database
from app.infra.storage.models import (
    AnswerRecordRow,
    AssessmentLevelRow,
    AssessmentRow,
    ChatRoundRow,
    CheckpointRow,
    ConversationRow,
    KnowledgePointRow,
    LearningPathRow,
    SessionStateRow,
    StudyNoteRow,
    SubjectRow,
    UsageRecordRow,
    WeakPointRow,
    utcnow,
)
from app.services.learning.entities import (
    ANSWER_SOURCE_TEACH,
    AnswerRecord,
    Assessment,
    Checkpoint,
    KnowledgePoint,
    LearningPath,
    ProfileLevel,
    ReviewDueStat,
    SessionSnapshot,
    StudyNote,
    Subject,
    SubjectProgress,
    SubjectSummary,
    WeakPoint,
)


def new_id() -> str:
    """生成实体 id。"""
    return str(uuid.uuid4())


def _dump_list(values: Iterable[str] | None) -> str | None:
    items = [str(value) for value in (values or []) if str(value).strip()]
    return json.dumps(items, ensure_ascii=False) if items else None


def _load_list(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(loaded, list):
        return ()
    return tuple(str(item) for item in loaded)


def _as_str_list(value: object) -> list[str]:
    """把"可能不是字符串列表"的输入收敛为 ``list[str]``（模型输出未必守规）。

    三条规则：

    - 标量字符串按**单元素列表**处理而不是丢弃：模型把列表写成单个字符串时，
      内容仍然有效，丢掉反而会静默损失要点；
    - ``None`` 元素**丢弃**（不能用 ``str(None)``，否则界面上会出现字面量 "None"）；
    - 其余元素转字符串，空串丢弃。
    """
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list | tuple):
        return []

    items: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def _as_mapping_list(value: object) -> Sequence[Mapping[str, object]]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, dict)]


def _to_subject(row: SubjectRow) -> Subject:
    return Subject(
        id=row.id,
        title=row.title,
        goal=row.goal,
        background=row.background,
        depth=row.depth,
        status=SubjectState(row.status),
        created_at=row.created_at,
        updated_at=row.updated_at,
        archived_at=row.archived_at,
    )


def _to_session(row: SessionStateRow) -> SessionSnapshot:
    return SessionSnapshot(
        subject_id=row.subject_id,
        state=SubjectState(row.state),
        current_node_id=row.current_node_id,
        pending_checkpoint_id=row.pending_checkpoint_id,
        pending_question=row.pending_question,
        node_attempts=row.node_attempts,
        taught_node_id=row.taught_node_id,
        awaiting_decision=bool(row.awaiting_decision),
        last_action=row.last_action,
        review_mode=row.review_mode,
        review_node_id=row.review_node_id,
        review_checkpoint_id=row.review_checkpoint_id,
        adjacent_review_for=row.adjacent_review_for,
        updated_at=row.updated_at,
    )


def _to_assessment(row: AssessmentRow) -> Assessment:
    return Assessment(
        id=row.id,
        subject_id=row.subject_id,
        status=AssessmentStatus(row.status),
        turn_count=row.turn_count,
        max_turns=row.max_turns,
        topics=_load_list(row.topics_json),
        summary=row.summary,
        suggested_start_node=row.suggested_start_node,
        skipped=row.skipped,
        completed_at=row.completed_at,
    )


def _to_checkpoint(row: CheckpointRow) -> Checkpoint:
    return Checkpoint(
        id=row.id,
        node_id=row.node_id,
        order_index=row.order_index,
        question=row.question,
        expected_points=_load_list(row.expected_points_json),
        reference_answer=row.reference_answer,
    )


def _to_node(row: KnowledgePointRow, checkpoints: Sequence[Checkpoint] = ()) -> KnowledgePoint:
    return KnowledgePoint(
        id=row.id,
        path_id=row.path_id,
        order_index=row.order_index,
        title=row.title,
        objective=row.objective,
        prerequisites=_load_list(row.prerequisites_json),
        key_points=_load_list(row.key_points_json),
        state=NodeState(row.state),
        attempts=row.attempts,
        is_quick_review=row.is_quick_review,
        mastered_at=row.mastered_at,
        last_reviewed_at=row.last_reviewed_at,
        review_box=row.review_box or 0,
        review_due_at=row.review_due_at,
        summary_note=row.summary_note,
        checkpoints=tuple(checkpoints),
    )


def _to_weak_point(row: WeakPointRow) -> WeakPoint:
    return WeakPoint(
        id=row.id,
        subject_id=row.subject_id,
        node_id=row.node_id,
        reason=WeakPointReason(row.reason),
        fail_count=row.fail_count,
        last_attempt_at=row.last_attempt_at,
        resolved_at=row.resolved_at,
    )


def _to_answer(row: AnswerRecordRow) -> AnswerRecord:
    return AnswerRecord(
        id=row.id,
        subject_id=row.subject_id,
        node_id=row.node_id,
        checkpoint_id=row.checkpoint_id,
        question=row.question,
        user_answer=row.user_answer,
        verdict=row.verdict,
        confidence=row.confidence,
        feedback=row.feedback,
        hit_points=_load_list(row.hit_points_json),
        missing_points=_load_list(row.missing_points_json),
        attempt_index=row.attempt_index,
        source=row.source or ANSWER_SOURCE_TEACH,
        created_at=row.created_at,
    )


class SubjectRepository:
    """主题的读写（FR-101~FR-106）。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(
        self,
        title: str,
        *,
        goal: str | None = None,
        background: str | None = None,
        depth: str | None = None,
        status: SubjectState = SubjectState.DRAFT,
    ) -> Subject:
        row = SubjectRow(
            id=new_id(),
            title=title,
            goal=goal,
            background=background,
            depth=depth,
            status=status.value,
        )
        with self._db.session() as session:
            session.add(row)
        return _to_subject(row)

    def get(self, subject_id: str) -> Subject | None:
        with self._db.session() as session:
            row = session.get(SubjectRow, subject_id)
            return _to_subject(row) if row is not None else None

    def list_all(self, *, include_archived: bool = False) -> list[Subject]:
        """列出主题。

        方法名不能叫 ``list``：那会在类作用域里遮蔽内置 ``list``，使同类的返回注解
        ``list[Subject]`` 解析到方法本身（mypy 会直接报 "not valid as a type"）。
        """
        statement = select(SubjectRow).order_by(SubjectRow.created_at.desc())
        if not include_archived:
            statement = statement.where(SubjectRow.archived_at.is_(None))
        with self._db.session() as session:
            return [_to_subject(row) for row in session.scalars(statement)]

    def update(
        self,
        subject_id: str,
        *,
        title: str | None = None,
        goal: str | None = None,
        background: str | None = None,
        depth: str | None = None,
    ) -> Subject | None:
        with self._db.session() as session:
            row = session.get(SubjectRow, subject_id)
            if row is None:
                return None
            if title is not None:
                row.title = title
            if goal is not None:
                row.goal = goal
            if background is not None:
                row.background = background
            if depth is not None:
                row.depth = depth
            session.add(row)
            session.flush()
            return _to_subject(row)

    def set_status(self, subject_id: str, status: SubjectState) -> None:
        with self._db.session() as session:
            row = session.get(SubjectRow, subject_id)
            if row is not None:
                row.status = status.value
                session.add(row)

    def archive(self, subject_id: str, *, archived: bool = True) -> bool:
        with self._db.session() as session:
            row = session.get(SubjectRow, subject_id)
            if row is None:
                return False
            row.archived_at = utcnow() if archived else None
            session.add(row)
            return True

    def delete(self, subject_id: str) -> bool:
        """删除主题并清理其全部下级数据（FR-104 的级联语义）。"""
        with self._db.session() as session:
            if session.get(SubjectRow, subject_id) is None:
                return False

            path_ids = list(
                session.scalars(select(LearningPathRow.id).where(LearningPathRow.subject_id == subject_id))
            )
            if path_ids:
                node_ids = list(
                    session.scalars(
                        select(KnowledgePointRow.id).where(KnowledgePointRow.path_id.in_(path_ids))
                    )
                )
                if node_ids:
                    session.execute(delete(CheckpointRow).where(CheckpointRow.node_id.in_(node_ids)))
                session.execute(delete(KnowledgePointRow).where(KnowledgePointRow.path_id.in_(path_ids)))

            assessment_ids = list(
                session.scalars(select(AssessmentRow.id).where(AssessmentRow.subject_id == subject_id))
            )
            if assessment_ids:
                session.execute(
                    delete(AssessmentLevelRow).where(AssessmentLevelRow.assessment_id.in_(assessment_ids))
                )

            conversation_ids = list(
                session.scalars(
                    select(ConversationRow.id).where(ConversationRow.subject_id == subject_id)
                )
            )
            if conversation_ids:
                round_ids = list(
                    session.scalars(
                        select(ChatRoundRow.id).where(ChatRoundRow.conversation_id.in_(conversation_ids))
                    )
                )
                if round_ids:
                    from app.infra.storage.models import MessageRow

                    session.execute(delete(MessageRow).where(MessageRow.round_id.in_(round_ids)))
                session.execute(delete(ChatRoundRow).where(ChatRoundRow.conversation_id.in_(conversation_ids)))

            session.execute(delete(ConversationRow).where(ConversationRow.subject_id == subject_id))
            session.execute(delete(AssessmentRow).where(AssessmentRow.subject_id == subject_id))
            session.execute(delete(LearningPathRow).where(LearningPathRow.subject_id == subject_id))
            session.execute(delete(SessionStateRow).where(SessionStateRow.subject_id == subject_id))
            session.execute(delete(AnswerRecordRow).where(AnswerRecordRow.subject_id == subject_id))
            session.execute(delete(WeakPointRow).where(WeakPointRow.subject_id == subject_id))
            session.execute(delete(StudyNoteRow).where(StudyNoteRow.subject_id == subject_id))
            session.execute(delete(UsageRecordRow).where(UsageRecordRow.subject_id == subject_id))
            session.execute(delete(SubjectRow).where(SubjectRow.id == subject_id))
            return True

    def progress(self, subject_id: str) -> SubjectProgress:
        """进度概览（FR-106）：节点状态分布、薄弱点数、累计 token。"""
        with self._db.session() as session:
            path_id = session.scalar(
                select(LearningPathRow.id)
                .where(LearningPathRow.subject_id == subject_id, LearningPathRow.is_active.is_(True))
                .limit(1)
            )

            state_counts: dict[str, int] = {}
            if path_id is not None:
                rows = session.execute(
                    select(KnowledgePointRow.state, func.count(KnowledgePointRow.id))
                    .where(KnowledgePointRow.path_id == path_id)
                    .group_by(KnowledgePointRow.state)
                ).all()
                state_counts = {str(state): int(count) for state, count in rows}

            weak_points = session.scalar(
                select(func.count(WeakPointRow.id)).where(
                    WeakPointRow.subject_id == subject_id,
                    WeakPointRow.resolved_at.is_(None),
                )
            )
            total_tokens = session.scalar(
                select(func.coalesce(func.sum(UsageRecordRow.total_tokens), 0)).where(
                    UsageRecordRow.subject_id == subject_id
                )
            )

        return SubjectProgress(
            total_nodes=sum(state_counts.values()),
            mastered_nodes=state_counts.get(NodeState.MASTERED.value, 0),
            partial_nodes=state_counts.get(NodeState.PARTIAL.value, 0),
            skipped_nodes=state_counts.get(NodeState.SKIPPED.value, 0),
            weak_points=int(weak_points or 0),
            total_tokens=int(total_tokens or 0),
        )

    def last_studied_at(self, subject_id: str) -> datetime | None:
        """最近一次学习动作的时间：取作答记录与用量记录中较晚者。"""
        with self._db.session() as session:
            by_answer = session.scalar(
                select(func.max(AnswerRecordRow.created_at)).where(AnswerRecordRow.subject_id == subject_id)
            )
            by_usage = session.scalar(
                select(func.max(UsageRecordRow.created_at)).where(UsageRecordRow.subject_id == subject_id)
            )

        candidates = [value for value in (by_answer, by_usage) if value is not None]
        return max(candidates) if candidates else None

    def review_stats(self, *, interval_days: int, now: datetime | None = None) -> dict[str, ReviewDueStat]:
        """各主题的待复习计数（FR-505），供主题列表一次取回。

        列表页每行都要这个数字，逐主题查就是 N+1；这里用一条分组查询算出两种情形：
        已排期且到期（``due``）、以及从未排期但最近触达已超阈值（``stale``）——
        后者覆盖"排期功能上线前就已掌握"的历史节点。

        只看活动路径：旧版本路径上的节点已经不是用户眼前的进度，把它们算进来会
        让列表页的提醒永远清不掉。
        """
        moment = now or utcnow()
        cutoff = moment - timedelta(days=max(1, int(interval_days)))
        anchor = func.coalesce(KnowledgePointRow.last_reviewed_at, KnowledgePointRow.mastered_at)
        due_condition = and_(
            KnowledgePointRow.review_due_at.is_not(None),
            KnowledgePointRow.review_due_at <= moment,
        )
        stale_condition = and_(
            KnowledgePointRow.review_due_at.is_(None),
            anchor.is_not(None),
            anchor < cutoff,
        )
        statement = (
            select(
                LearningPathRow.subject_id,
                func.count(case((due_condition, 1))).label("due_count"),
                func.count(case((stale_condition, 1))).label("stale_count"),
            )
            .join(KnowledgePointRow, KnowledgePointRow.path_id == LearningPathRow.id)
            .where(
                LearningPathRow.is_active.is_(True),
                KnowledgePointRow.state == NodeState.MASTERED.value,
            )
            .group_by(LearningPathRow.subject_id)
        )

        with self._db.session() as session:
            rows = session.execute(statement).all()
        return {
            subject_id: ReviewDueStat(due=int(due_count or 0), stale=int(stale_count or 0))
            for subject_id, due_count, stale_count in rows
        }

    def summaries(self, *, include_archived: bool = False) -> list[SubjectSummary]:
        return [
            SubjectSummary(
                subject=subject,
                progress=self.progress(subject.id),
                last_studied_at=self.last_studied_at(subject.id),
            )
            for subject in self.list_all(include_archived=include_archived)
        ]


class SessionStateRepository:
    """状态机持久化（FR-410）。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, subject_id: str, *, state: SubjectState = SubjectState.DRAFT) -> SessionSnapshot:
        row = SessionStateRow(subject_id=subject_id, state=state.value)
        with self._db.session() as session:
            session.add(row)
        return _to_session(row)

    def get(self, subject_id: str) -> SessionSnapshot | None:
        with self._db.session() as session:
            row = session.get(SessionStateRow, subject_id)
            return _to_session(row) if row is not None else None

    def ensure(self, subject_id: str) -> SessionSnapshot:
        """读取快照；不存在则创建（避免"主题建好了但没有状态行"的中间态）。"""
        existing = self.get(subject_id)
        return existing if existing is not None else self.create(subject_id)

    def update(self, subject_id: str, **fields: object) -> SessionSnapshot:
        """按字段更新；``None`` 是有效取值（表示清空该字段）。"""
        allowed = {
            "state",
            "current_node_id",
            "pending_checkpoint_id",
            "pending_question",
            "node_attempts",
            "taught_node_id",
            "awaiting_decision",
            "last_action",
            "review_mode",
            "review_node_id",
            "review_checkpoint_id",
            "adjacent_review_for",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不支持更新的字段: {sorted(unknown)}")

        with self._db.session() as session:
            row = session.get(SessionStateRow, subject_id)
            if row is None:
                raise KeyError(f"会话状态不存在: {subject_id}")
            for name, value in fields.items():
                setattr(row, name, value.value if isinstance(value, SubjectState) else value)
            session.add(row)
            session.flush()
            return _to_session(row)

    def clear_pending(self, subject_id: str) -> SessionSnapshot:
        """清空待回答问题（作答已受理或节点已切换时调用）。"""
        return self.update(
            subject_id,
            pending_checkpoint_id=None,
            pending_question=None,
        )


class AssessmentRepository:
    """测评与画像持久化（FR-201~FR-208）。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, subject_id: str, *, max_turns: int, topics: Sequence[str] = ()) -> Assessment:
        row = AssessmentRow(
            id=new_id(),
            subject_id=subject_id,
            status=AssessmentStatus.RUNNING.value,
            turn_count=0,
            max_turns=max_turns,
            topics_json=_dump_list(topics),
        )
        with self._db.session() as session:
            session.add(row)
        return _to_assessment(row)

    def get(self, assessment_id: str) -> Assessment | None:
        with self._db.session() as session:
            row = session.get(AssessmentRow, assessment_id)
            return _to_assessment(row) if row is not None else None

    def get_latest(self, subject_id: str) -> Assessment | None:
        """最近一次测评（无论是否结束）。"""
        with self._db.session() as session:
            row = session.scalar(
                select(AssessmentRow)
                .where(AssessmentRow.subject_id == subject_id)
                .order_by(AssessmentRow.created_at.desc())
                .limit(1)
            )
            return _to_assessment(row) if row is not None else None

    def get_running(self, subject_id: str) -> Assessment | None:
        with self._db.session() as session:
            row = session.scalar(
                select(AssessmentRow)
                .where(
                    AssessmentRow.subject_id == subject_id,
                    AssessmentRow.status == AssessmentStatus.RUNNING.value,
                )
                .order_by(AssessmentRow.created_at.desc())
                .limit(1)
            )
            return _to_assessment(row) if row is not None else None

    def increment_turn(self, assessment_id: str, *, topics: Sequence[str] | None = None) -> int:
        """轮次 +1 并返回新的轮次值（轮次上限判定依赖它）。"""
        with self._db.session() as session:
            row = session.get(AssessmentRow, assessment_id)
            if row is None:
                raise KeyError(f"测评不存在: {assessment_id}")
            row.turn_count += 1
            if topics is not None:
                row.topics_json = _dump_list(topics)
            session.add(row)
            session.flush()
            return row.turn_count

    def set_topics(self, assessment_id: str, topics: Sequence[str]) -> None:
        with self._db.session() as session:
            row = session.get(AssessmentRow, assessment_id)
            if row is not None:
                row.topics_json = _dump_list(topics)
                session.add(row)

    def finish(
        self,
        assessment_id: str,
        *,
        summary: str | None = None,
        suggested_start_node: str | None = None,
        skipped: bool = False,
    ) -> Assessment | None:
        with self._db.session() as session:
            row = session.get(AssessmentRow, assessment_id)
            if row is None:
                return None
            row.status = (AssessmentStatus.SKIPPED if skipped else AssessmentStatus.COMPLETED).value
            row.summary = summary
            row.suggested_start_node = suggested_start_node
            row.skipped = skipped
            row.completed_at = utcnow()
            session.add(row)
            session.flush()
            return _to_assessment(row)

    def save_levels(self, assessment_id: str, levels: Sequence[ProfileLevel]) -> None:
        """整体重写画像明细（重新生成时覆盖旧结果）。"""
        with self._db.session() as session:
            session.execute(
                delete(AssessmentLevelRow).where(AssessmentLevelRow.assessment_id == assessment_id)
            )
            for level in levels:
                session.add(
                    AssessmentLevelRow(
                        assessment_id=assessment_id,
                        topic=level.topic,
                        level=level.level.value,
                        evidence=level.evidence,
                        user_overridden=level.user_overridden,
                    )
                )

    def levels(self, assessment_id: str) -> list[ProfileLevel]:
        with self._db.session() as session:
            rows = session.scalars(
                select(AssessmentLevelRow)
                .where(AssessmentLevelRow.assessment_id == assessment_id)
                .order_by(AssessmentLevelRow.id)
            )
            return [
                ProfileLevel(
                    topic=row.topic,
                    level=MasteryLevel(row.level),
                    evidence=row.evidence,
                    user_overridden=row.user_overridden,
                )
                for row in rows
            ]

    def override_level(self, assessment_id: str, topic: str, level: MasteryLevel) -> bool:
        """手动修正某个子领域的水平（FR-207）。"""
        with self._db.session() as session:
            row = session.scalar(
                select(AssessmentLevelRow).where(
                    AssessmentLevelRow.assessment_id == assessment_id,
                    AssessmentLevelRow.topic == topic,
                )
            )
            if row is None:
                return False
            row.level = level.value
            row.user_overridden = True
            session.add(row)
            return True


class PathRepository:
    """路径与节点持久化（FR-301~FR-309）。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def create(
        self,
        subject_id: str,
        *,
        rationale: str,
        model: str | None,
        nodes: Sequence[dict[str, object]],
    ) -> LearningPath:
        """新建路径（并把旧路径置为非活动，保留版本历史，FR-307）。

        ``nodes`` 每项需含 ``title`` / ``objective`` / ``prerequisites`` / ``key_points``
        / ``checkpoints``（每项含 question / expected_points / reference_answer）。
        """
        path_id = new_id()
        row = LearningPathRow(
            id=path_id,
            subject_id=subject_id,
            version=1,
            rationale=rationale,
            model=model,
            is_active=True,
        )

        with self._db.session() as session:
            previous = list(
                session.scalars(
                    select(LearningPathRow).where(
                        LearningPathRow.subject_id == subject_id,
                        LearningPathRow.is_active.is_(True),
                    )
                )
            )
            for old in previous:
                old.is_active = False
                session.add(old)
            row.version = (max((old.version for old in previous), default=0)) + 1
            session.add(row)
            # 必须先 flush 出 path 行，否则随后插入的 node 会撞外键约束
            session.flush()

            for index, node in enumerate(nodes):
                self._insert_node(session, path_id, index, node)

        created = self.get(path_id)
        assert created is not None
        return created

    def get(self, path_id: str) -> LearningPath | None:
        with self._db.session() as session:
            row = session.get(LearningPathRow, path_id)
            if row is None:
                return None
            nodes = self._load_nodes(session, path_id)

        return LearningPath(
            id=row.id,
            subject_id=row.subject_id,
            version=row.version,
            rationale=row.rationale,
            model=row.model,
            is_active=row.is_active,
            created_at=row.created_at,
            nodes=tuple(nodes),
        )

    def get_active(self, subject_id: str) -> LearningPath | None:
        with self._db.session() as session:
            path_id = session.scalar(
                select(LearningPathRow.id)
                .where(LearningPathRow.subject_id == subject_id, LearningPathRow.is_active.is_(True))
                .order_by(LearningPathRow.version.desc())
                .limit(1)
            )
        return self.get(path_id) if path_id is not None else None

    def list_versions(self, subject_id: str) -> list[tuple[str, int, bool, datetime]]:
        with self._db.session() as session:
            rows = session.scalars(
                select(LearningPathRow)
                .where(LearningPathRow.subject_id == subject_id)
                .order_by(LearningPathRow.version.desc())
            )
            return [(row.id, row.version, row.is_active, row.created_at) for row in rows]

    @staticmethod
    def _load_nodes(session: object, path_id: str) -> list[KnowledgePoint]:
        """一次查出全部节点与检查点，避免 N+1。"""
        node_rows = list(
            session.scalars(  # type: ignore[attr-defined]
                select(KnowledgePointRow)
                .where(KnowledgePointRow.path_id == path_id)
                .order_by(KnowledgePointRow.order_index)
            )
        )
        if not node_rows:
            return []

        node_ids = [row.id for row in node_rows]
        checkpoint_rows = list(
            session.scalars(  # type: ignore[attr-defined]
                select(CheckpointRow)
                .where(CheckpointRow.node_id.in_(node_ids))
                .order_by(CheckpointRow.node_id, CheckpointRow.order_index)
            )
        )

        grouped: dict[str, list[Checkpoint]] = {node_id: [] for node_id in node_ids}
        for row in checkpoint_rows:
            grouped[row.node_id].append(_to_checkpoint(row))

        return [_to_node(row, grouped[row.id]) for row in node_rows]

    def node(self, node_id: str) -> KnowledgePoint | None:
        with self._db.session() as session:
            return self._node_with_checkpoints(session, node_id)

    @staticmethod
    def _node_with_checkpoints(session: object, node_id: str) -> KnowledgePoint | None:
        """读取节点**并带上检查点**。

        ``next_node`` / ``first_node`` 也走这里：教学流程必须先拿到节点的检查点才能提问，
        返回"没有检查点的节点"会让调用方在运行时才发现问题。
        """
        row = session.get(KnowledgePointRow, node_id)  # type: ignore[attr-defined]
        if row is None:
            return None
        checkpoints = [
            _to_checkpoint(item)
            for item in session.scalars(  # type: ignore[attr-defined]
                select(CheckpointRow)
                .where(CheckpointRow.node_id == node_id)
                .order_by(CheckpointRow.order_index)
            )
        ]
        return _to_node(row, checkpoints)

    def checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        with self._db.session() as session:
            row = session.get(CheckpointRow, checkpoint_id)
            return _to_checkpoint(row) if row is not None else None

    def first_checkpoint(self, node_id: str) -> Checkpoint | None:
        with self._db.session() as session:
            row = session.scalar(
                select(CheckpointRow)
                .where(CheckpointRow.node_id == node_id)
                .order_by(CheckpointRow.order_index)
                .limit(1)
            )
            return _to_checkpoint(row) if row is not None else None

    def next_checkpoint(self, node_id: str, after_order_index: int) -> Checkpoint | None:
        with self._db.session() as session:
            row = session.scalar(
                select(CheckpointRow)
                .where(CheckpointRow.node_id == node_id, CheckpointRow.order_index > after_order_index)
                .order_by(CheckpointRow.order_index)
                .limit(1)
            )
            return _to_checkpoint(row) if row is not None else None

    def update_node(self, node_id: str, **fields: object) -> KnowledgePoint | None:
        allowed = {
            "state",
            "attempts",
            "mastered_at",
            "last_reviewed_at",
            "review_box",
            "review_due_at",
            "summary_note",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不支持更新的字段: {sorted(unknown)}")

        with self._db.session() as session:
            row = session.get(KnowledgePointRow, node_id)
            if row is None:
                return None
            for name, value in fields.items():
                setattr(row, name, value.value if isinstance(value, NodeState) else value)
            session.add(row)
            session.flush()
            return _to_node(row)

    def increment_attempts(self, node_id: str) -> int:
        with self._db.session() as session:
            row = session.get(KnowledgePointRow, node_id)
            if row is None:
                raise KeyError(f"节点不存在: {node_id}")
            row.attempts += 1
            session.add(row)
            session.flush()
            return row.attempts

    def activate(self, path_id: str) -> bool:
        with self._db.session() as session:
            row = session.get(LearningPathRow, path_id)
            if row is None:
                return False
            others = list(
                session.scalars(
                    select(LearningPathRow).where(
                        LearningPathRow.subject_id == row.subject_id,
                        LearningPathRow.id != path_id,
                    )
                )
            )
            for other in others:
                other.is_active = False
                session.add(other)
            row.is_active = True
            session.add(row)
            return True

    def next_node(self, path_id: str, *, after_order_index: int) -> KnowledgePoint | None:
        """下一个待学节点（含检查点）；已全部走完时返回 None（用于判定"学习完成"）。"""
        with self._db.session() as session:
            node_id = session.scalar(
                select(KnowledgePointRow.id)
                .where(KnowledgePointRow.path_id == path_id, KnowledgePointRow.order_index > after_order_index)
                .order_by(KnowledgePointRow.order_index)
                .limit(1)
            )
            return self._node_with_checkpoints(session, node_id) if node_id is not None else None

    def first_node(self, path_id: str) -> KnowledgePoint | None:
        return self.next_node(path_id, after_order_index=-1)

    # ------------------------------------------------------------------ #
    # 编辑（FR-304、FR-305）
    # ------------------------------------------------------------------ #

    def replace_nodes(self, path_id: str, nodes: Sequence[dict[str, object]]) -> LearningPath | None:
        """用给定的节点列表**整体替换**路径内容（顺序即 ``order_index``）。

        编辑操作统一走"读出 → 在内存里改 → 整体写回"，而不是对单行做增量 UPDATE：
        排序、插入、删除都要重排 ``order_index``，增量更新极易留下重复序号，
        而重复序号会让 ``next_node`` 随机挑到一个节点。

        已有的 ``answer_record`` 不删——它是学习者自己的作答历史，
        节点被删掉后仍然应该能在档案里查到。
        """
        with self._db.session() as session:
            row = session.get(LearningPathRow, path_id)
            if row is None:
                return None

            old_node_ids = list(
                session.scalars(select(KnowledgePointRow.id).where(KnowledgePointRow.path_id == path_id))
            )
            if old_node_ids:
                # checkpoint 与 node 都是 ondelete CASCADE，但显式删除更直白
                session.execute(delete(CheckpointRow).where(CheckpointRow.node_id.in_(old_node_ids)))
                session.execute(delete(KnowledgePointRow).where(KnowledgePointRow.id.in_(old_node_ids)))
            session.flush()

            for index, node in enumerate(nodes):
                self._insert_node(session, path_id, index, node)

        return self.get(path_id)

    def update_rationale(self, path_id: str, rationale: str) -> bool:
        with self._db.session() as session:
            row = session.get(LearningPathRow, path_id)
            if row is None:
                return False
            row.rationale = rationale
            session.add(row)
            return True

    def replace_node(self, node_id: str, payload: dict[str, object]) -> KnowledgePoint | None:
        """替换单个节点的内容，**保留它的 id 与学习状态**。

        "重生成单个节点"必须保留 id：节点的 ``answer_record``、笔记、薄弱点都挂在 id 上，
        换 id 等于把这些记录变成孤儿。
        """
        with self._db.session() as session:
            row = session.get(KnowledgePointRow, node_id)
            if row is None:
                return None

            row.title = str(payload.get("title", row.title))
            row.objective = str(payload.get("objective", row.objective))
            if "prerequisites" in payload:
                row.prerequisites_json = _dump_list(_as_str_list(payload.get("prerequisites")))
            if "key_points" in payload:
                row.key_points_json = _dump_list(_as_str_list(payload.get("key_points")))
            if "is_quick_review" in payload:
                row.is_quick_review = bool(payload.get("is_quick_review"))
            session.add(row)

            if "checkpoints" in payload:
                session.execute(delete(CheckpointRow).where(CheckpointRow.node_id == node_id))
                session.flush()
                self._insert_checkpoints(session, node_id, _as_mapping_list(payload.get("checkpoints")))

            session.flush()
            return self._node_with_checkpoints(session, node_id)

    @staticmethod
    def _insert_node(session: object, path_id: str, index: int, node: Mapping[str, object]) -> str:
        """插入一个节点及其检查点，返回节点 id。

        **优先复用 ``node["id"]``**：编辑路径时整表重写，若给所有节点换新 id，
        挂在旧 id 上的作答记录、笔记、薄弱点就全成了孤儿。只有新增节点才生成新 id。
        """
        raw_state = node.get("state")
        # 编辑已有路径时要把节点的学习状态带过去（重排顺序不该把"已掌握"重置成"未开始"）
        state = raw_state.value if isinstance(raw_state, NodeState) else str(raw_state or NodeState.NOT_STARTED.value)
        raw_id = node.get("id")
        node_id = str(raw_id) if isinstance(raw_id, str) and raw_id else new_id()

        node_row = KnowledgePointRow(
            id=node_id,
            path_id=path_id,
            order_index=index,
            title=str(node.get("title", "")),
            objective=str(node.get("objective", "")),
            prerequisites_json=_dump_list(_as_str_list(node.get("prerequisites"))),
            key_points_json=_dump_list(_as_str_list(node.get("key_points"))),
            is_quick_review=bool(node.get("is_quick_review", False)),
            state=state,
        )
        session.add(node_row)  # type: ignore[attr-defined]
        # checkpoint 引用 node，必须先落盘 node 行
        session.flush()  # type: ignore[attr-defined]
        PathRepository._insert_checkpoints(session, node_row.id, _as_mapping_list(node.get("checkpoints")))
        return node_row.id

    @staticmethod
    def _insert_checkpoints(session: object, node_id: str, checkpoints: Sequence[Mapping[str, object]]) -> None:
        for order, checkpoint in enumerate(checkpoints):
            raw_id = checkpoint.get("id")
            checkpoint_id = str(raw_id) if isinstance(raw_id, str) and raw_id else new_id()
            session.add(  # type: ignore[attr-defined]
                CheckpointRow(
                    id=checkpoint_id,
                    node_id=node_id,
                    order_index=order,
                    question=str(checkpoint.get("question", "")),
                    expected_points_json=_dump_list(_as_str_list(checkpoint.get("expected_points"))),
                    reference_answer=str(checkpoint.get("reference_answer", "")),
                )
            )


class AnswerRepository:
    """作答记录（FR-404、FR-802）。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(
        self,
        *,
        subject_id: str,
        node_id: str,
        checkpoint_id: str | None,
        question: str,
        user_answer: str,
        verdict: str,
        confidence: float | None,
        feedback: str,
        hit_points: Sequence[str] = (),
        missing_points: Sequence[str] = (),
        attempt_index: int = 1,
        source: str = ANSWER_SOURCE_TEACH,
    ) -> AnswerRecord:
        row = AnswerRecordRow(
            subject_id=subject_id,
            node_id=node_id,
            checkpoint_id=checkpoint_id,
            question=question,
            user_answer=user_answer,
            verdict=verdict,
            confidence=confidence,
            feedback=feedback,
            hit_points_json=_dump_list(hit_points),
            missing_points_json=_dump_list(missing_points),
            attempt_index=attempt_index,
            source=source,
        )
        with self._db.session() as session:
            session.add(row)
            session.flush()
            return _to_answer(row)

    def list_for_node(self, node_id: str) -> list[AnswerRecord]:
        with self._db.session() as session:
            rows = session.scalars(
                select(AnswerRecordRow)
                .where(AnswerRecordRow.node_id == node_id)
                .order_by(AnswerRecordRow.id)
            )
            return [_to_answer(row) for row in rows]

    def list_for_subject(self, subject_id: str, *, limit: int = 200) -> list[AnswerRecord]:
        with self._db.session() as session:
            rows = session.scalars(
                select(AnswerRecordRow)
                .where(AnswerRecordRow.subject_id == subject_id)
                .order_by(AnswerRecordRow.id.desc())
                .limit(limit)
            )
            return [_to_answer(row) for row in rows]

    def count_for_node(self, node_id: str) -> int:
        with self._db.session() as session:
            return int(
                session.scalar(
                    select(func.count(AnswerRecordRow.id)).where(AnswerRecordRow.node_id == node_id)
                )
                or 0
            )


class WeakPointRepository:
    """薄弱点（FR-502~FR-504）。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def record(self, subject_id: str, node_id: str, reason: WeakPointReason) -> WeakPoint:
        """记录薄弱点；同一节点已有未解决记录时累加失败次数并更新成因。"""
        with self._db.session() as session:
            row = session.scalar(
                select(WeakPointRow).where(
                    WeakPointRow.subject_id == subject_id,
                    WeakPointRow.node_id == node_id,
                    WeakPointRow.resolved_at.is_(None),
                )
            )
            if row is None:
                row = WeakPointRow(
                    subject_id=subject_id,
                    node_id=node_id,
                    reason=reason.value,
                    fail_count=1,
                )
            else:
                row.fail_count += 1
                row.reason = reason.value
                row.last_attempt_at = utcnow()
            session.add(row)
            session.flush()
            return _to_weak_point(row)

    def list_open(self, subject_id: str) -> list[WeakPoint]:
        with self._db.session() as session:
            rows = session.scalars(
                select(WeakPointRow)
                .where(WeakPointRow.subject_id == subject_id, WeakPointRow.resolved_at.is_(None))
                .order_by(WeakPointRow.last_attempt_at.desc())
            )
            return [_to_weak_point(row) for row in rows]

    def resolve(self, subject_id: str, node_id: str) -> bool:
        """复习通过后关闭薄弱点（FR-503）。"""
        with self._db.session() as session:
            rows = list(
                session.scalars(
                    select(WeakPointRow).where(
                        WeakPointRow.subject_id == subject_id,
                        WeakPointRow.node_id == node_id,
                        WeakPointRow.resolved_at.is_(None),
                    )
                )
            )
            for row in rows:
                row.resolved_at = utcnow()
                session.add(row)
            return bool(rows)

    def count_open(self, subject_id: str) -> int:
        with self._db.session() as session:
            return int(
                session.scalar(
                    select(func.count(WeakPointRow.id)).where(
                        WeakPointRow.subject_id == subject_id,
                        WeakPointRow.resolved_at.is_(None),
                    )
                )
                or 0
            )


class NoteRepository:
    """笔记（FR-801、FR-415）。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert(
        self,
        subject_id: str,
        node_id: str | None,
        content: str,
        *,
        source: str = "auto",
    ) -> StudyNote:
        with self._db.session() as session:
            row = session.scalar(
                select(StudyNoteRow).where(
                    StudyNoteRow.subject_id == subject_id,
                    StudyNoteRow.node_id == node_id,
                )
            )
            if row is None:
                row = StudyNoteRow(
                    id=new_id(),
                    subject_id=subject_id,
                    node_id=node_id,
                    content=content,
                    source=source,
                )
            else:
                row.content = content
                if source:
                    row.source = source
            session.add(row)
            session.flush()
            return StudyNote(
                id=row.id,
                subject_id=row.subject_id,
                node_id=row.node_id,
                content=row.content,
                source=row.source,
                updated_at=row.updated_at,
            )

    def get(self, subject_id: str, node_id: str | None) -> StudyNote | None:
        with self._db.session() as session:
            row = session.scalar(
                select(StudyNoteRow).where(
                    StudyNoteRow.subject_id == subject_id,
                    StudyNoteRow.node_id == node_id,
                )
            )
            if row is None:
                return None
            return StudyNote(
                id=row.id,
                subject_id=row.subject_id,
                node_id=row.node_id,
                content=row.content,
                source=row.source,
                updated_at=row.updated_at,
            )

    def list_for_subject(self, subject_id: str) -> list[StudyNote]:
        with self._db.session() as session:
            rows = session.scalars(
                select(StudyNoteRow)
                .where(StudyNoteRow.subject_id == subject_id)
                .order_by(StudyNoteRow.updated_at.desc())
            )
            return [
                StudyNote(
                    id=row.id,
                    subject_id=row.subject_id,
                    node_id=row.node_id,
                    content=row.content,
                    source=row.source,
                    updated_at=row.updated_at,
                )
                for row in rows
            ]


class DomainRepositories:
    """把领域仓储聚合在一起，便于注入与测试替身。"""

    def __init__(self, db: Database) -> None:
        self.subjects = SubjectRepository(db)
        self.sessions = SessionStateRepository(db)
        self.assessments = AssessmentRepository(db)
        self.paths = PathRepository(db)
        self.answers = AnswerRepository(db)
        self.weak_points = WeakPointRepository(db)
        self.notes = NoteRepository(db)
