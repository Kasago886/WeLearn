"""薄弱点与复习（T3.3/T4.1，FR-502~FR-507）。

设计要点

- **复习会话有自己的记忆容器**（``ConversationKind.REVIEW``）：FR-602 要求测评/教学/复习
  三类记忆相互隔离。复习时把检查点问答混进教学记忆，会让教学上下文里出现一段
  "来自另一个节点"的问答，后面的讲解就会被带偏。
- **提问直接复用节点已有的检查点**，不为复习再生成新问题（FR-503 的"复用提问+判定机制"）。
  这样既省一次模型调用，又保证判定有可依据的 ``expected_points``——没有要点就无法判定。
- **通过才升级为已掌握并解除薄弱点**（FR-503）；不通过则累加失败次数并给出更基础的
  建议（FR-504），节点状态保持不变——"复习没通过"不该把原本的 partial 变成别的状态。
- **复习结果同时决定下一次复习时间**（FR-506）：通过升档、不通过回第一档（阶梯见
  ``schedule``）。只记录对错而不排下一次，"已掌握"的内容就再也不会被回顾，而遗忘
  恰恰发生在掌握之后。
"""

from dataclasses import dataclass, replace
from datetime import datetime

from app.contracts.schemas import ConversationKind, JudgementResult, NodeState, WeakPointReason
from app.infra.logging.queue_logger import CHANNEL_APP, get_logger
from app.infra.storage.models import utcnow
from app.services.learning.entities import (
    ANSWER_SOURCE_REVIEW,
    REVIEW_MODE_WEAK,
    AnswerRecord,
    Checkpoint,
    KnowledgePoint,
    Subject,
    WeakPoint,
)
from app.services.learning.planning import PlanningService
from app.services.learning.repository import DomainRepositories
from app.services.learning.schedule import (
    interval_days,
    is_due,
    schedule_after_fail,
    schedule_after_mastery,
    schedule_after_pass,
)
from app.services.learning.session import ConversationSession, ConversationSessionFactory
from app.services.learning.state import SessionStateService
from app.services.llm.agent import DeltaCallback
from app.services.llm.judge import JudgeInput
from app.services.llm.memory import user_message
from app.services.llm.structured import StructuredOutputError

SUGGESTION_HEAD = "建议回到教学页把这个节点再过一遍"


class ReviewError(RuntimeError):
    """复习的前置条件不满足，消息直接面向用户。"""


@dataclass(frozen=True)
class WeakPointView:
    """薄弱点面板的一项（FR-502）。"""

    weak_point: WeakPoint
    node: KnowledgePoint | None
    node_index: int
    node_total: int

    @property
    def node_title(self) -> str:
        return self.node.title if self.node is not None else "（节点已不存在）"


@dataclass(frozen=True)
class DueReviewView:
    """到期复习项（FR-506）：一个"现在就该回顾"的已掌握节点。"""

    node: KnowledgePoint
    node_index: int
    node_total: int
    interval_days: int
    """当前箱号对应的间隔天数，界面用它说明"这次隔了几天"。"""

    @property
    def node_title(self) -> str:
        return self.node.title

    @property
    def due_at(self) -> datetime | None:
        return self.node.review_due_at


@dataclass(frozen=True)
class ReviewOverview:
    """复习概览（FR-502、FR-506、FR-804）。"""

    items: tuple[WeakPointView, ...]
    node_total: int
    mastered_count: int
    due: tuple[DueReviewView, ...] = ()

    @property
    def open_count(self) -> int:
        return len(self.items)

    @property
    def due_count(self) -> int:
        return len(self.due)


@dataclass
class ReviewTurn:
    """一次复习交互的产出，供 API 层转成 SSE。"""

    text: str = ""
    phase: str = ""
    mode: str = REVIEW_MODE_WEAK
    node: KnowledgePoint | None = None
    checkpoint: Checkpoint | None = None
    verdict: str | None = None
    confidence: float = 0.0
    feedback: str = ""
    hit_points: tuple[str, ...] = ()
    missing_points: tuple[str, ...] = ()
    # 通过：升级为已掌握并解除薄弱点
    promoted: bool = False
    resolved: bool = False
    # 不通过：薄弱点仍在，这是累计失败次数
    fail_count: int = 0
    suggestion: str = ""
    node_index: int = 0
    node_total: int = 0
    error_message: str = ""
    error_category: str | None = None
    retryable: bool = False

    @property
    def ok(self) -> bool:
        return not self.error_message


class ReviewService:
    """薄弱点复习（FR-502~FR-504、FR-507）。"""

    def __init__(
        self,
        repos: DomainRepositories,
        state: SessionStateService,
        sessions: ConversationSessionFactory,
        planning: PlanningService,
    ) -> None:
        self._repos = repos
        self._state = state
        self._sessions = sessions
        self._planning = planning
        self._log = get_logger(CHANNEL_APP)

    # ------------------------------------------------------------------ #
    # 概览与清单
    # ------------------------------------------------------------------ #

    def overview(self, subject_id: str, *, now: datetime | None = None) -> ReviewOverview:
        """薄弱点清单 + 进度 + 到期复习（FR-502、FR-506、FR-804）。

        **懒补齐为什么放在读取路径上**：节点变成"已掌握"的入口不止一个——教学判定
        通过、复习通过、路径重生成都会产生 mastered 节点，逐个入口挂钩既容易漏
        （新加一个入口就少一处），也会把排期逻辑摊到三个模块里。而补齐只在
        ``review_due_at`` 为空时发生：第一次读取写入初始排期，之后读多少次都不再
        改动，因此放在读取路径上不会产生副作用累积，也顺带覆盖了功能上线前的
        历史数据。
        """
        self._require_subject(subject_id)
        moment = now or utcnow()
        path = self._planning.current_path(subject_id)
        ordered = path.ordered_nodes() if path is not None else ()
        ordered = self._backfill_schedules(ordered, moment)
        index_of = {node.id: position for position, node in enumerate(ordered, start=1)}
        mastered = sum(1 for node in ordered if node.state is NodeState.MASTERED)

        items = tuple(
            WeakPointView(
                weak_point=point,
                node=next((node for node in ordered if node.id == point.node_id), None),
                node_index=index_of.get(point.node_id, 0),
                node_total=len(ordered),
            )
            for point in self._repos.weak_points.list_open(subject_id)
        )
        due = tuple(
            sorted(
                (
                    DueReviewView(
                        node=node,
                        node_index=index_of.get(node.id, 0),
                        node_total=len(ordered),
                        interval_days=interval_days(node.review_box),
                    )
                    for node in ordered
                    if node.state is NodeState.MASTERED and is_due(node.review_due_at, moment)
                ),
                # 拖得最久的排在最前：用户最可能已经忘了它
                key=lambda view: view.node.review_due_at or moment,
            )
        )
        return ReviewOverview(
            items=items,
            node_total=len(ordered),
            mastered_count=mastered,
            due=due,
        )

    def history(self, subject_id: str, node_id: str) -> list[AnswerRecord]:
        """某节点的复习记录（FR-507）。"""
        self._require_subject(subject_id)
        return [
            record
            for record in self._repos.answers.list_for_node(node_id)
            if record.source == ANSWER_SOURCE_REVIEW
        ]

    def current(self, subject_id: str) -> ReviewTurn | None:
        """进行中的复习（刷新恢复）。没有在复习时返回 ``None``。"""
        snapshot = self._state.snapshot(subject_id)
        review_node_id = snapshot.review_node_id
        if not snapshot.has_review_question or not review_node_id:
            return None
        node = self._repos.paths.node(review_node_id)
        checkpoint = (
            self._repos.paths.checkpoint(snapshot.review_checkpoint_id)
            if snapshot.review_checkpoint_id
            else None
        )
        if node is None or checkpoint is None:
            # 节点/检查点已被删掉（路径被重新生成过）：静默结束这次复习，
            # 否则用户会卡在一个永远无法作答的问题上
            self._state.end_review(subject_id)
            return None
        return ReviewTurn(
            text=checkpoint.question,
            phase="question",
            mode=snapshot.review_mode or REVIEW_MODE_WEAK,
            node=node,
            checkpoint=checkpoint,
            node_index=self._node_index(subject_id, node.id),
            node_total=self._node_total(subject_id),
        )

    # ------------------------------------------------------------------ #
    # 复习会话
    # ------------------------------------------------------------------ #

    def start(self, subject_id: str, node_id: str) -> ReviewTurn:
        """为某个节点开一次复习：挑一个检查点提问（不调用模型）。"""
        self._require_subject(subject_id)
        node = self._require_node(subject_id, node_id)
        checkpoints = list(node.checkpoints)
        if not checkpoints:
            raise ReviewError("这个节点没有可用来复习的检查点，请先补一个。")

        # 同一个节点复习多次时轮着问，避免每次都问同一句（用户会直接背答案）
        attempts = max(1, self._repos.answers.count_for_node(node.id))
        checkpoint = checkpoints[(attempts - 1) % len(checkpoints)]
        self._state.begin_review(
            subject_id,
            mode=REVIEW_MODE_WEAK,
            node_id=node.id,
            checkpoint_id=checkpoint.id,
        )
        self._log.info(f"开始复习 subject={subject_id} node={node.id} checkpoint={checkpoint.id}")
        return ReviewTurn(
            text=self._question_text(node, checkpoint),
            phase="question",
            node=node,
            checkpoint=checkpoint,
            node_index=self._node_index(subject_id, node.id),
            node_total=self._node_total(subject_id),
        )

    async def answer(
        self,
        subject_id: str,
        answer: str,
        *,
        on_delta: DeltaCallback | None = None,
    ) -> ReviewTurn:
        """提交复习作答：通过则升级为已掌握并解除薄弱点（FR-503）。"""
        text = (answer or "").strip()
        if not text:
            return ReviewTurn(
                error_message="请先写下你的回答。",
                error_category="validation",
            )

        snapshot = self._state.snapshot(subject_id)
        review_node_id = snapshot.review_node_id
        if not snapshot.has_review_question or not review_node_id:
            raise ReviewError("当前没有待回答的复习问题，请从薄弱点清单里选择一项开始复习。")

        node = self._require_node(subject_id, review_node_id)
        checkpoint = (
            self._repos.paths.checkpoint(snapshot.review_checkpoint_id)
            if snapshot.review_checkpoint_id
            else None
        )
        if checkpoint is None:
            self._state.end_review(subject_id)
            raise ReviewError("这次复习的检查点已失效，请重新开始复习。")

        session = self._sessions.open(subject_id, ConversationKind.REVIEW)
        session.memory.add_round().add(user_message(text))
        session.persist()

        judgement, failure = await self._judge(node, checkpoint, text, session)
        if failure is not None:
            # 判定不可用属于技术故障：不写记录、不改状态，用户可直接重试
            return ReviewTurn(
                error_message=failure,
                error_category="structured_output",
                retryable=True,
                node=node,
                checkpoint=checkpoint,
                fail_count=self._open_fail_count(subject_id, node.id),
                node_index=self._node_index(subject_id, node.id),
                node_total=self._node_total(subject_id),
            )

        assert judgement is not None
        self._repos.answers.add(
            subject_id=subject_id,
            node_id=node.id,
            checkpoint_id=checkpoint.id,
            question=checkpoint.question,
            user_answer=text,
            verdict=judgement.verdict.value,
            confidence=judgement.confidence,
            feedback=judgement.feedback,
            hit_points=judgement.hit_points,
            missing_points=judgement.missing_points,
            attempt_index=self._repos.answers.count_for_node(node.id),
            source=ANSWER_SOURCE_REVIEW,
        )

        if judgement.verdict.value == "pass":
            return self._pass_turn(subject_id, node, checkpoint, judgement)
        return self._fail_turn(subject_id, node, checkpoint, judgement)

    # ------------------------------------------------------------------ #
    # 通过 / 不通过
    # ------------------------------------------------------------------ #

    def _pass_turn(
        self,
        subject_id: str,
        node: KnowledgePoint,
        checkpoint: Checkpoint,
        judgement: JudgementResult,
    ) -> ReviewTurn:
        """复习通过：升级为已掌握 + 解除薄弱点 + 排期后移（FR-503、FR-506，即 AC-22）。"""
        moment = utcnow()
        box, due_at = schedule_after_pass(node.review_box, moment)
        self._repos.paths.update_node(
            node.id,
            state=NodeState.MASTERED,
            review_box=box,
            review_due_at=due_at,
            last_reviewed_at=moment,
            # 复习通过才第一次变成已掌握的节点，这时才需要补掌握时间；
            # 已经有掌握时间的节点保留原值，否则"什么时候学会的"会被复习时间覆盖。
            mastered_at=node.mastered_at or moment,
        )
        self._repos.weak_points.resolve(subject_id, node.id)
        self._state.end_review(subject_id)
        refreshed = self._repos.paths.node(node.id) or node
        self._log.info(f"复习通过，节点升级为已掌握 subject={subject_id} node={node.id}")
        return ReviewTurn(
            text="答对了，这个节点已经补齐，薄弱点已从清单中移除。",
            phase="passed",
            node=refreshed,
            checkpoint=checkpoint,
            verdict=judgement.verdict.value,
            confidence=judgement.confidence,
            feedback=judgement.feedback,
            hit_points=tuple(judgement.hit_points),
            missing_points=tuple(judgement.missing_points),
            promoted=True,
            resolved=True,
            node_index=self._node_index(subject_id, node.id),
            node_total=self._node_total(subject_id),
        )

    def _fail_turn(
        self,
        subject_id: str,
        node: KnowledgePoint,
        checkpoint: Checkpoint,
        judgement: JudgementResult,
    ) -> ReviewTurn:
        """复习不通过：保留薄弱点、累计失败次数，并给出更基础的建议（FR-504）。"""
        # 排期回到第一档：刚证明记不住，下次仍然隔一天就问，别让节点一直躺在
        # "已掌握、几十天后才复习"的状态里（FR-506）。
        box, due_at = schedule_after_fail(utcnow())
        self._repos.paths.update_node(node.id, review_box=box, review_due_at=due_at)
        recorded = self._repos.weak_points.record(subject_id, node.id, WeakPointReason.PARTIAL)
        self._state.end_review(subject_id)
        suggestion = self._suggestion(judgement.missing_points)
        self._log.info(
            f"复习未通过，薄弱点保留 subject={subject_id} node={node.id} fail_count={recorded.fail_count}"
        )
        return ReviewTurn(
            text=f"这次还没通过。{suggestion}",
            phase="failed",
            node=node,
            checkpoint=checkpoint,
            verdict=judgement.verdict.value,
            confidence=judgement.confidence,
            feedback=judgement.feedback,
            hit_points=tuple(judgement.hit_points),
            missing_points=tuple(judgement.missing_points),
            fail_count=recorded.fail_count,
            suggestion=suggestion,
            node_index=self._node_index(subject_id, node.id),
            node_total=self._node_total(subject_id),
        )

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    async def _judge(
        self,
        node: KnowledgePoint,
        checkpoint: Checkpoint,
        answer_text: str,
        session: ConversationSession,
    ) -> tuple[JudgementResult | None, str | None]:
        """独立判定，与教学用同一套判定器（FR-405）。"""
        try:
            result = await session.judge.judge(
                JudgeInput(
                    node_title=node.title,
                    objective=node.objective,
                    question=checkpoint.question,
                    expected_points=list(checkpoint.expected_points),
                    reference_answer=checkpoint.reference_answer,
                    user_answer=answer_text,
                )
            )
        except StructuredOutputError as exc:
            self._log.warning(f"复习判定不可用: {exc.user_message}")
            return None, "这次没能完成判定，请再提交一次回答（你的作答没有丢失）。"
        return result, None

    @staticmethod
    def _question_text(node: KnowledgePoint, checkpoint: Checkpoint) -> str:
        return f"回顾一下「{node.title}」：\n\n{checkpoint.question}"

    @staticmethod
    def _suggestion(missing_points: tuple[str, ...] | list[str]) -> str:
        """把"还缺什么"变成一句可执行的建议（FR-504）。"""
        if missing_points:
            points = "、".join(missing_points)
            return f"{SUGGESTION_HEAD}，重点补上：{points}。"
        return f"{SUGGESTION_HEAD}，换一个更基础的讲法再听一遍。"

    def _open_fail_count(self, subject_id: str, node_id: str) -> int:
        point = next(
            (item for item in self._repos.weak_points.list_open(subject_id) if item.node_id == node_id),
            None,
        )
        return point.fail_count if point is not None else 0

    def _backfill_schedules(
        self,
        nodes: tuple[KnowledgePoint, ...],
        now: datetime,
    ) -> tuple[KnowledgePoint, ...]:
        """给"已掌握但没有排期"的节点补上初始排期（幂等）。

        初始到期日以 ``mastered_at`` 为锚点：老节点往往几十天前就学会了，用
        "现在 + 1 天"会让它看起来还有一天才到期，实际上早该复习。没有掌握时间的
        （历史上教学通过时没记 ``mastered_at``）才退回当前时间。

        只改 ``review_box`` / ``review_due_at`` 这两个字段，其余字段原样返回，
        调用方不需要再查一次路径。
        """
        if not nodes:
            return nodes

        filled: list[KnowledgePoint] = []
        for node in nodes:
            if node.state is not NodeState.MASTERED or node.review_due_at is not None:
                filled.append(node)
                continue
            box, due_at = schedule_after_mastery(node.mastered_at or now)
            self._repos.paths.update_node(node.id, review_box=box, review_due_at=due_at)
            filled.append(replace(node, review_box=box, review_due_at=due_at))
        return tuple(filled)

    def _node_index(self, subject_id: str, node_id: str) -> int:
        path = self._planning.current_path(subject_id)
        if path is None:
            return 0
        for position, node in enumerate(path.ordered_nodes(), start=1):
            if node.id == node_id:
                return position
        return 0

    def _node_total(self, subject_id: str) -> int:
        path = self._planning.current_path(subject_id)
        return path.node_count if path is not None else 0

    def _require_subject(self, subject_id: str) -> Subject:
        subject = self._repos.subjects.get(subject_id)
        if subject is None:
            raise KeyError(f"主题不存在: {subject_id}")
        return subject

    def _require_node(self, subject_id: str, node_id: str) -> KnowledgePoint:
        node = self._repos.paths.node(node_id)
        if node is None:
            raise ReviewError("这个节点已不存在（路径可能被重新生成过），请刷新页面。")
        return node
