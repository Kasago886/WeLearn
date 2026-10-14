"""问答式教学服务（T2.5，FR-401~FR-416）。

这是整个项目最复杂的一段，先把不变量写清楚：

```text
start()    ← 进入/继续当前节点，必要时先讲解
  ├─ 解析"当前该学的节点"：跳过已进入终态（mastered/partial/skipped）的节点
  ├─ 若该节点尚未讲解 → 流式讲解（prompts/tutor.md），然后标记 taught
  ├─ 取该节点"第一个尚未通过的检查点"，登记为待回答
  └─ 输出 = 讲解 + 过渡语 + 检查点问题（问题原文取自库，保证"问的"与"判的"是同一件事）

answer()   ← 学习者作答
  ├─ 守卫：必须有待回答的检查点（否则拒绝，见 SessionStateService）
  ├─ 独立判定（不携带教学对话，避免模型给自己刚讲过的内容打高分）
  │    └─ 判定不可用（结构化失败）→ 可重试错误，**不改节点状态、不清待回答问题**
  ├─ 落 AnswerRecord（含命中/缺失要点）
  ├─ pass
  │    ├─ 还有下一个检查点 → 登记并抛出下一个问题
  │    └─ 已是最后一个  → 节点 mastered + 生成小结入笔记
  │         └─ 还有后续节点 → 提示"进入下一节点"；已无后续节点 → 主题 completed
  └─ partial / fail
       ├─ 尝试次数 ≤ node_max_retries → 反馈 + 换角度重讲 + 再问同一检查点
       └─ 超过上限 → 置 awaiting_decision，发 options，**停下等用户三选一**（FR-408）

decide()   ← 用户三选一
  ├─ reteach     → 按指定方向重讲（可无限次，不计入自动重试上限）+ 再问同一检查点
  ├─ show_answer → 给参考答案 + 简化讲解；节点 → partial；薄弱点（reason=show_answer）
  └─ advance     → 节点 → partial；薄弱点（reason=advanced）
```

**两个容易做错的地方**（都有对应测试守着）：

1. **判定失败 ≠ 未通过**。前者是技术故障，必须可重试且**不能**消耗尝试次数、
   不能清掉待回答问题；后者是学习结果，要推进流程。混为一谈会让用户在一次网络抖动后
   白白丢掉一次机会。
2. **重试耗尽后系统必须停下**。FR-408 明确要求用户自己选"继续重讲"还是"进入下一节点"，
   系统不得代劳。这里通过置位 ``awaiting_decision`` 并返回 ``decision_required=True`` 实现，
   而且该标记落库，因此刷新/重启后用户仍会看到三个选项（不会被重新讲一遍）。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

from app.contracts.schemas import (
    ConversationKind,
    JudgementResult,
    LlmPurpose,
    NodeDecision,
    NodeState,
    SubjectState,
    Verdict,
    WeakPointReason,
)
from app.infra.deepseek.protocol import StreamEvent
from app.infra.logging.queue_logger import CHANNEL_APP, get_logger
from app.infra.settings import AppSettings
from app.infra.storage.models import utcnow
from app.prompts import (
    TUTOR,
    TUTOR_FEEDBACK,
    TUTOR_HINT,
    TUTOR_OFF_TOPIC,
    TUTOR_REFERENCE,
    TUTOR_SUMMARY,
    render,
)
from app.services.learning.assessment import AssessmentService
from app.services.learning.entities import (
    ANSWER_SOURCE_REVIEW,
    FRESH_NODE_STATES,
    REVIEW_MODE_ADJACENT,
    TERMINAL_NODE_STATES,
    Checkpoint,
    KnowledgePoint,
    LearningPath,
    SessionSnapshot,
    Subject,
)
from app.services.learning.planning import PlanningService
from app.services.learning.repository import DomainRepositories
from app.services.learning.session import ConversationSession, ConversationSessionFactory
from app.services.learning.state import AnswerNotAllowedError, SessionStateService
from app.services.llm.agent import DeltaCallback
from app.services.llm.judge import JudgeInput
from app.services.llm.memory import KIND_OFF_TOPIC, ensure_leading_system, user_message
from app.services.llm.result import RoundResult
from app.services.llm.structured import StructuredOutputError


class _Counter(TypedDict):
    """节点序号信息。用 TypedDict 而非 ``dict[str, int]``，这样 ``**`` 展开能通过类型检查。"""

    node_index: int
    node_total: int


QUESTION_TRANSITION = "\n\n---\n\n**来检验一下你的理解：**\n\n"
REFERENCE_PREFIX = "**这是参考答案：**\n\n"
SIMPLIFIED_PREFIX = "\n\n**换个更基础的角度再说一遍：**\n\n"

DIRECTION_HINTS: dict[str, str] = {
    "simpler": "学习者希望讲得更基础一些：请从最小必要的前置概念讲起。",
    "example": "学习者希望换个例子：请换一个更具体、可验算的例子。",
    "hint": "学习者希望得到提示：只给方向性提示，不要给出完整答案。",
}

# --------------------------------------------------------------------------- #
# 分级提示（FR-407）
# --------------------------------------------------------------------------- #

HINT_LEVEL_MIN = 1
HINT_LEVEL_MAX = 3
HINT_LEVEL_LABELS: dict[int, str] = {
    1: "方向性提示",
    2: "关键要素提示",
    3: "近似答案",
}

ACTION_EXPLAIN = "explain"
ACTION_RETEACH = "reteach"
ACTION_REFERENCE = "reference"
ACTION_ASK = "ask"


def hint_level_for(attempts: int) -> int:
    """连续失败越多，默认给的提示越具体（FR-407 的"自动降级"）。

    映射：还没答过 / 答错 1 次 → 1 级；答错 2 次 → 2 级；答错 3 次及以上 → 3 级。
    刻意把第 1 级保留两次：第一次失败后直接上"关键要素"，等于把题目拆掉了。
    """
    return max(HINT_LEVEL_MIN, min(attempts, HINT_LEVEL_MAX))


def hint_action(level: int) -> str:
    return f"hint:{level}"


def _clamp_level(level: int) -> int:
    return max(HINT_LEVEL_MIN, min(level, HINT_LEVEL_MAX))


def _parse_level(action: str) -> int:
    _, _, raw = action.partition(":")
    try:
        return int(raw)
    except ValueError:
        return HINT_LEVEL_MIN


def _hint_request_text(level: int, *, auto: bool) -> str:
    """写进记忆的"用户请求"。

    显式索要时写清级别，这样模型重读上下文时能看出它已经给过几级；
    自动给级时用中性说法，避免让用户看到"你失败了 3 次"这类措辞（FR-210）。
    """
    if auto:
        return "我有点卡住了，给我一点提示吧。"
    return f"请给我第 {level} 级提示（{HINT_LEVEL_LABELS[level]}），不要直接给答案。"


def _reteach_action(direction: str | None) -> str:
    return f"{ACTION_RETEACH}:{direction}" if direction else ACTION_RETEACH


class TeachingError(RuntimeError):
    """教学流程错误（消息面向用户）。"""


@dataclass
class TeachingTurn:
    """一次教学交互的产出，供 API 层直接转成 SSE。"""

    text: str = ""
    node: KnowledgePoint | None = None
    node_index: int = 0
    node_total: int = 0
    phase: str = ""
    """``explain`` / ``question`` / ``reteach`` / ``reference`` / ``mastered`` /
    ``advanced`` / ``completed`` / ``skipped``。"""

    verdict: JudgementResult | None = None
    decision_required: bool = False
    completed: bool = False
    attempts: int = 0
    hint_level: int = 0
    """本轮给出的提示级别（0 表示本轮不是提示）。"""
    pending_question: str = ""
    """提示/穿插提问后重新提醒的检查点问题（用于 ``hint`` 事件）。"""
    result: RoundResult | None = None
    error_message: str = ""
    error_category: str | None = None
    retryable: bool = False

    @property
    def ok(self) -> bool:
        return not self.error_message

    @property
    def passed(self) -> bool:
        return self.verdict is not None and self.verdict.verdict is Verdict.PASS


@dataclass
class TeachingView:
    """教学页数据（FR-410 的读侧）。"""

    snapshot: SessionSnapshot
    path: LearningPath | None
    node: KnowledgePoint | None
    checkpoint: Checkpoint | None
    node_index: int = 0
    node_total: int = 0

    @property
    def can_answer(self) -> bool:
        return self.snapshot.has_pending_question

    @property
    def needs_decision(self) -> bool:
        return self.snapshot.needs_user_decision


class TeachingService:
    """教学循环编排。"""

    def __init__(
        self,
        repos: DomainRepositories,
        state: SessionStateService,
        sessions: ConversationSessionFactory,
        planning: PlanningService,
        assessments: AssessmentService,
        settings_provider: Callable[[], AppSettings],
    ) -> None:
        self._repos = repos
        self._state = state
        self._sessions = sessions
        self._planning = planning
        self._assessments = assessments
        self._settings = settings_provider
        self._log = get_logger(CHANNEL_APP)

    # ------------------------------------------------------------------ #
    # 读取
    # ------------------------------------------------------------------ #

    def view(self, subject_id: str) -> TeachingView:
        snapshot = self._state.snapshot(subject_id)
        path = self._planning.current_path(subject_id)
        node = self._repos.paths.node(snapshot.current_node_id) if snapshot.current_node_id else None

        checkpoint = None
        if snapshot.pending_checkpoint_id:
            checkpoint = self._repos.paths.checkpoint(snapshot.pending_checkpoint_id)

        ordered = path.ordered_nodes() if path is not None else ()
        index = next((i for i, item in enumerate(ordered) if item.id == snapshot.current_node_id), 0)
        return TeachingView(
            snapshot=snapshot,
            path=path,
            node=node,
            checkpoint=checkpoint,
            node_index=index + 1 if node is not None else 0,
            node_total=len(ordered),
        )

    def current_node(self, subject_id: str) -> KnowledgePoint | None:
        snapshot = self._state.snapshot(subject_id)
        if not snapshot.current_node_id:
            return None
        return self._repos.paths.node(snapshot.current_node_id)

    # ------------------------------------------------------------------ #
    # 讲解 + 提问
    # ------------------------------------------------------------------ #

    async def start(self, subject_id: str, *, on_delta: DeltaCallback | None = None) -> TeachingTurn:
        """进入（或继续）当前节点：必要时讲解，然后抛出检查点问题。"""
        self._require_teaching_state(subject_id)
        path = self._require_path(subject_id)
        snapshot = self._state.snapshot(subject_id)

        # 已经在等三选一：不要重复讲解，直接把选项再给一次
        if snapshot.needs_user_decision:
            node = self._repos.paths.node(snapshot.current_node_id) if snapshot.current_node_id else None
            return TeachingTurn(
                text="这个节点还需要你选择下一步怎么做。",
                node=node,
                phase="decision",
                decision_required=True,
                attempts=snapshot.node_attempts,
                **self._counter(path, node),
            )

        # 已经在等作答：不要重新讲解一遍（AC-12 的基础）
        if snapshot.has_pending_question:
            node = self._repos.paths.node(snapshot.current_node_id) if snapshot.current_node_id else None
            question = snapshot.pending_question or ""
            text = QUESTION_TRANSITION + question
            self._emit(on_delta, text)
            return TeachingTurn(
                text=text,
                node=node,
                phase="question",
                attempts=snapshot.node_attempts,
                **self._counter(path, node),
            )

        # 正在等一次"回顾上一节点"的作答：把同一个问题再给一次，不要重新挑题
        if snapshot.has_review_question:
            return await self._resume_adjacent_review(subject_id, path, snapshot, on_delta=on_delta)

        node = self._resolve_node(path, snapshot)
        if node is None:
            # 全部节点都已完成
            self._finish_subject(subject_id)
            return TeachingTurn(
                text="本次学习的所有节点都已完成。",
                phase="completed",
                completed=True,
                node_total=path.node_count,
            )

        # FR-414：进入新节点前先快速回顾上一节点。答对才继续讲新节点，
        # 答错则回到上一节点补强——不复习就直接往下讲，学过的会很快忘掉。
        if snapshot.adjacent_review_for != node.id:
            pending = self._adjacent_review_target(path, node)
            if pending is not None:
                return self._ask_adjacent_review(subject_id, path, node, pending, on_delta=on_delta)

        # FR-309：进入节点就算"学习中"。不标记的话节点列表会一直显示"未开始"，
        # 用户看不出自己正在学哪个节点。
        if node.state in FRESH_NODE_STATES:
            self._repos.paths.update_node(node.id, state=NodeState.LEARNING)
            node = self._repos.paths.node(node.id) or node

        session = self._sessions.open(subject_id, ConversationKind.TEACH)
        if not self._state.snapshot(subject_id).node_taught:
            explanation = await self._explain(subject_id, path, node, session, on_delta=on_delta)
            if not explanation.ok:
                return TeachingTurn(
                    result=explanation,
                    error_message=explanation.message,
                    error_category=explanation.error_category,
                    retryable=explanation.retryable,
                    node=node,
                    phase="explain",
                    **self._counter(path, node),
                )
            self._state.mark_taught(subject_id)
            self._state.set_last_action(subject_id, ACTION_EXPLAIN)
            session.persist()
            explanation_text = explanation.content
        else:
            explanation_text = ""

        checkpoint = self._next_checkpoint(node)
        if checkpoint is None:
            # 节点没有检查点（理论上不会发生）：直接视为已掌握并推进
            self._log.warning(f"节点缺少检查点，按已掌握处理: {node.id}")
            self._master_node(subject_id, node)
            return TeachingTurn(
                text="这个节点没有配置检查点，已标记为已完成。",
                node=node,
                phase="mastered",
                **self._counter(path, node),
            )

        self._state.set_pending_question(
            subject_id,
            checkpoint_id=checkpoint.id,
            question=checkpoint.question,
        )
        question_text = QUESTION_TRANSITION + checkpoint.question
        self._emit(on_delta, question_text)

        return TeachingTurn(
            text=explanation_text + question_text,
            node=node,
            phase="question",
            attempts=self._state.snapshot(subject_id).node_attempts,
            **self._counter(path, node),
        )

    # ------------------------------------------------------------------ #
    # 相邻复习（FR-414、FR-501）
    # ------------------------------------------------------------------ #

    def _adjacent_review_target(
        self,
        path: LearningPath,
        node: KnowledgePoint,
    ) -> KnowledgePoint | None:
        """该不该在进入 ``node`` 之前回顾上一节点？返回要回顾的节点。

        条件是"上一节点确实被答过"——没答过就无从回顾。已掌握、部分掌握、跳过的节点
        都算答过（跳过的节点也有作答记录），因为它们的检查点都问过。
        """
        previous = self._previous_node(path, node)
        if previous is None:
            return None
        if self._repos.answers.count_for_node(previous.id) <= 0:
            return None
        return previous

    @staticmethod
    def _previous_node(path: LearningPath, node: KnowledgePoint) -> KnowledgePoint | None:
        ordered = path.ordered_nodes()
        for position, item in enumerate(ordered):
            if item.id == node.id:
                return ordered[position - 1] if position > 0 else None
        return None

    def _ask_adjacent_review(
        self,
        subject_id: str,
        path: LearningPath,
        node: KnowledgePoint,
        previous: KnowledgePoint,
        *,
        on_delta: DeltaCallback | None,
    ) -> TeachingTurn:
        """抛出"回顾上一节点"的快速提问，暂不改动当前节点。"""
        checkpoint = self._repos.paths.first_checkpoint(previous.id)
        if checkpoint is None:
            # 上一节点没有检查点就没法回顾，直接放行去讲新节点
            self._state.mark_adjacent_review_done(subject_id, node_id=node.id)
            return self._adjacent_skip_turn(subject_id, path, node, on_delta=on_delta)

        self._state.begin_review(
            subject_id,
            mode=REVIEW_MODE_ADJACENT,
            node_id=previous.id,
            checkpoint_id=checkpoint.id,
        )
        text = (
            f"在进入「{node.title}」之前，先快速回顾一下上一节「{previous.title}」。\n\n"
            f"{checkpoint.question}"
        )
        self._emit(on_delta, text)
        return TeachingTurn(
            text=text,
            node=node,
            phase="review_question",
            attempts=self._state.snapshot(subject_id).node_attempts,
            **self._counter(path, node),
        )

    def _adjacent_skip_turn(
        self,
        subject_id: str,
        path: LearningPath,
        node: KnowledgePoint,
        *,
        on_delta: DeltaCallback | None,
    ) -> TeachingTurn:
        """没有可回顾的内容时，接着说清当前节点。"""
        return TeachingTurn(
            text="",
            node=node,
            phase="question",
            attempts=self._state.snapshot(subject_id).node_attempts,
            **self._counter(path, node),
        )

    async def _resume_adjacent_review(
        self,
        subject_id: str,
        path: LearningPath,
        snapshot: SessionSnapshot,
        *,
        on_delta: DeltaCallback | None,
    ) -> TeachingTurn:
        """刷新后再点「继续学习」：把同一个回顾问题再给一次。"""
        node = self._repos.paths.node(snapshot.current_node_id) if snapshot.current_node_id else None
        checkpoint = (
            self._repos.paths.checkpoint(snapshot.review_checkpoint_id)
            if snapshot.review_checkpoint_id
            else None
        )
        if node is None or checkpoint is None:
            # 节点或检查点已被删掉（例如路径被重新生成）：静默收尾，不要卡住
            self._state.end_review(subject_id)
            if node is not None:
                return await self.start(subject_id, on_delta=on_delta)
            raise TeachingError("当前节点已不存在，请重新生成学习路径。")

        text = checkpoint.question
        self._emit(on_delta, text)
        return TeachingTurn(
            text=text,
            node=node,
            phase="review_question",
            attempts=snapshot.node_attempts,
            **self._counter(path, node),
        )

    async def _answer_adjacent_review(
        self,
        subject_id: str,
        text: str,
        *,
        on_delta: DeltaCallback | None,
    ) -> TeachingTurn:
        """处理"回顾上一节点"的作答：答对继续讲新节点，答错回到上一节点补强。"""
        snapshot = self._state.snapshot(subject_id)
        previous = self._require_node(snapshot.review_node_id)
        checkpoint = (
            self._repos.paths.checkpoint(snapshot.review_checkpoint_id)
            if snapshot.review_checkpoint_id
            else None
        )
        if checkpoint is None:
            self._state.end_review(subject_id)
            raise TeachingError("这次回顾的检查点已失效，请重新点击「继续学习」。")

        session = self._sessions.open(subject_id, ConversationKind.TEACH)
        session.memory.add_round().add(user_message(text))
        session.persist()

        judgement, failure = await self._judge(previous, checkpoint, text, session)
        if failure is not None:
            # 判定故障：不动任何状态，用户直接重试
            return TeachingTurn(
                error_message=failure,
                error_category="structured_output",
                retryable=True,
                node=previous,
                phase="review_question",
            )

        assert judgement is not None
        self._repos.answers.add(
            subject_id=subject_id,
            node_id=previous.id,
            checkpoint_id=checkpoint.id,
            question=checkpoint.question,
            user_answer=text,
            verdict=judgement.verdict.value,
            confidence=judgement.confidence,
            feedback=judgement.feedback,
            hit_points=judgement.hit_points,
            missing_points=judgement.missing_points,
            attempt_index=self._repos.answers.count_for_node(previous.id),
            source=ANSWER_SOURCE_REVIEW,
        )

        if judgement.verdict.value == "pass":
            return await self._adjacent_passed(subject_id, previous, on_delta=on_delta)
        return await self._adjacent_failed(
            subject_id, previous, checkpoint, judgement, on_delta=on_delta
        )

    async def _adjacent_passed(
        self,
        subject_id: str,
        previous: KnowledgePoint,
        *,
        on_delta: DeltaCallback | None,
    ) -> TeachingTurn:
        """答对：简述带过，接着进入新节点（FR-414）。"""
        snapshot = self._state.snapshot(subject_id)
        entering_id = snapshot.current_node_id or previous.id
        self._state.end_review(subject_id)
        self._state.mark_adjacent_review_done(subject_id, node_id=entering_id)
        self._repos.paths.update_node(previous.id, last_reviewed_at=utcnow())

        ack = f"回顾「{previous.title}」通过了，我们继续。\n\n"
        self._emit(on_delta, ack)
        turn = await self.start(subject_id, on_delta=on_delta)
        turn.text = ack + turn.text
        return turn

    async def _adjacent_failed(
        self,
        subject_id: str,
        previous: KnowledgePoint,
        checkpoint: Checkpoint,
        judgement: JudgementResult,
        *,
        on_delta: DeltaCallback | None,
    ) -> TeachingTurn:
        """答错：回到上一节点的补强流程（FR-414）。

        做法是"切回上一节点 + 重新讲解 + 再次提问"，也就是 FR-406 的"换角度重讲 → 再次提问"。
        同时记一个薄弱点，这样它会出现在复习页里，不会因为"只是个回顾题"而被放过。
        """
        missing = "、".join(judgement.missing_points)
        reason = f"刚才回顾时还差：{missing}。" if missing else "刚才回顾时没答上来。"
        self._state.end_review(subject_id)
        self._repos.weak_points.record(subject_id, previous.id, WeakPointReason.PARTIAL)
        # 回顾没答上来 = 其实并没有真正掌握，所以把它降回"学习中"再补一遍。
        # **只清会话指针是不够的**：`_resolve_node` 会跳过终态节点，节点保持 mastered
        # 的话下一句就会被解析回新节点，"回到上一节点补强"等于没发生。
        self._repos.paths.update_node(previous.id, state=NodeState.LEARNING)
        self._state.enter_node(subject_id, previous)
        # 回到上一节点是"结果"，不是"又要进入新节点"：抑制它自己的回顾环节
        self._state.mark_adjacent_review_done(subject_id, node_id=previous.id)

        lead = f"这次回顾没过：{reason}\n\n我们回到「{previous.title}」再补一遍。\n\n"
        self._emit(on_delta, lead)
        path = self._require_path(subject_id)
        return await self._reinforce(
            subject_id, path, previous, checkpoint, lead, on_delta=on_delta
        )

    async def _reinforce(
        self,
        subject_id: str,
        path: LearningPath,
        node: KnowledgePoint,
        checkpoint: Checkpoint,
        lead: str,
        *,
        on_delta: DeltaCallback | None,
    ) -> TeachingTurn:
        """补强：重讲一遍，然后**重问同一个检查点**。

        这里不能走 ``_next_checkpoint``：它返回"第一个尚未通过的检查点"，而上一节点的检查点
        在之前的教学里已经通过过了，于是它会返回 ``None``，``start()`` 就按"无检查点=已掌握"
        把节点又标回 mastered——刚降的级和刚记的薄弱点一起被抹掉，补强等于空转。
        回顾失败恰恰说明"这道题现在答不上来"，所以重问它才是补强。
        """
        session = self._sessions.open(subject_id, ConversationKind.TEACH)
        explanation = await self._explain(subject_id, path, node, session, on_delta=on_delta)
        if not explanation.ok:
            return TeachingTurn(
                result=explanation,
                error_message=explanation.message,
                error_category=explanation.error_category,
                retryable=explanation.retryable,
                node=node,
                phase="explain",
                **self._counter(path, node),
            )

        self._state.mark_taught(subject_id)
        self._state.set_last_action(subject_id, ACTION_EXPLAIN)
        self._state.set_pending_question(
            subject_id,
            checkpoint_id=checkpoint.id,
            question=checkpoint.question,
        )
        session.persist()

        question_text = QUESTION_TRANSITION + checkpoint.question
        self._emit(on_delta, question_text)
        return TeachingTurn(
            text=lead + explanation.content + question_text,
            node=node,
            phase="question",
            attempts=self._state.snapshot(subject_id).node_attempts,
            **self._counter(path, node),
        )

    # ------------------------------------------------------------------ #
    # 作答与判定
    # ------------------------------------------------------------------ #

    async def answer(
        self,
        subject_id: str,
        answer: str,
        *,
        on_delta: DeltaCallback | None = None,
    ) -> TeachingTurn:
        """提交检查点作答：判定 → 通过则推进，未通过则重讲或交给用户决定。"""
        text = (answer or "").strip()
        if not text:
            return TeachingTurn(
                error_message="请先写出你的回答。",
                error_category="validation",
            )

        # 正停在"回顾上一节点"上：这条回答属于那次回顾，不属于当前节点的检查点
        if self._state.snapshot(subject_id).review_mode == REVIEW_MODE_ADJACENT:
            return await self._answer_adjacent_review(subject_id, text, on_delta=on_delta)

        # 守卫会抛 AnswerNotAllowedError（没有待回答问题 / 正在等三选一 / 已暂停等）
        _snapshot, checkpoint_id, question = self._state.require_pending_question(subject_id)
        path = self._require_path(subject_id)
        node = self._require_node(_snapshot.current_node_id)
        checkpoint = self._repos.paths.checkpoint(checkpoint_id)
        if checkpoint is None:
            raise TeachingError("待回答的检查点已失效，请点击「继续学习」重新提问。")

        session = self._sessions.open(subject_id, ConversationKind.TEACH)
        session.memory.add_round().add(user_message(text))
        session.persist()

        # 先判定再计次：判定不可用属于技术故障，不该消耗用户的尝试机会
        judgement, failure = await self._judge(node, checkpoint, text, session)
        if failure is not None:
            # 判定不可用：不改节点状态、不清待回答问题，用户可直接重试
            return TeachingTurn(
                error_message=failure,
                error_category="structured_output",
                retryable=True,
                node=node,
                phase="judged",
                attempts=self._state.snapshot(subject_id).node_attempts,
                **self._counter(path, node),
            )

        assert judgement is not None
        attempts = self._state.increment_attempts(subject_id)
        self._repos.answers.add(
            subject_id=subject_id,
            node_id=node.id,
            checkpoint_id=checkpoint.id,
            question=question,
            user_answer=text,
            verdict=judgement.verdict.value,
            confidence=judgement.confidence,
            feedback=judgement.feedback,
            hit_points=judgement.hit_points,
            missing_points=judgement.missing_points,
            attempt_index=attempts,
        )

        if judgement.verdict is Verdict.PASS:
            return await self._handle_pass(
                subject_id, path, node, session, checkpoint, text, judgement, on_delta, attempts
            )

        return await self._handle_failure(
            subject_id, path, node, session, checkpoint, judgement, text, on_delta, attempts
        )

    # ------------------------------------------------------------------ #
    # 用户三选一
    # ------------------------------------------------------------------ #

    async def decide(
        self,
        subject_id: str,
        decision: NodeDecision,
        *,
        direction: str | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> TeachingTurn:
        """处理重试耗尽后的三选一（FR-408）。"""
        self._state.require_decision_pending(subject_id)
        path = self._require_path(subject_id)
        snapshot = self._state.snapshot(subject_id)
        node = self._require_node(snapshot.current_node_id)
        session = self._sessions.open(subject_id, ConversationKind.TEACH)

        if decision is NodeDecision.RETEACH:
            return await self._reteach(
                subject_id, path, node, session, direction=direction, on_delta=on_delta
            )

        if decision is NodeDecision.SHOW_ANSWER:
            reference = await self._show_reference(subject_id, path, node, session, on_delta=on_delta)
            self._mark_node_partial(subject_id, node, reason=WeakPointReason.SHOW_ANSWER)
            self._state.clear_decision(subject_id)
            self._state.clear_pending(subject_id)
            self._state.set_last_action(subject_id, ACTION_REFERENCE)
            if not reference.ok:
                return TeachingTurn(
                    result=reference,
                    error_message=reference.message,
                    error_category=reference.error_category,
                    retryable=reference.retryable,
                    node=node,
                    phase="reference",
                    **self._counter(path, node),
                )
            return TeachingTurn(
                text=reference.content,
                node=node,
                phase="reference",
                attempts=snapshot.node_attempts,
                **self._counter(path, node),
            )

        # NodeDecision.ADVANCE：用户选择直接进入下一节点
        self._mark_node_partial(subject_id, node, reason=WeakPointReason.ADVANCED)
        self._state.clear_decision(subject_id)
        self._state.clear_pending(subject_id)
        self._log.info(f"用户选择直接进入下一节点 subject={subject_id} node={node.id}")

        next_node = self._repos.paths.next_node(path.id, after_order_index=node.order_index)
        if next_node is None:
            self._finish_subject(subject_id)
            return TeachingTurn(
                text="已跳过这个节点的检查，本次学习的所有节点都已完成。",
                node=node,
                phase="completed",
                completed=True,
                **self._counter(path, node),
            )

        # 用户已经决定离开这个节点，会话就落在下一个节点上，避免出现
        # "node_state 报的是节点 2 但状态接口说当前还在节点 1" 这种互相矛盾的状态。
        self._state.enter_node(subject_id, next_node)

        return TeachingTurn(
            text="好的，我们先进入下一个节点。",
            node=next_node,
            phase="advanced",
            **self._counter(path, next_node),
        )

    # ------------------------------------------------------------------ #
    # 跳过与暂停
    # ------------------------------------------------------------------ #

    def skip_node(self, subject_id: str) -> TeachingTurn:
        """主动跳过当前节点（FR-409）：标记 skipped 并写入薄弱点。"""
        self._require_teaching_state(subject_id)
        path = self._require_path(subject_id)
        snapshot = self._state.snapshot(subject_id)
        node = self._require_node(snapshot.current_node_id)

        self._repos.paths.update_node(node.id, state=NodeState.SKIPPED)
        self._repos.weak_points.record(subject_id, node.id, WeakPointReason.SKIPPED)
        self._state.clear_pending(subject_id)
        self._state.clear_decision(subject_id)
        self._log.info(f"用户跳过节点 subject={subject_id} node={node.id}")

        refreshed = self._repos.paths.node(node.id)
        next_node = self._repos.paths.next_node(path.id, after_order_index=node.order_index)
        if next_node is None:
            self._finish_subject(subject_id)
            return TeachingTurn(
                text="已跳过该节点，本次学习的所有节点都已完成。",
                node=refreshed,
                phase="completed",
                completed=True,
                **self._counter(path, refreshed),
            )

        return TeachingTurn(
            text="已跳过该节点，它可以稍后在薄弱点复习里补上。",
            node=refreshed,
            phase="skipped",
            **self._counter(path, refreshed),
        )

    def pause(self, subject_id: str) -> SessionSnapshot:
        return self._state.transition(subject_id, SubjectState.PAUSED)

    def resume(self, subject_id: str) -> SessionSnapshot:
        return self._state.transition(subject_id, SubjectState.TEACHING)

    # ------------------------------------------------------------------ #
    # 分级提示（FR-407）
    # ------------------------------------------------------------------ #

    async def hint(
        self,
        subject_id: str,
        *,
        level: int | None = None,
        on_delta: DeltaCallback | None = None,
    ) -> TeachingTurn:
        """给一级提示。用户可主动索取；不指定级别时按"连续失败次数"自动选级。

        **提示不消耗尝试次数**：它是脚手架，不是作答。也不改变节点状态——
        用户拿着提示答对了，判定仍然是独立的判定，该通过才通过。
        """
        snapshot, checkpoint_id, _question = self._state.require_pending_question(subject_id)
        path = self._require_path(subject_id)
        node = self._require_node(snapshot.current_node_id)
        checkpoint = self._repos.paths.checkpoint(checkpoint_id)
        if checkpoint is None:
            raise TeachingError("待回答的检查点已失效，请点击「继续学习」重新提问。")

        auto = level is None
        chosen = hint_level_for(snapshot.node_attempts) if level is None else _clamp_level(level)

        session = self._sessions.open(subject_id, ConversationKind.TEACH)
        prompt = render(
            TUTOR_HINT,
            concept=self._require_subject(subject_id).title,
            node_title=node.title,
            objective=node.objective,
            key_points=list(node.key_points),
            question=checkpoint.question,
            expected_points=list(checkpoint.expected_points),
            history=self._answer_history(node.id),
            level=chosen,
            level_label=HINT_LEVEL_LABELS[chosen],
            reason=(
                f"（学习者已连续作答 {snapshot.node_attempts} 次未通过，系统自动给到这一级。）"
                if auto and snapshot.node_attempts > 0
                else "（学习者主动索要提示。）"
            ),
        )
        session.memory.add_round().add(user_message(_hint_request_text(chosen, auto=auto)))
        self._set_prompt(session, prompt)
        result = await session.executor.stream("", on_delta=on_delta, purpose=LlmPurpose.TEACH)
        session.persist()

        if not result.ok:
            return TeachingTurn(
                result=result,
                error_message=result.message,
                error_category=result.error_category,
                retryable=result.retryable,
                node=node,
                phase="hint",
                attempts=snapshot.node_attempts,
                **self._counter(path, node),
            )

        # 提示之后仍然保持"待回答同一个检查点"：这不是作答，问题不能变
        self._state.set_last_action(subject_id, hint_action(chosen))
        tail = QUESTION_TRANSITION + checkpoint.question
        self._emit(on_delta, tail)

        return TeachingTurn(
            text=result.content + tail,
            node=node,
            phase="hint",
            hint_level=chosen,
            pending_question=checkpoint.question,
            attempts=snapshot.node_attempts,
            result=result,
            **self._counter(path, node),
        )

    # ------------------------------------------------------------------ #
    # 自由提问（FR-411）
    # ------------------------------------------------------------------ #

    async def ask(
        self,
        subject_id: str,
        question: str,
        *,
        on_delta: DeltaCallback | None = None,
    ) -> TeachingTurn:
        """穿插提问：先回答问题，再回到教学主线。

        **绝不改变教学状态**：不写 ``answer_record``、不动尝试次数、不动待回答的检查点。
        没有这条约束，用户随口问一句"这个和 XX 有什么区别"就会被当成检查点答案送去判定。

        消息在记忆里标记为 :data:`KIND_OFF_TOPIC`，这样"教学主线"与"随口提问"
        在存储层面就能分开（档案与日志都用得上）。

        也**不写** ``last_action``：随口提问是岔路，"重新生成上一条"应该重跑主线上的那一句，
        而不是把用户自己的问题再回答一遍。
        """
        text = (question or "").strip()
        if not text:
            return TeachingTurn(error_message="请先写下你的问题。", error_category="validation")

        self._require_teaching_state(subject_id)
        path = self._require_path(subject_id)
        snapshot = self._state.snapshot(subject_id)
        node = self._require_node(snapshot.current_node_id)
        subject = self._require_subject(subject_id)

        session = self._sessions.open(subject_id, ConversationKind.TEACH)
        session.memory.add_round().add(user_message(text), kind=KIND_OFF_TOPIC)
        prompt = render(
            TUTOR_OFF_TOPIC,
            concept=subject.title,
            node_title=node.title,
            objective=node.objective,
            question=snapshot.pending_question or "（当前没有待回答的检查点问题）",
            off_topic_question=text,
        )
        self._set_prompt(session, prompt)
        result = await session.executor.stream("", on_delta=on_delta, purpose=LlmPurpose.TEACH)
        if result.ok:
            # 标记必须落在**提交之后**的轮次上：执行器在快照上运行、成功后整体提交，
            # 提交后 memory 里已经是新的轮次对象，先前持有的引用已经不在记忆里了。
            for envelope in session.memory.rounds[-1].messages:
                envelope.kind = KIND_OFF_TOPIC
        session.persist()

        if not result.ok:
            return TeachingTurn(
                result=result,
                error_message=result.message,
                error_category=result.error_category,
                retryable=result.retryable,
                node=node,
                phase="off_topic",
                attempts=snapshot.node_attempts,
                **self._counter(path, node),
            )

        # 回到主线：把待答问题再说一遍，让用户知道接着做什么
        tail = QUESTION_TRANSITION + snapshot.pending_question if snapshot.pending_question else ""
        if tail:
            self._emit(on_delta, tail)

        return TeachingTurn(
            text=result.content + tail,
            node=node,
            phase="off_topic",
            attempts=snapshot.node_attempts,
            result=result,
            **self._counter(path, node),
        )

    # ------------------------------------------------------------------ #
    # 重新生成上一条回复（FR-413）
    # ------------------------------------------------------------------ #

    async def regenerate_last(self, subject_id: str, *, on_delta: DeltaCallback | None = None) -> TeachingTurn:
        """按记录下来的"上一步动作"重跑一次，产出一条新的回复。

        重生成**不撤销已发生的判定**：如果上一条回复是重讲或提示，说明用户已经答错过，
        那些作答记录是事实，不该因为"再来一次"就消失。重生成只是换一种说法再说一遍。
        """
        self._require_teaching_state(subject_id)
        action = self._state.snapshot(subject_id).last_action
        if not action:
            raise TeachingError("现在还没有可以重新生成的回复，请先让老师讲解或提问。")

        if action == ACTION_EXPLAIN:
            path = self._require_path(subject_id)
            node = self._require_node(self._state.snapshot(subject_id).current_node_id)
            session = self._sessions.open(subject_id, ConversationKind.TEACH)
            result = await self._explain(subject_id, path, node, session, on_delta=on_delta)
            if not result.ok:
                return self._failure_turn(result, path, node, phase="explain")
            return TeachingTurn(
                text=result.content,
                node=node,
                phase="explain",
                attempts=self._state.snapshot(subject_id).node_attempts,
                result=result,
                **self._counter(path, node),
            )

        if action.startswith("hint:"):
            level = _clamp_level(_parse_level(action))
            return await self.hint(subject_id, level=level, on_delta=on_delta)

        direction = action.split(":", 1)[1] if ":" in action else None
        path = self._require_path(subject_id)
        node = self._require_node(self._state.snapshot(subject_id).current_node_id)
        session = self._sessions.open(subject_id, ConversationKind.TEACH)
        return await self._reteach(
            subject_id,
            path,
            node,
            session,
            direction=direction,
            on_delta=on_delta,
            attempts=self._state.snapshot(subject_id).node_attempts,
        )

    # ------------------------------------------------------------------ #
    # 通过 / 未通过
    # ------------------------------------------------------------------ #

    async def _handle_pass(
        self,
        subject_id: str,
        path: LearningPath,
        node: KnowledgePoint,
        session: ConversationSession,
        checkpoint: Checkpoint,
        answer_text: str,
        judgement: JudgementResult,
        on_delta: DeltaCallback | None,
        attempts: int,
    ) -> TeachingTurn:
        next_checkpoint = self._repos.paths.next_checkpoint(node.id, checkpoint.order_index)

        if next_checkpoint is not None:
            # 同一节点还有下一个检查点：继续问，不标为已掌握
            self._state.set_pending_question(
                subject_id, checkpoint_id=next_checkpoint.id, question=next_checkpoint.question
            )
            # 上一条回复已经翻篇（继续问下一个检查点），不能再"重新生成"它
            self._state.clear_last_action(subject_id)
            text = "答对了。\n\n" + QUESTION_TRANSITION + next_checkpoint.question
            self._emit(on_delta, text)
            return TeachingTurn(
                text=text,
                node=node,
                phase="question",
                verdict=judgement,
                attempts=attempts,
                **self._counter(path, node),
            )

        summary = await self._summarise(node, checkpoint, answer_text, session)
        self._master_node(subject_id, node, summary=summary)
        refreshed = self._repos.paths.node(node.id)

        next_node = self._repos.paths.next_node(path.id, after_order_index=node.order_index)
        if next_node is None:
            self._finish_subject(subject_id)
            return TeachingTurn(
                text=self._pass_text(summary) + "\n\n本次学习的所有节点都已完成。",
                node=refreshed,
                phase="completed",
                verdict=judgement,
                completed=True,
                attempts=attempts,
                **self._counter(path, refreshed),
            )

        return TeachingTurn(
            text=self._pass_text(summary),
            node=refreshed,
            phase="mastered",
            verdict=judgement,
            attempts=attempts,
            **self._counter(path, refreshed),
        )

    async def _handle_failure(
        self,
        subject_id: str,
        path: LearningPath,
        node: KnowledgePoint,
        session: ConversationSession,
        checkpoint: Checkpoint,
        judgement: JudgementResult,
        answer_text: str,
        on_delta: DeltaCallback | None,
        attempts: int,
    ) -> TeachingTurn:
        limit = self._settings().node_max_retries

        if attempts > limit:
            # FR-408：停下，等用户决定，绝不自动前进或自动放弃
            self._state.request_decision(subject_id)
            self._log.info(
                f"节点重试耗尽，等待用户选择 subject={subject_id} node={node.id} attempts={attempts} limit={limit}"
            )
            return TeachingTurn(
                node=node,
                phase="decision",
                verdict=judgement,
                decision_required=True,
                attempts=attempts,
                **self._counter(path, node),
            )

        return await self._reteach(
            subject_id,
            path,
            node,
            session,
            direction=None,
            on_delta=on_delta,
            judgement=judgement,
            checkpoint=checkpoint,
            answer_text=answer_text,
            attempts=attempts,
        )

    async def _reteach(
        self,
        subject_id: str,
        path: LearningPath,
        node: KnowledgePoint,
        session: ConversationSession,
        *,
        direction: str | None,
        on_delta: DeltaCallback | None,
        judgement: JudgementResult | None = None,
        checkpoint: Checkpoint | None = None,
        answer_text: str = "",
        attempts: int | None = None,
    ) -> TeachingTurn:
        """反馈 + 换角度重讲，然后重新问同一个检查点。"""
        target = checkpoint or self._last_checkpoint(node)
        if target is None:
            raise TeachingError("该节点没有可用的检查点，无法继续。")

        history = self._repos.answers.list_for_node(node.id)
        last = judgement
        if last is None and history:
            latest = history[-1]
            last = JudgementResult(
                verdict=Verdict(latest.verdict),
                confidence=latest.confidence or 0.0,
                feedback=latest.feedback,
                hit_points=list(latest.hit_points),
                missing_points=list(latest.missing_points),
            )

        prompt = render(
            TUTOR_FEEDBACK,
            node_title=node.title,
            objective=node.objective,
            question=target.question,
            expected_points=list(target.expected_points),
            user_answer=answer_text or (history[-1].user_answer if history else "（无）"),
            verdict_feedback=last.feedback if last is not None else "（无）",
            missing_points=list(last.missing_points) if last is not None else [],
            direction_hint=DIRECTION_HINTS.get(direction or "", "（无额外要求）"),
        )
        self._set_prompt(session, prompt)
        result = await session.executor.stream(
            "",
            on_delta=on_delta,
            purpose=LlmPurpose.TEACH,
        )
        session.persist()

        if not result.ok:
            return TeachingTurn(
                result=result,
                error_message=result.message,
                error_category=result.error_category,
                retryable=result.retryable,
                node=node,
                phase="reteach",
                attempts=attempts or self._state.snapshot(subject_id).node_attempts,
                **self._counter(path, node),
            )

        # 重讲后重新登记同一个检查点（FR-406：换角度重讲 → 再次提问）
        self._state.set_pending_question(subject_id, checkpoint_id=target.id, question=target.question)
        self._state.set_last_action(subject_id, _reteach_action(direction))
        tail = QUESTION_TRANSITION + target.question
        self._emit(on_delta, tail)

        return TeachingTurn(
            text=result.content + tail,
            node=node,
            phase="reteach",
            verdict=judgement,
            attempts=attempts or self._state.snapshot(subject_id).node_attempts,
            result=result,
            **self._counter(path, node),
        )

    async def _show_reference(
        self,
        subject_id: str,
        path: LearningPath,
        node: KnowledgePoint,
        session: ConversationSession,
        *,
        on_delta: DeltaCallback | None,
    ) -> RoundResult:
        """给参考答案 + 简化讲解（FR-408 的第二个选项）。"""
        checkpoint = self._last_checkpoint(node)
        if checkpoint is None:
            raise TeachingError("该节点没有可用的检查点，无法给出参考答案。")

        history = self._repos.answers.list_for_node(node.id)
        prompt = render(
            TUTOR_REFERENCE,
            node_title=node.title,
            objective=node.objective,
            question=checkpoint.question,
            expected_points=list(checkpoint.expected_points),
            reference_answer=checkpoint.reference_answer,
            user_answer=history[-1].user_answer if history else "（无）",
        )
        self._set_prompt(session, prompt)
        result = await session.executor.stream(
            "",
            on_delta=on_delta,
            purpose=LlmPurpose.TEACH,
        )
        session.persist()
        return result

    # ------------------------------------------------------------------ #
    # 讲解与小结
    # ------------------------------------------------------------------ #

    async def _explain(
        self,
        subject_id: str,
        path: LearningPath,
        node: KnowledgePoint,
        session: ConversationSession,
        *,
        on_delta: DeltaCallback | None,
    ) -> RoundResult:
        subject = self._require_subject(subject_id)
        ordered = path.ordered_nodes()
        index = next((i for i, item in enumerate(ordered) if item.id == node.id), 0)

        prompt = render(
            TUTOR,
            node_title=node.title,
            objective=node.objective,
            key_points=list(node.key_points),
            concept=subject.title,
            node_index=index + 1,
            node_total=len(ordered),
        )
        session.memory.add_round().add(user_message(_explain_instruction(index + 1, len(ordered))))
        self._set_prompt(session, prompt)
        result = await session.executor.stream(
            "",
            on_delta=on_delta,
            purpose=LlmPurpose.TEACH,
        )
        session.persist()
        return result

    async def _summarise(
        self,
        node: KnowledgePoint,
        checkpoint: Checkpoint,
        answer_text: str,
        session: ConversationSession,
    ) -> str:
        """生成节点小结（FR-415）；失败时降级为不写小结，不影响主流程。"""
        prompt = render(
            TUTOR_SUMMARY,
            node_title=node.title,
            objective=node.objective,
            key_points=list(node.key_points),
            question=checkpoint.question,
            user_answer=answer_text,
        )
        self._set_prompt(session, prompt)
        result = await session.executor.stream(
            "",
            purpose=LlmPurpose.TEACH,
        )
        if not result.ok:
            self._log.warning(f"节点小结生成失败（已忽略）: {result.message}")
            return ""
        session.persist()
        return result.content.strip()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _failure_turn(
        self,
        result: RoundResult,
        path: LearningPath,
        node: KnowledgePoint,
        *,
        phase: str,
    ) -> TeachingTurn:
        """把一次失败的流式调用统一转成回合（不动任何状态）。"""
        return TeachingTurn(
            result=result,
            error_message=result.message,
            error_category=result.error_category,
            retryable=result.retryable,
            node=node,
            phase=phase,
            attempts=self._state.snapshot(path.subject_id).node_attempts,
            **self._counter(path, node),
        )

    def _answer_history(self, node_id: str, *, limit: int = 5) -> str:
        """把该节点最近的作答整理成提示词可用的紧凑文本。"""
        records = self._repos.answers.list_for_node(node_id)[-limit:]
        if not records:
            return "（还没有作答记录）"
        lines: list[str] = []
        for record in records:
            verdict = {"pass": "通过", "partial": "部分通过", "fail": "未通过"}.get(
                record.verdict, record.verdict
            )
            lines.append(f"- 第 {record.attempt_index} 次（{verdict}）：{record.user_answer}")
        return "\n".join(lines)

    def _resolve_node(self, path: LearningPath, snapshot: SessionSnapshot) -> KnowledgePoint | None:
        """确定当前该学的节点。

        优先复用会话里记录的当前节点（前提是它还没进入终态），否则取路径中第一个未完成的节点。
        """
        if snapshot.current_node_id:
            node = self._repos.paths.node(snapshot.current_node_id)
            if node is not None and node.state not in TERMINAL_NODE_STATES:
                return node

        for candidate in path.ordered_nodes():
            if candidate.state not in TERMINAL_NODE_STATES:
                self._state.enter_node(path.subject_id, candidate)
                return candidate
        return None

    def _next_checkpoint(self, node: KnowledgePoint) -> Checkpoint | None:
        """该节点第一个尚未通过的检查点。

        以 AnswerRecord 为准而不是"记住上次问到哪"，这样刷新、崩溃重启后都能正确续上。
        """
        passed = {
            record.checkpoint_id
            for record in self._repos.answers.list_for_node(node.id)
            if record.verdict == Verdict.PASS.value
        }
        for checkpoint in sorted(node.checkpoints, key=lambda item: item.order_index):
            if checkpoint.id not in passed:
                return checkpoint
        return None

    def _last_checkpoint(self, node: KnowledgePoint) -> Checkpoint | None:
        return self._next_checkpoint(node) or (
            max(node.checkpoints, key=lambda item: item.order_index) if node.checkpoints else None
        )

    async def _judge(
        self,
        node: KnowledgePoint,
        checkpoint: Checkpoint,
        answer_text: str,
        session: ConversationSession,
    ) -> tuple[JudgementResult | None, str | None]:
        """独立判定（FR-405）。返回 (结果, 失败说明)。"""
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
            self._log.warning(f"判定不可用: {exc.user_message}")
            return None, "这次没能完成判定，请再提交一次回答（你的作答没有丢失）。"
        return result, None

    def _master_node(
        self,
        subject_id: str,
        node: KnowledgePoint,
        *,
        summary: str = "",
    ) -> None:
        self._repos.paths.update_node(
            node.id,
            state=NodeState.MASTERED,
            summary_note=summary or None,
        )
        if summary:
            self._repos.notes.upsert(subject_id, node.id, summary, source="auto")
        # 复习通过后薄弱点自动关闭（FR-503 的前置语义）
        self._repos.weak_points.resolve(subject_id, node.id)
        self._state.clear_pending(subject_id)
        self._state.clear_decision(subject_id)
        # 节点已完成，上一条回复不再属于当前进度
        self._state.clear_last_action(subject_id)
        self._log.info(f"节点已掌握 subject={subject_id} node={node.id}")

    def _mark_node_partial(self, subject_id: str, node: KnowledgePoint, *, reason: WeakPointReason) -> None:
        self._repos.paths.update_node(node.id, state=NodeState.PARTIAL)
        self._repos.weak_points.record(subject_id, node.id, reason)
        self._log.info(f"节点标记为部分掌握 subject={subject_id} node={node.id} reason={reason.value}")

    def _finish_subject(self, subject_id: str) -> None:
        snapshot = self._state.snapshot(subject_id)
        if snapshot.state is SubjectState.TEACHING:
            self._state.transition(subject_id, SubjectState.COMPLETED)
            self._log.info(f"全部节点完成，主题标记为已完成 subject={subject_id}")

    def _counter(self, path: LearningPath, node: KnowledgePoint | None) -> _Counter:
        ordered = path.ordered_nodes()
        index = next((i for i, item in enumerate(ordered) if node is not None and item.id == node.id), -1)
        return _Counter(
            node_index=index + 1 if index >= 0 else 0,
            node_total=len(ordered),
        )

    def _pass_text(self, summary: str) -> str:
        base = "答对了，这个知识点你已经掌握。"
        return f"{base}\n\n{summary}" if summary else base

    @staticmethod
    def _set_prompt(session: ConversationSession, prompt: str) -> None:
        """把本轮提示词写进会话的首条 system 消息。

        ``RoundExecutor`` 只在记忆为空时接受 ``system_prompt``，因此后续轮次必须**就地改写
        首条 system 消息**，否则重讲/参考答案的提示词会被静默丢弃——模型拿到的还是上一轮的提示。
        """
        ensure_leading_system(session.memory, prompt)

    @staticmethod
    def _emit(on_delta: DeltaCallback | None, text: str) -> None:
        """把非流式产生的文本也推给前端，让界面只需"追加收到的一切"。"""
        if on_delta is not None and text:
            on_delta(StreamEvent(content=text))

    def _require_subject(self, subject_id: str) -> Subject:
        subject = self._repos.subjects.get(subject_id)
        if subject is None:
            raise KeyError(f"主题不存在: {subject_id}")
        return subject

    def _require_node(self, node_id: str | None) -> KnowledgePoint:
        if not node_id:
            raise TeachingError("当前没有进行中的节点，请点击「继续学习」。")
        node = self._repos.paths.node(node_id)
        if node is None:
            raise TeachingError("当前节点已不存在，请重新生成学习路径。")
        return node

    def _require_path(self, subject_id: str) -> LearningPath:
        path = self._planning.current_path(subject_id)
        if path is None:
            raise TeachingError("还没有学习路径，请先生成并确认。")
        return path

    def _require_teaching_state(self, subject_id: str) -> None:
        snapshot = self._state.snapshot(subject_id)
        if snapshot.state is SubjectState.PAUSED:
            raise AnswerNotAllowedError("学习已暂停，请先点击「继续学习」。", reason="paused")
        if snapshot.state is SubjectState.COMPLETED:
            raise AnswerNotAllowedError("本次学习已完成。可在薄弱点复习里继续练习。", reason="completed")
        if snapshot.state is not SubjectState.TEACHING:
            raise AnswerNotAllowedError(
                "学习还没开始，请先完成知识基础确认并确认学习路径。",
                reason="wrong_stage",
            )


def _explain_instruction(node_index: int, node_total: int) -> str:
    return f"（系统提示：请讲解第 {node_index}/{node_total} 个节点。不要提问。）"
