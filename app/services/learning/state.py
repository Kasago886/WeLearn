"""会话状态机（T2.2，FR-410）。

**为什么要有这一层**：状态散落在各处用 ``if`` 判断，就会出现"从 draft 直接进 teaching"
这类脏状态，而且没人说得清当前到底允许做什么。这里把"允许什么"集中成一张迁移表，
业务代码只能通过 :meth:`SessionStateService.transition` 改状态。

两条贯穿全局的守卫：

1. **作答守卫**：只有 ``teaching`` 且存在待回答的检查点时才接受作答。
   否则用户随手发一句话就会被当成检查点答案喂给判定器，污染作答记录与节点状态；
2. **前进守卫**：``awaiting_decision=True`` 时禁止提交新作答——必须先把三选一做完，
   否则用户会被"重新开始新一轮判定"而不是看到选项（FR-408 的核心）。
"""

from collections.abc import Callable
from dataclasses import dataclass

from app.contracts.schemas import SubjectState
from app.infra.logging.queue_logger import CHANNEL_APP, get_logger
from app.services.learning.entities import KnowledgePoint, SessionSnapshot, Subject
from app.services.learning.repository import SessionStateRepository, SubjectRepository

NodeLoader = Callable[[str], "KnowledgePoint | None"]
"""节点加载器：由路径仓储提供，避免本模块直接依赖路径仓储。"""

ALLOWED_TRANSITIONS: dict[SubjectState, frozenset[SubjectState]] = {
    SubjectState.DRAFT: frozenset({SubjectState.ASSESSING, SubjectState.PATH_REVIEW}),
    # 测评只能走向"待确认路径"；重新测评由 path_review → assessing 承担
    SubjectState.ASSESSING: frozenset({SubjectState.PATH_REVIEW}),
    SubjectState.PATH_REVIEW: frozenset({SubjectState.ASSESSING, SubjectState.TEACHING}),
    # 教学态可以暂停、可以回到路径重新规划、也可以直接完成（用户提前结束）
    SubjectState.TEACHING: frozenset({SubjectState.PAUSED, SubjectState.COMPLETED, SubjectState.PATH_REVIEW}),
    SubjectState.PAUSED: frozenset({SubjectState.TEACHING, SubjectState.COMPLETED}),
    # 完成后仍可继续学习（复习/重学）
    SubjectState.COMPLETED: frozenset({SubjectState.TEACHING}),
}

ANSWERING_STATES = frozenset({SubjectState.TEACHING})
"""允许提交作答的状态。暂停与完成态都必须拒绝，并给出可操作提示。"""


class TransitionError(RuntimeError):
    """非法状态迁移。"""

    def __init__(self, current: SubjectState, target: SubjectState) -> None:
        allowed = "、".join(sorted(state.value for state in ALLOWED_TRANSITIONS[current]))
        super().__init__(f"当前状态为 {current.value}，不能切换到 {target.value}（允许: {allowed}）")
        self.current = current
        self.target = target


class AnswerNotAllowedError(RuntimeError):
    """当前不接受作答。"""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ResumeView:
    """刷新恢复视图（FR-410）：前端据此决定展示什么。

    它把"该显示输入框还是显示三个选项"这类判断放在服务端，
    避免前端各自猜一套规则后行为不一致。
    """

    snapshot: SessionSnapshot
    subject: Subject
    node: KnowledgePoint | None

    @property
    def state(self) -> SubjectState:
        return self.snapshot.state

    @property
    def can_answer(self) -> bool:
        """是否应展示作答输入框。"""
        return self.snapshot.state in ANSWERING_STATES and self.snapshot.has_pending_question

    @property
    def needs_decision(self) -> bool:
        """是否应展示三选一。"""
        return self.snapshot.state in ANSWERING_STATES and self.snapshot.needs_user_decision

    @property
    def next_action(self) -> str:
        """下一步该做什么，供前端直接使用。

        取值：``assess``（待测评）/ ``confirm_path``（待确认路径）/ ``teach``（待讲解）/
        ``answer``（待作答）/ ``decide``（待三选一）/ ``completed``（已完成）/ ``paused``（已暂停）。
        """
        state = self.snapshot.state
        if state is SubjectState.DRAFT:
            return "assess"
        if state in {SubjectState.ASSESSING, SubjectState.PATH_REVIEW}:
            return "confirm_path" if state is SubjectState.PATH_REVIEW else "assess"
        if state is SubjectState.PAUSED:
            return "paused"
        if state is SubjectState.COMPLETED:
            return "completed"
        if self.snapshot.needs_user_decision:
            return "decide"
        if self.snapshot.has_pending_question:
            return "answer"
        return "teach"


class SessionStateService:
    """状态迁移与恢复视图。"""

    def __init__(
        self,
        sessions: SessionStateRepository,
        subjects: SubjectRepository,
    ) -> None:
        self._sessions = sessions
        self._subjects = subjects
        self._log = get_logger(CHANNEL_APP)

    # ------------------------------------------------------------------ #
    # 读取
    # ------------------------------------------------------------------ #

    def snapshot(self, subject_id: str) -> SessionSnapshot:
        self._require_subject(subject_id)
        return self._sessions.ensure(subject_id)

    def resume(self, subject_id: str, *, node_loader: NodeLoader | None = None) -> ResumeView:
        """拼装恢复视图。``node_loader`` 用于加载当前节点（避免本模块依赖路径仓储）。"""
        subject = self._require_subject(subject_id)
        snapshot = self._sessions.ensure(subject_id)
        node = None
        if node_loader is not None and snapshot.current_node_id:
            node = node_loader(snapshot.current_node_id)
        return ResumeView(snapshot=snapshot, subject=subject, node=node)

    # ------------------------------------------------------------------ #
    # 迁移
    # ------------------------------------------------------------------ #

    def transition(self, subject_id: str, target: SubjectState) -> SessionSnapshot:
        """迁移状态；非法迁移抛 :class:`TransitionError`。

        目标与当前相同视为**幂等空操作**（例如"重新生成路径"时重复进入 ``path_review``），
        不报错也不写库。
        """
        current = self._sessions.ensure(subject_id)

        if current.state is target:
            return current

        if target not in ALLOWED_TRANSITIONS[current.state]:
            self._log.warning(f"拒绝非法状态迁移: {current.state.value} → {target.value}")
            raise TransitionError(current.state, target)

        snapshot = self._sessions.update(subject_id, state=target)
        self._subjects.set_status(subject_id, target)
        self._log.info(f"状态迁移 {current.state.value} → {target.value}（subject={subject_id}）")
        return snapshot

    def can_transition(self, subject_id: str, target: SubjectState) -> bool:
        current = self._sessions.ensure(subject_id)
        return current.state is target or target in ALLOWED_TRANSITIONS[current.state]

    # ------------------------------------------------------------------ #
    # 节点与待回答问题
    # ------------------------------------------------------------------ #

    def enter_node(self, subject_id: str, node: KnowledgePoint) -> SessionSnapshot:
        """进入某节点：切换当前节点、清零尝试次数、清空待回答与待决策标记。

        同时把 ``taught_node_id`` 清空——新节点尚未讲解，教学服务据此决定是否要生成讲解。
        """
        return self._sessions.update(
            subject_id,
            current_node_id=node.id,
            node_attempts=0,
            taught_node_id=None,
            pending_checkpoint_id=None,
            pending_question=None,
            awaiting_decision=False,
        )

    def mark_taught(self, subject_id: str) -> SessionSnapshot:
        """标记"当前节点的讲解已生成"（防止恢复后重复讲一遍）。"""
        snapshot = self._sessions.ensure(subject_id)
        return self._sessions.update(subject_id, taught_node_id=snapshot.current_node_id)

    def set_pending_question(
        self,
        subject_id: str,
        *,
        checkpoint_id: str,
        question: str,
    ) -> SessionSnapshot:
        """登记待回答的检查点。"""
        return self._sessions.update(
            subject_id,
            pending_checkpoint_id=checkpoint_id,
            pending_question=question,
            awaiting_decision=False,
        )

    def clear_pending(self, subject_id: str) -> SessionSnapshot:
        return self._sessions.clear_pending(subject_id)

    def request_decision(self, subject_id: str) -> SessionSnapshot:
        """进入"等用户三选一"：清空待回答并置位 awaiting_decision。"""
        self._sessions.update(subject_id, pending_checkpoint_id=None, pending_question=None)
        return self._sessions.update(subject_id, awaiting_decision=True)

    def clear_decision(self, subject_id: str) -> SessionSnapshot:
        return self._sessions.update(subject_id, awaiting_decision=False)

    def increment_attempts(self, subject_id: str) -> int:
        """本轮节点的尝试次数 +1，返回新值。"""
        snapshot = self._sessions.ensure(subject_id)
        updated = self._sessions.update(subject_id, node_attempts=snapshot.node_attempts + 1)
        return updated.node_attempts

    def set_attempts(self, subject_id: str, value: int) -> SessionSnapshot:
        """直接设置尝试次数（用于"判定失败不计入尝试"这类回退场景）。"""
        return self._sessions.update(subject_id, node_attempts=max(0, value))

    # ------------------------------------------------------------------ #
    # 守卫
    # ------------------------------------------------------------------ #

    def require_answering_state(self, subject_id: str) -> SessionSnapshot:
        """要求当前处于可作答状态，否则抛 :class:`AnswerNotAllowedError`。"""
        snapshot = self._sessions.ensure(subject_id)

        if snapshot.state is SubjectState.PAUSED:
            raise AnswerNotAllowedError(
                "学习已暂停，请先点击「继续学习」再作答。",
                reason="paused",
            )
        if snapshot.state is SubjectState.COMPLETED:
            raise AnswerNotAllowedError(
                "本次学习已完成。可在薄弱点复习中继续练习。",
                reason="completed",
            )
        if snapshot.state not in ANSWERING_STATES:
            raise AnswerNotAllowedError(
                f"当前处于{_STATE_LABELS.get(snapshot.state, snapshot.state.value)}阶段，还不能提交作答。",
                reason="wrong_stage",
            )
        return snapshot

    def require_pending_question(self, subject_id: str) -> tuple[SessionSnapshot, str, str]:
        """要求"有正在等待回答的检查点"，返回 (快照, checkpoint_id, question)。

        这是本模块最重要的守卫：没有它，用户在空闲时打的任何字都会被当作答案。
        """
        snapshot = self.require_answering_state(subject_id)

        if snapshot.needs_user_decision:
            raise AnswerNotAllowedError(
                "当前需要你先选择「继续重讲 / 查看参考答案 / 直接进入下一节点」。",
                reason="awaiting_decision",
            )
        if not snapshot.has_pending_question:
            raise AnswerNotAllowedError(
                "当前没有等待回答的问题，请先点击「开始/继续学习」让老师提问。",
                reason="no_pending_question",
            )

        assert snapshot.pending_checkpoint_id is not None
        assert snapshot.pending_question is not None
        return snapshot, snapshot.pending_checkpoint_id, snapshot.pending_question

    def require_decision_pending(self, subject_id: str) -> SessionSnapshot:
        """要求当前确实在等用户三选一（防止前端乱发决策请求）。"""
        snapshot = self.require_answering_state(subject_id)
        if not snapshot.needs_user_decision:
            raise AnswerNotAllowedError(
                "当前不需要做出选择。",
                reason="decision_not_expected",
            )
        return snapshot

    # ------------------------------------------------------------------ #

    def _require_subject(self, subject_id: str) -> Subject:
        subject = self._subjects.get(subject_id)
        if subject is None:
            raise KeyError(f"主题不存在: {subject_id}")
        return subject


_STATE_LABELS: dict[SubjectState, str] = {
    SubjectState.DRAFT: "创建",
    SubjectState.ASSESSING: "知识基础确认",
    SubjectState.PATH_REVIEW: "路径确认",
    SubjectState.TEACHING: "教学",
    SubjectState.PAUSED: "暂停",
    SubjectState.COMPLETED: "完成",
}
