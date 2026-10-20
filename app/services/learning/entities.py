"""领域读模型（T2.1）。

**为什么不让服务层直接用 SQLAlchemy 行对象**：行对象带着会话状态，一旦跨出
``with db.session()`` 就会触发懒加载或 DetachedInstanceError；而领域服务需要在事务外
自由传递数据（例如拼装 SSE 载荷）。因此仓储统一把行转换成不可变 dataclass。
"""

from dataclasses import dataclass, field
from datetime import datetime

from app.contracts.schemas import (
    AssessmentStatus,
    MasteryLevel,
    NodeState,
    SubjectState,
    WeakPointReason,
)

TERMINAL_NODE_STATES = frozenset({NodeState.MASTERED, NodeState.PARTIAL, NodeState.SKIPPED})
"""节点已"走完"的状态：教学推进时跳过它们。"""

FRESH_NODE_STATES = frozenset({NodeState.NOT_STARTED, NodeState.REVIEW_NEEDED})
"""尚未开始学习的节点：进入节点时从这些状态改成 ``learning``（FR-309）。"""

MOVABLE_NODE_STATES = FRESH_NODE_STATES
"""允许被删除、调整顺序、单节点重生成的状态。

``learning``（正在学）与三个终态都不能动：把"已掌握"挪到别的位置，等于让掌握状态
落在另一个知识点上；删除正在学的节点会让会话指向一个不存在的节点。
放在本模块是因为路径编辑（``planning``）与教学（``teaching``）都要用它，
而 ``teaching`` 依赖 ``planning``——从 teaching 导入会形成循环。
"""

ANSWER_SOURCE_TEACH = "teach"
ANSWER_SOURCE_REVIEW = "review"
"""作答来源：教学 / 复习（FR-507）。

复习作答与教学作答写在同一张表里（都带节点与检查点），靠这个字段区分。
不这样做的话，"这个节点复习了几次、结果如何"就无从统计。
"""

REVIEW_MODE_ADJACENT = "adjacent"
REVIEW_MODE_WEAK = "weak"
"""复习会话的两种来源。

``adjacent``：进入新节点前回顾上一节点（FR-414），答对即过、答错回到上一节点补强；
``weak``：从薄弱点面板发起的针对性复习（FR-503），通过则解除该薄弱点。
两者提问与判定机制相同，但**通过后的后果不同**，所以必须区分。
"""


@dataclass(frozen=True)
class Subject:
    """学习主题（FR-101~FR-106）。"""

    id: str
    title: str
    goal: str | None
    background: str | None
    depth: str | None
    status: SubjectState
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None = None

    @property
    def archived(self) -> bool:
        return self.archived_at is not None


@dataclass(frozen=True)
class SubjectProgress:
    """主题进度概览（FR-106）。"""

    total_nodes: int = 0
    mastered_nodes: int = 0
    partial_nodes: int = 0
    skipped_nodes: int = 0
    weak_points: int = 0
    total_tokens: int = 0

    @property
    def mastery_ratio(self) -> float:
        return self.mastered_nodes / self.total_nodes if self.total_nodes else 0.0


@dataclass(frozen=True)
class SubjectSummary:
    """主题 + 进度，用于列表页（FR-103、FR-106）。"""

    subject: Subject
    progress: SubjectProgress
    last_studied_at: datetime | None = None


@dataclass(frozen=True)
class ReviewDueStat:
    """主题维度的待复习统计（FR-505），只用于主题列表。

    ``due``：已排期且到期的节点数；``stale``：从未排期、但最近触达已超过阈值的
    节点数（历史数据在第一次打开复习页前就是这个状态）。两者都算"需要复习"，
    否则老主题在列表上永远不会被提醒。
    """

    due: int = 0
    stale: int = 0

    @property
    def due_reviews(self) -> int:
        return self.due + self.stale

    @property
    def needs_review(self) -> bool:
        return self.due_reviews > 0


@dataclass(frozen=True)
class SessionSnapshot:
    """状态机快照：刷新恢复的唯一数据来源（FR-410）。"""

    subject_id: str
    state: SubjectState
    current_node_id: str | None = None
    pending_checkpoint_id: str | None = None
    pending_question: str | None = None
    node_attempts: int = 0
    taught_node_id: str | None = None
    awaiting_decision: bool = False
    last_action: str | None = None
    """最近一次教学动作（``explain`` / ``reteach:simpler`` / ``hint:2`` …）。

    "重新生成上一条回复"（FR-413）必须知道上一步做了什么，否则无从重跑。
    """
    review_mode: str | None = None
    """复习会话的类型：``adjacent``（进入新节点前回顾上一节点，FR-414）/
    ``weak``（针对薄弱点的复习，FR-503）。为空表示当前不在复习。

    与教学状态**分开存**：复习旧节点不该挪动 ``current_node_id``。
    """
    review_node_id: str | None = None
    review_checkpoint_id: str | None = None
    adjacent_review_for: str | None = None
    """为进入哪个节点已经做过"回顾上一节点"（FR-414）。

    没有它，用户刷新后点「继续学习」会被重复问一遍上一节点。
    ``enter_node`` 会清空它，因此每个节点的进入只会回顾一次。
    """
    updated_at: datetime | None = None

    @property
    def reviewing(self) -> bool:
        """是否正在进行一次复习会话。"""
        return self.review_mode is not None and self.review_node_id is not None

    @property
    def has_review_question(self) -> bool:
        """复习是否停在"等用户回答"这一步。"""
        return self.reviewing and self.review_checkpoint_id is not None

    @property
    def has_pending_question(self) -> bool:
        """是否正在等待用户回答某个检查点。

        这是"允许提交作答"的唯一前置条件——为空时必须拒绝作答，
        否则用户随手一句会被当成答案喂给判定器（FR-410 的核心不变量）。
        """
        return bool(self.pending_question) and bool(self.pending_checkpoint_id)

    @property
    def node_taught(self) -> bool:
        """当前节点是否已生成过讲解（避免恢复后重复讲一遍）。"""
        return self.current_node_id is not None and self.taught_node_id == self.current_node_id

    @property
    def needs_user_decision(self) -> bool:
        """是否正在等用户就"继续重讲 / 看参考答案 / 直接进入下一节点"做出选择（FR-408）。"""
        return self.awaiting_decision


@dataclass(frozen=True)
class Assessment:
    """一次知识基础确认（FR-201~FR-208）。"""

    id: str
    subject_id: str
    status: AssessmentStatus
    turn_count: int
    max_turns: int
    topics: tuple[str, ...] = ()
    summary: str | None = None
    suggested_start_node: str | None = None
    skipped: bool = False
    completed_at: datetime | None = None

    @property
    def finished(self) -> bool:
        return self.status in {AssessmentStatus.COMPLETED, AssessmentStatus.SKIPPED}

    @property
    def turns_left(self) -> int:
        return max(0, self.max_turns - self.turn_count)


@dataclass(frozen=True)
class ProfileLevel:
    """画像中的一行：子领域 × 水平（FR-206）。"""

    topic: str
    level: MasteryLevel
    evidence: str
    user_overridden: bool = False


@dataclass(frozen=True)
class Checkpoint:
    """检查点（FR-302）。"""

    id: str
    node_id: str
    order_index: int
    question: str
    expected_points: tuple[str, ...]
    reference_answer: str


@dataclass(frozen=True)
class KnowledgePoint:
    """知识点 / 教学节点（FR-302、FR-309）。"""

    id: str
    path_id: str
    order_index: int
    title: str
    objective: str
    prerequisites: tuple[str, ...]
    key_points: tuple[str, ...]
    state: NodeState
    attempts: int = 0
    is_quick_review: bool = False
    mastered_at: datetime | None = None
    last_reviewed_at: datetime | None = None
    review_box: int = 0
    """间隔复习的 Leitner 箱号（FR-506）。0 表示"还没排过期"。"""
    review_due_at: datetime | None = None
    """下一次该复习的时间（UTC naive）。为空即"未排期"，由复习页懒补齐。"""
    summary_note: str | None = None
    checkpoints: tuple[Checkpoint, ...] = ()


@dataclass(frozen=True)
class LearningPath:
    """学习路径（FR-301~FR-308）。"""

    id: str
    subject_id: str
    version: int
    rationale: str
    model: str | None
    is_active: bool
    created_at: datetime
    nodes: tuple[KnowledgePoint, ...] = field(default_factory=tuple)
    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def node_by_id(self, node_id: str) -> KnowledgePoint | None:
        return next((node for node in self.nodes if node.id == node_id), None)

    def ordered_nodes(self) -> tuple[KnowledgePoint, ...]:
        return tuple(sorted(self.nodes, key=lambda node: node.order_index))


@dataclass(frozen=True)
class AnswerRecord:
    """作答与判定记录（FR-404、FR-802）。"""

    id: int
    subject_id: str
    node_id: str
    checkpoint_id: str | None
    question: str
    user_answer: str
    verdict: str
    confidence: float | None
    feedback: str
    hit_points: tuple[str, ...]
    missing_points: tuple[str, ...]
    attempt_index: int
    created_at: datetime
    source: str = "teach"
    """``teach`` = 教学作答；``review`` = 复习作答（FR-507）。

    档案页据此区分"首次学会"与"复习通过"，复习记录也靠它统计。
    """


@dataclass(frozen=True)
class WeakPoint:
    """薄弱点（FR-502~FR-504）。"""

    id: int
    subject_id: str
    node_id: str
    reason: WeakPointReason
    fail_count: int
    last_attempt_at: datetime
    resolved_at: datetime | None = None

    @property
    def resolved(self) -> bool:
        return self.resolved_at is not None


@dataclass(frozen=True)
class StudyNote:
    """节点笔记（FR-801）。"""

    id: str
    subject_id: str
    node_id: str | None
    content: str
    source: str
    updated_at: datetime
