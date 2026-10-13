"""学习路径规划服务（T2.4，FR-301~FR-309）。

流程：

```text
generate()   ← 需要画像（或"未测评"的明确标记）
  ├─ 结构化生成节点草稿（含 rationale 与每个节点的检查点）
  ├─ 校验节点数落在配置区间（不满足则带约束重试一次）
  ├─ 落库（旧路径自动置为非活动，保留版本历史）
  └─ 状态机 → path_review

confirm()    ← 确认门禁（FR-306）
  └─ 状态机 → teaching

编辑操作（FR-304）
  update_node() / add_node() / delete_node() / move_node() / regenerate_node()

regenerate() ← 允许在 teaching 态重新规划（FR-305）
  └─ 先转回 path_review，再走 generate

**编辑一律"读出 → 内存里改 → 整体写回"**：排序、插入、删除都要重排 ``order_index``，
对单行做增量 UPDATE 极易留下重复序号，而重复序号会让 ``next_node`` 随机挑到一个节点。
写回时**复用原有的节点与检查点 id**，否则挂在 id 上的作答记录、笔记、薄弱点会变成孤儿。
```

**为什么节点数校验放在服务层而不是靠 pydantic**：schema 只能表达"3~15"这个绝对边界，
而用户配置的区间（如 5~8）是运行时参数。把区间写进 schema 需要动态建模，反而更难读；
这里用一个显式的两轮循环，并把"必须落在 N~M"回灌到提示里，行为直观且可测试。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from app.contracts.schemas import (
    ConversationKind,
    LearningPathDraft,
    LlmPurpose,
    MasteryLevel,
    NodeState,
    PathNodeDraft,
    SubjectState,
)
from app.infra.logging.queue_logger import CHANNEL_APP, get_logger
from app.infra.settings import AppSettings
from app.prompts import PLANNER, PLANNER_NODE, render
from app.services.learning.assessment import AssessmentService
from app.services.learning.entities import (
    MOVABLE_NODE_STATES,
    KnowledgePoint,
    LearningPath,
    ProfileLevel,
    Subject,
)
from app.services.learning.repository import DomainRepositories
from app.services.learning.session import ConversationSessionFactory
from app.services.learning.state import SessionStateService
from app.services.llm.policy import ModelPolicy
from app.services.llm.result import RoundResult
from app.services.llm.structured import StructuredOutputError

MAX_GENERATION_ATTEMPTS = 2
NO_ASSESSMENT_NOTICE = "尚未进行知识基础确认，按零基础从最基础的方面开始安排。"


class PlanningError(RuntimeError):
    """路径规划流程错误（消息面向用户）。"""


@dataclass
class PathGenerationResult:
    """一次路径生成的产出。"""

    path: LearningPath | None = None
    rationale: str = ""
    notice: str = ""
    error_message: str = ""
    error_category: str | None = None
    retryable: bool = False
    result: RoundResult | None = None

    @property
    def ok(self) -> bool:
        return self.path is not None and not self.error_message


@dataclass
class NodeRegenerationResult:
    """单节点重生成的产出（FR-304）。"""

    path: LearningPath | None = None
    rationale: str = ""
    error_message: str = ""
    error_category: str | None = None
    retryable: bool = False

    @property
    def ok(self) -> bool:
        return self.path is not None and not self.error_message


@dataclass
class PathView:
    """路径页数据（FR-303、FR-308、FR-309）。"""

    path: LearningPath
    profile_levels: list[ProfileLevel] = field(default_factory=list)
    completed: bool = False

    @property
    def nodes(self) -> tuple[KnowledgePoint, ...]:
        return self.path.ordered_nodes()

    @property
    def current_index(self) -> int:
        """第一个尚未完成的节点序号（用于高亮"当前节点"）。"""
        pending = {NodeState.NOT_STARTED, NodeState.LEARNING, NodeState.REVIEW_NEEDED}
        for index, node in enumerate(self.nodes):
            if node.state in pending:
                return index
        return len(self.nodes)


class PlanningService:
    """路径生成、读取与确认。"""

    def __init__(
        self,
        repos: DomainRepositories,
        state: SessionStateService,
        sessions: ConversationSessionFactory,
        assessments: AssessmentService,
        settings_provider: Callable[[], AppSettings],
        policy: ModelPolicy,
    ) -> None:
        self._repos = repos
        self._state = state
        self._sessions = sessions
        self._assessments = assessments
        self._settings = settings_provider
        self._policy = policy
        self._log = get_logger(CHANNEL_APP)

    # ------------------------------------------------------------------ #
    # 读取
    # ------------------------------------------------------------------ #

    def current_path(self, subject_id: str) -> LearningPath | None:
        return self._repos.paths.get_active(subject_id)

    def view(self, subject_id: str) -> PathView | None:
        path = self.current_path(subject_id)
        if path is None:
            return None
        return PathView(
            path=path,
            profile_levels=self._assessments.levels(subject_id),
            completed=self.all_nodes_done(path),
        )

    def versions(self, subject_id: str) -> list[tuple[str, int, bool, datetime]]:
        """历史版本（FR-307）。"""
        return self._repos.paths.list_versions(subject_id)

    @staticmethod
    def all_nodes_done(path: LearningPath) -> bool:
        """所有节点都进入终态（已掌握/部分掌握/已跳过）即视为学习完成。"""
        terminal = {NodeState.MASTERED, NodeState.PARTIAL, NodeState.SKIPPED}
        nodes = path.ordered_nodes()
        return bool(nodes) and all(node.state in terminal for node in nodes)

    # ------------------------------------------------------------------ #
    # 生成
    # ------------------------------------------------------------------ #

    async def generate(self, subject_id: str) -> PathGenerationResult:
        """生成学习路径。"""
        subject = self._require_subject(subject_id)
        snapshot = self._state.snapshot(subject_id)

        # 教学态重新规划：先回到 path_review，保持状态机语义一致
        if snapshot.state is SubjectState.TEACHING:
            self._state.transition(subject_id, SubjectState.PATH_REVIEW)

        if snapshot.state is SubjectState.DRAFT:
            raise PlanningError("请先完成知识基础确认（或点击「跳过测评」）再生成学习路径。")
        if snapshot.state is SubjectState.ASSESSING:
            raise PlanningError("知识基础确认还没结束，请先完成它。")
        if snapshot.state is SubjectState.COMPLETED:
            raise PlanningError("本次学习已完成。如需重新规划，请先点击「继续学习」。")

        return await self._generate_with(subject, requirement="")

    async def _generate_with(self, subject: Subject, *, requirement: str) -> PathGenerationResult:
        """路径生成的主体（`generate` 与 `regenerate` 共用）。"""
        settings = self._settings()
        profile_text, notice = self._profile_text(subject.id)
        min_nodes = settings.path_min_nodes
        max_nodes = settings.path_max_nodes
        session = self._sessions.open(subject.id, ConversationKind.TEACH)

        if requirement.strip():
            profile_text += f"\n\n学习者的额外要求（必须满足）：{requirement.strip()}"

        draft: LearningPathDraft | None = None
        last_problem = ""

        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            try:
                candidate, _usage = await session.structured.request(
                    LlmPurpose.PLAN,
                    system=render(
                        PLANNER,
                        concept=subject.title,
                        goal=subject.goal or "（未填写）",
                        depth=subject.depth or "（未填写）",
                        profile=profile_text,
                        min_nodes=min_nodes,
                        max_nodes=max_nodes,
                    ),
                    user="请按上述要求输出 json 学习路径。",
                    schema=LearningPathDraft,
                )
            except StructuredOutputError as exc:
                self._log.warning(
                    f"路径生成失败（第 {attempt} 次）: {exc.user_message}；"
                    f"状态保持 {self._state.snapshot(subject.id).state.value}"
                )
                return PathGenerationResult(
                    notice=notice,
                    error_message="没能生成学习路径，请稍后重试。",
                    error_category="structured_output",
                    retryable=True,
                )

            node_count = len(candidate.nodes)
            if min_nodes <= node_count <= max_nodes:
                draft = candidate
                break

            last_problem = f"节点数 {node_count} 不在 {min_nodes}~{max_nodes} 之间"
            self._log.warning(f"路径节点数不合规（第 {attempt} 次）: {last_problem}")
            profile_text += (
                f"\n\n（上次生成的节点数为 {node_count}，必须改为 {min_nodes}~{max_nodes} 个节点。）"
            )

        if draft is None:
            return PathGenerationResult(
                notice=notice,
                error_message=f"模型给出的节点数量不符合设置（{last_problem}），请调整节点数量设置后重试。",
                error_category="node_count",
                retryable=True,
            )

        path = self._repos.paths.create(
            subject.id,
            rationale=draft.rationale,
            model=self._sessions_model_name(),
            nodes=[_node_payload(node) for node in draft.nodes],
        )
        self._state.transition(subject.id, SubjectState.PATH_REVIEW)

        quick = sum(1 for node in draft.nodes if node.is_quick_review)
        self._log.info(
            f"生成学习路径 subject={subject.id} version={path.version} "
            f"nodes={path.node_count} quick_review={quick}"
        )

        return PathGenerationResult(path=path, rationale=draft.rationale, notice=notice)

    # ------------------------------------------------------------------ #
    # 确认
    # ------------------------------------------------------------------ #

    def confirm(self, subject_id: str) -> LearningPath:
        """确认路径并进入教学（FR-306 的门禁）。"""
        path = self.current_path(subject_id)
        if path is None:
            raise PlanningError("还没有学习路径，请先生成。")

        self._state.transition(subject_id, SubjectState.TEACHING)
        self._log.info(f"用户确认学习路径 subject={subject_id} path={path.id} v{path.version}")
        return path

    # ------------------------------------------------------------------ #
    # 编辑（FR-304）
    # ------------------------------------------------------------------ #

    def _require_editable(self, subject_id: str) -> LearningPath:
        """取可编辑的当前路径。

        已开始的节点（``learning``/``mastered``）不允许被删掉或移动位置：
        删掉正在学的节点会让会话指向一个不存在的节点，移动已掌握的节点会打乱复习顺序。
        但**允许改标题与要点**——那是纠错，不是改教学进度。
        """
        path = self.current_path(subject_id)
        if path is None:
            raise PlanningError("还没有学习路径，请先生成。")
        return path

    def update_node(
        self,
        subject_id: str,
        node_id: str,
        *,
        title: str | None = None,
        objective: str | None = None,
        key_points: Sequence[str] | None = None,
        prerequisites: Sequence[str] | None = None,
    ) -> LearningPath:
        """改标题 / 目标 / 要点（FR-304）。"""
        path = self._require_editable(subject_id)
        self._require_node_in(path, node_id)

        payload: dict[str, object] = {}
        if title is not None:
            cleaned = title.strip()
            if not cleaned:
                raise PlanningError("节点标题不能为空。")
            payload["title"] = cleaned
        if objective is not None:
            payload["objective"] = objective.strip()
        if key_points is not None:
            payload["key_points"] = [item.strip() for item in key_points if item.strip()]
        if prerequisites is not None:
            payload["prerequisites"] = [item.strip() for item in prerequisites if item.strip()]

        if payload:
            self._repos.paths.replace_node(node_id, payload)
            self._log.info(f"编辑节点 subject={subject_id} node={node_id} fields={sorted(payload)}")

        return self._reload(path.id)

    def add_node(self, subject_id: str, *, after_node_id: str | None = None) -> LearningPath:
        """在指定节点之后插入一个空白节点（FR-304）。

        新节点刻意**留空**而不是让模型随便生成一个：用户点"插入节点"时心里通常已经有了
        要补的内容，直接给他一处可编辑的空位比塞一段不想要的讲解更符合意图。
        """
        path = self._require_editable(subject_id)

        nodes = [self._editable(node) for node in path.ordered_nodes()]
        position = len(nodes)
        if after_node_id is not None:
            self._require_node_in(path, after_node_id)
            position = next(i for i, node in enumerate(nodes) if node["id"] == after_node_id) + 1

        blank: dict[str, object] = {
            "title": "新的知识点",
            "objective": "（待填写：学完后能做到什么）",
            "prerequisites": [],
            "key_points": ["（待填写：要点一）"],
            "checkpoints": [
                {
                    "question": "（待填写：用于检验掌握的问题）",
                    "expected_points": ["（待填写：判定要点）"],
                    "reference_answer": "（待填写：参考答案）",
                }
            ],
            "is_quick_review": False,
        }
        nodes.insert(position, blank)
        self._repos.paths.replace_nodes(path.id, nodes)
        self._log.info(f"插入节点 subject={subject_id} position={position + 1}")

        return self._reload(path.id)

    def delete_node(self, subject_id: str, node_id: str) -> LearningPath:
        """删除节点（FR-304）。"""
        path = self._require_editable(subject_id)
        node = self._require_node_in(path, node_id)

        if node.state not in MOVABLE_NODE_STATES:
            raise PlanningError(
                f"「{node.title}」已经进入学习（当前状态：{node.state.value}），"
                "不能删除。可以先跳过它，或整条重新生成路径。"
            )
        if path.node_count <= 1:
            raise PlanningError("路径至少要保留一个节点。")

        nodes = [self._editable(item) for item in path.ordered_nodes() if item.id != node_id]
        self._repos.paths.replace_nodes(path.id, nodes)
        session = self._state.snapshot(subject_id)
        if session.current_node_id == node_id:
            # 当前节点被删掉了：清空指针，否则下一次进入教学会指向不存在的节点
            self._state.clear_pending(subject_id)
            self._repos.sessions.update(subject_id, current_node_id=None, taught_node_id=None)
        self._log.info(f"删除节点 subject={subject_id} node={node_id}（未开始）")

        return self._reload(path.id)

    def move_node(self, subject_id: str, node_id: str, *, offset: int) -> LearningPath:
        """上移 / 下移节点（FR-304）。``offset`` 为 -1 或 1。"""
        if offset not in {-1, 1}:
            raise PlanningError("移动方向只能是上移或下移。")

        path = self._require_editable(subject_id)
        nodes = [self._editable(node) for node in path.ordered_nodes()]
        index = next((i for i, node in enumerate(nodes) if node["id"] == node_id), None)
        if index is None:
            raise PlanningError("要移动的节点不存在。")

        target = index + offset
        if target < 0 or target >= len(nodes):
            raise PlanningError("已经到头了，不能再移动。")

        # 参与交换的两个节点都必须是"还没开始"的：把已掌握的节点挪到别的位置，
        # 会让它的掌握状态落在另一个知识点上，等于篡改了学习记录
        for position in (index, target):
            moving = path.ordered_nodes()[position]
            if moving.state not in MOVABLE_NODE_STATES:
                raise PlanningError(
                    f"「{moving.title}」已经进入学习（当前状态：{moving.state.value}），不能调整顺序。"
                )

        nodes[index], nodes[target] = nodes[target], nodes[index]
        self._repos.paths.replace_nodes(path.id, nodes)
        self._log.info(f"移动节点 subject={subject_id} node={node_id} {index + 1} → {target + 1}")

        return self._reload(path.id)

    async def regenerate_node(
        self,
        subject_id: str,
        node_id: str,
        *,
        requirement: str = "",
    ) -> NodeRegenerationResult:
        """重生成单个节点（FR-304），**保留节点 id**。

        保留 id 是有意的：作答记录、笔记、薄弱点都挂在 id 上，
        换 id 等于把这些记录变成孤儿。
        """
        path = self._require_editable(subject_id)
        node = self._require_node_in(path, node_id)
        if node.state not in MOVABLE_NODE_STATES:
            raise PlanningError(
                f"「{node.title}」已经学过了（当前状态：{node.state.value}）。"
                "重生成会换掉它的检查点，已产生的判定记录就失去依据，请先整条重新生成路径。"
            )

        subject = self._require_subject(subject_id)
        profile_text, _notice = self._profile_text(subject_id)
        session = self._sessions.open(subject_id, ConversationKind.TEACH)

        ordered = path.ordered_nodes()
        index = next(i for i, item in enumerate(ordered) if item.id == node_id)

        try:
            draft, _usage = await session.structured.request(
                LlmPurpose.PLAN,
                system=render(
                    PLANNER_NODE,
                    concept=subject.title,
                    goal=subject.goal or "（未填写）",
                    profile=profile_text,
                    neighbours=_neighbour_text(ordered, index),
                    node_index=index + 1,
                    node_total=len(ordered),
                    node_title=node.title,
                    objective=node.objective,
                    key_points=list(node.key_points),
                    requirement=requirement.strip() or "（无额外要求）",
                ),
                user="请按上述要求输出 json 节点。",
                schema=PathNodeDraft,
            )
        except StructuredOutputError as exc:
            self._log.warning(f"单节点重生成失败: {exc.user_message}")
            return NodeRegenerationResult(
                path=path,
                error_message="没能重新设计这个节点，请稍后重试。",
                error_category="structured_output",
                retryable=True,
            )

        self._repos.paths.replace_node(node_id, _node_payload(draft))
        self._log.info(f"重生成节点 subject={subject_id} node={node_id} title={draft.title}")

        return NodeRegenerationResult(path=self._reload(path.id), rationale=draft.title)

    async def regenerate(self, subject_id: str, *, requirement: str = "") -> PathGenerationResult:
        """整条重新生成（FR-305）：保留旧版本，可附加用户的额外要求。"""
        subject = self._require_subject(subject_id)
        self._state.transition(subject_id, SubjectState.PATH_REVIEW)

        # 记住已被引用的当前节点：重生成后节点全换了 id，指针必须清掉，
        # 否则"刷新恢复"会指向一个已经不存在的节点
        session = self._state.snapshot(subject_id)
        result = await self._generate_with(subject, requirement=requirement)
        if session.current_node_id is not None:
            self._state.clear_pending(subject_id)
            self._repos.sessions.update(
                subject_id, current_node_id=None, taught_node_id=None, node_attempts=0
            )
        return result

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _require_node_in(self, path: LearningPath, node_id: str) -> KnowledgePoint:
        node = next((item for item in path.ordered_nodes() if item.id == node_id), None)
        if node is None:
            raise PlanningError("节点不存在，可能路径已经被重新生成。")
        return node

    @staticmethod
    def _editable(node: KnowledgePoint) -> dict[str, object]:
        """把节点转成可写回的字典（含 id、状态与要点，供整体替换使用）。

        **必须带上节点与检查点的 id**：写回时是整表重写，若不带 id，``_insert_node``
        会给所有行生成新 id，于是作答记录、笔记、薄弱点，以及会话里的
        ``pending_checkpoint_id`` 全部指向不存在的行。
        """
        return {
            "id": node.id,
            "title": node.title,
            "objective": node.objective,
            "prerequisites": list(node.prerequisites),
            "key_points": list(node.key_points),
            "checkpoints": [
                {
                    "id": checkpoint.id,
                    "question": checkpoint.question,
                    "expected_points": list(checkpoint.expected_points),
                    "reference_answer": checkpoint.reference_answer,
                }
                for checkpoint in sorted(node.checkpoints, key=lambda item: item.order_index)
            ],
            "is_quick_review": node.is_quick_review,
            "state": node.state,
        }

    def _reload(self, path_id: str) -> LearningPath:
        path = self._repos.paths.get(path_id)
        if path is None:
            raise PlanningError("路径不存在。")
        return path

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    def _profile_text(self, subject_id: str) -> tuple[str, str]:
        """把画像整理成提示词可用的文本；未测评时给出明确标记。"""
        levels = self._assessments.levels(subject_id)
        if not levels:
            return NO_ASSESSMENT_NOTICE, "本次未进行知识基础确认，路径将按零基础安排。"

        view = self._assessments.view(subject_id)
        lines = [_level_line(level) for level in levels]
        text = "\n".join(lines)

        if view is not None and view.assessment.summary:
            text += f"\n\n总体评价：{view.assessment.summary}"
        if view is not None and view.assessment.suggested_start_node:
            text += f"\n建议起点：{view.assessment.suggested_start_node}"

        if view is not None and view.assessment.skipped:
            return text, "本次未进行知识基础确认，路径将按零基础安排。"

        return text, ""

    def _sessions_model_name(self) -> str | None:
        """记录生成路径时用的模型（便于回溯"为什么这条路径是这样"）。"""
        try:
            return str(self._policy.resolve(LlmPurpose.PLAN).model)
        except Exception:  # 取不到模型名不该让路径生成失败
            return None

    def _require_subject(self, subject_id: str) -> Subject:
        subject = self._repos.subjects.get(subject_id)
        if subject is None:
            raise KeyError(f"主题不存在: {subject_id}")
        return subject


_LEVEL_LABELS: dict[MasteryLevel, str] = {
    MasteryLevel.NONE: "未掌握",
    MasteryLevel.BASIC: "初步了解",
    MasteryLevel.PROFICIENT: "已熟练",
}


def _level_line(level: ProfileLevel) -> str:
    label = _LEVEL_LABELS.get(level.level, level.level.value)
    suffix = "（用户手动修正）" if level.user_overridden else ""
    return f"- {level.topic}：{label}{suffix} —— {level.evidence}"


def _node_payload(node: PathNodeDraft) -> dict[str, object]:
    """把 pydantic 节点转成仓储需要的字典。"""
    return {
        "title": node.title,
        "objective": node.objective,
        "prerequisites": list(node.prerequisites),
        "key_points": list(node.key_points),
        "checkpoints": [
            {
                "question": checkpoint.question,
                "expected_points": list(checkpoint.expected_points),
                "reference_answer": checkpoint.reference_answer,
            }
            for checkpoint in node.checkpoints
        ],
        "is_quick_review": node.is_quick_review,
    }


def _neighbour_text(nodes: Sequence[KnowledgePoint], index: int) -> str:
    """给单节点重生成提供"这个节点在路径中的位置"上下文。"""
    lines: list[str] = []
    if index > 0:
        previous = nodes[index - 1]
        lines.append(f"上一个节点：{previous.title} —— {previous.objective}")
    else:
        lines.append("上一个节点：（这是第一个节点）")

    if index + 1 < len(nodes):
        following = nodes[index + 1]
        lines.append(f"下一个节点：{following.title} —— {following.objective}")
    else:
        lines.append("下一个节点：（这是最后一个节点）")
    return "\n".join(lines)
