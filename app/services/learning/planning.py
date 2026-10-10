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

regenerate() ← 允许在 teaching 态重新规划（FR-305）
  └─ 先转回 path_review，再走 generate
```

**为什么节点数校验放在服务层而不是靠 pydantic**：schema 只能表达"3~15"这个绝对边界，
而用户配置的区间（如 5~8）是运行时参数。把区间写进 schema 需要动态建模，反而更难读；
这里用一个显式的两轮循环，并把"必须落在 N~M"回灌到提示里，行为直观且可测试。
"""

from collections.abc import Callable
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
from app.prompts import PLANNER, render
from app.services.learning.assessment import AssessmentService
from app.services.learning.entities import KnowledgePoint, LearningPath, ProfileLevel, Subject
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
        settings = self._settings()

        # 教学态重新规划：先回到 path_review，保持状态机语义一致
        snapshot = self._state.snapshot(subject_id)
        if snapshot.state is SubjectState.TEACHING:
            self._state.transition(subject_id, SubjectState.PATH_REVIEW)

        if snapshot.state is SubjectState.DRAFT:
            raise PlanningError("请先完成知识基础确认（或点击「跳过测评」）再生成学习路径。")
        if snapshot.state is SubjectState.ASSESSING:
            raise PlanningError("知识基础确认还没结束，请先完成它。")
        if snapshot.state is SubjectState.COMPLETED:
            raise PlanningError("本次学习已完成。如需重新规划，请先点击「继续学习」。")

        profile_text, notice = self._profile_text(subject_id)
        min_nodes = settings.path_min_nodes
        max_nodes = settings.path_max_nodes
        session = self._sessions.open(subject_id, ConversationKind.TEACH)

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
                    f"状态保持 {self._state.snapshot(subject_id).state.value}"
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
            subject_id,
            rationale=draft.rationale,
            model=self._sessions_model_name(),
            nodes=[_node_payload(node) for node in draft.nodes],
        )
        self._state.transition(subject_id, SubjectState.PATH_REVIEW)

        quick = sum(1 for node in draft.nodes if node.is_quick_review)
        self._log.info(
            f"生成学习路径 subject={subject_id} version={path.version} "
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
