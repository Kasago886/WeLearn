"""学习领域服务装配（p2-api-sse）。

把 P2A 的全部服务按依赖顺序装成一个对象，API 层只与它打交道。
好处是依赖关系只在这里出现一次，测试也可以直接用同一份装配。
"""

from collections.abc import Callable
from dataclasses import dataclass

from app.contracts.schemas import ConversationKind
from app.infra.settings import AppSettings
from app.main_context import AppContext
from app.services.learning.assessment import AssessmentService
from app.services.learning.planning import PlanningService
from app.services.learning.repository import DomainRepositories
from app.services.learning.review import ReviewService
from app.services.learning.session import ConversationSessionFactory
from app.services.learning.state import SessionStateService
from app.services.learning.subjects import SubjectService
from app.services.learning.teaching import TeachingService
from app.services.llm.memory import visible_messages
from app.services.llm.policy import ModelPolicy
from app.services.llm.repository import MemoryRepository
from app.services.llm.usage import UsageRecorder


@dataclass
class LearningStack:
    """领域服务集合。"""

    repos: DomainRepositories
    state: SessionStateService
    subjects: SubjectService
    assessments: AssessmentService
    planning: PlanningService
    teaching: TeachingService
    reviews: ReviewService
    usage: UsageRecorder
    policy: ModelPolicy
    settings_provider: Callable[[], AppSettings]
    memories: MemoryRepository

    def refresh_settings(self, settings: AppSettings) -> None:
        """设置变更后同步用途策略（模型与思考档位）。"""
        self.policy.update_settings(settings)

    def transcript(self, subject_id: str, kind: ConversationKind, *, limit: int | None = None) -> list[dict[str, str]]:
        """只读取回某用途的对话正文（刷新恢复用）。

        刻意不用 ``ConversationSessionFactory.open``：它是"打开会话"的语义，会顺手建出会话行，
        而这是一个读接口，不该有副作用。
        """
        conversation_id = self.memories.conversation_id_for(subject_id, kind)
        if conversation_id is None:
            return []
        return visible_messages(self.memories.load(conversation_id), limit=limit)


def build_learning_stack(ctx: AppContext) -> LearningStack:
    """按环境装配学习领域服务。"""
    repos = DomainRepositories(ctx.db)
    settings_provider: Callable[[], AppSettings] = lambda: ctx.settings  # noqa: E731
    policy = ModelPolicy(ctx.settings)

    # 用量记录器不绑定主题，主题 id 在会话打开时按调用传入
    usage = UsageRecorder(ctx.db)
    sessions = ConversationSessionFactory(
        ctx.llm,
        ctx.memories,
        usage,
        policy,
        max_retries=ctx.settings.max_retries,
    )

    state = SessionStateService(repos.sessions, repos.subjects)
    subjects = SubjectService(repos)
    assessments = AssessmentService(repos, state, sessions, settings_provider)
    planning = PlanningService(repos, state, sessions, assessments, settings_provider, policy)
    teaching = TeachingService(repos, state, sessions, planning, assessments, settings_provider)
    reviews = ReviewService(repos, state, sessions, planning)

    return LearningStack(
        repos=repos,
        state=state,
        subjects=subjects,
        assessments=assessments,
        planning=planning,
        teaching=teaching,
        reviews=reviews,
        usage=usage,
        policy=policy,
        settings_provider=settings_provider,
        memories=ctx.memories,
    )
