"""学习领域测试脚手架。

把 P2A 的服务装配集中在一处，避免每个测试文件重复拼装依赖；同时让测试能
**用桩驱动完整流程**（不消耗真实额度、不受网络影响）。
"""

from dataclasses import dataclass, field

from app.infra.settings import AppSettings
from app.main_context import AppContext
from app.services.learning.repository import DomainRepositories
from app.services.learning.session import ConversationSessionFactory
from app.services.learning.state import SessionStateService
from app.services.learning.subjects import SubjectService
from app.services.llm.policy import ModelPolicy
from app.services.llm.usage import UsageRecorder
from tests.stubs import FakeLlmClient, ScriptedResponse


@dataclass
class LearningHarness:
    """一套可直接使用的领域服务 + 可控的 LLM 桩。"""

    ctx: AppContext
    client: FakeLlmClient
    settings: AppSettings
    repos: DomainRepositories
    state: SessionStateService
    subjects: SubjectService
    sessions: ConversationSessionFactory
    policy: ModelPolicy
    subjects_created: list[str] = field(default_factory=list)

    def new_subject(self, title: str = "贝叶斯定理", **kwargs: object) -> str:
        subject = self.subjects.create(title, **kwargs)  # type: ignore[arg-type]
        self.subjects_created.append(subject.id)
        return subject.id


def make_harness(
    ctx: AppContext,
    *,
    blocking: list[ScriptedResponse] | None = None,
    stream: list[ScriptedResponse] | None = None,
    settings: AppSettings | None = None,
    policy: ModelPolicy | None = None,
) -> LearningHarness:
    """装配测试用的一套服务。"""
    effective = settings or AppSettings()
    client = FakeLlmClient(blocking_script=blocking, stream_script=stream)
    repos = DomainRepositories(ctx.db)
    state = SessionStateService(repos.sessions, repos.subjects)
    subjects = SubjectService(repos)

    resolved_policy = policy or ModelPolicy(effective)
    usage = UsageRecorder(ctx.db)
    sessions = ConversationSessionFactory(client, ctx.memories, usage, resolved_policy)

    return LearningHarness(
        ctx=ctx,
        client=client,
        settings=effective,
        repos=repos,
        state=state,
        subjects=subjects,
        sessions=sessions,
        policy=resolved_policy,
    )
