"""知识基础确认服务（T2.3，FR-201~FR-210）。

流程（每轮只聚焦一个子领域，因此轮次数是确定的）：

```text
start()
  └─ 结构化调用：推导 3~6 个子领域 + 第一个问题
     ├─ 落库 topics（数量受 assess_max_turns 限制），状态机 → assessing
     └─ 写入首问

answer(文本)                                    ← 学习者作答
  ├─ 记入测评记忆，turn_count += 1
  ├─ 还有未考察的子领域 → 流式生成下一个问题（聚焦下一个子领域）
  └─ 已覆盖全部子领域   → 结构化生成画像 → 落库 → 状态机 → path_review

skip()                                          ← 跳过测评（FR-205）
  └─ 按零基础收尾：状态机 → path_review，并留下"未测评"痕迹
```

**为什么轮次数由子领域数量决定，而不是"聊到模型说够了"**：后者需要额外一次调用判断
"信息是否充分"，且结论不稳定——同一个学习者重来一次可能问 3 轮也可能问 6 轮。
按子领域推进是确定的，也能把"第 2/4 个方面"直接展示给用户（FR-202）。

**提示词注入方式**：测评的 system 提示随轮次变化（本轮聚焦哪个子领域），而
``RoundExecutor`` 只在记忆为空时接受 ``system_prompt``。因此这里把它写成**首条 system 消息并
每轮就地改写**——官方对 system 更新有 ``leading-only`` 与 ``in-history`` 两种语义，
只有改写开头那条在两种情况下都生效。

**记忆隔离**（FR-209）：测评用 ``ASSESS`` 会话、教学用 ``TEACH`` 会话；
教学只通过画像读取测评结论，不携带测评原始对话。
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from app.contracts.schemas import (
    AssessmentProfile,
    AssessmentTopics,
    ConversationKind,
    LlmPurpose,
    MasteryLevel,
    SubjectState,
)
from app.infra.logging.queue_logger import CHANNEL_APP, get_logger
from app.infra.settings import AppSettings
from app.prompts import ASSESSOR, ASSESSOR_PROFILE, ASSESSOR_TOPICS, render
from app.services.learning.entities import Assessment, ProfileLevel, Subject
from app.services.learning.repository import DomainRepositories
from app.services.learning.session import ConversationSession, ConversationSessionFactory
from app.services.learning.state import SessionStateService
from app.services.llm.agent import DeltaCallback
from app.services.llm.memory import (
    CONTENT_KEY,
    ROLE_KEY,
    Memory,
    assistant_message,
    set_leading_system,
    system_message,
    user_message,
)
from app.services.llm.result import RoundResult
from app.services.llm.structured import StructuredOutputError

TOPIC_MIN = 3
TOPIC_MAX = 6
TRANSCRIPT_LIMIT = 4000


@dataclass
class AssessmentTurn:
    """一次测评交互的产出，供 API 层直接转成 SSE。"""

    text: str = ""
    finished: bool = False
    skipped: bool = False
    turn_number: int = 0
    total_turns: int = 0
    current_topic: str | None = None
    topics: tuple[str, ...] = ()
    profile: AssessmentProfile | None = None
    result: RoundResult | None = None
    error_message: str = ""
    error_category: str | None = None
    retryable: bool = False

    @property
    def ok(self) -> bool:
        return not self.error_message


@dataclass
class AssessmentView:
    """测评详情页数据（FR-206、FR-208）。"""

    assessment: Assessment
    levels: list[ProfileLevel] = field(default_factory=list)

    @property
    def has_profile(self) -> bool:
        return bool(self.levels)


class AssessmentError(RuntimeError):
    """测评流程错误（消息面向用户）。"""


class AssessmentService:
    """测评编排。"""

    def __init__(
        self,
        repos: DomainRepositories,
        state: SessionStateService,
        sessions: ConversationSessionFactory,
        settings_provider: Callable[[], AppSettings],
    ) -> None:
        self._repos = repos
        self._state = state
        self._sessions = sessions
        self._settings = settings_provider
        self._log = get_logger(CHANNEL_APP)

    # ------------------------------------------------------------------ #
    # 读取
    # ------------------------------------------------------------------ #

    def view(self, subject_id: str) -> AssessmentView | None:
        assessment = self._repos.assessments.get_latest(subject_id)
        if assessment is None:
            return None
        return AssessmentView(assessment=assessment, levels=self._repos.assessments.levels(assessment.id))

    def levels(self, subject_id: str) -> list[ProfileLevel]:
        view = self.view(subject_id)
        return view.levels if view is not None else []

    def override_level(self, subject_id: str, topic: str, level: MasteryLevel) -> bool:
        """人工修正某个子领域的水平（FR-207）。"""
        assessment = self._repos.assessments.get_latest(subject_id)
        if assessment is None:
            return False
        return self._repos.assessments.override_level(assessment.id, topic, level)

    # ------------------------------------------------------------------ #
    # 开始测评
    # ------------------------------------------------------------------ #

    async def start(self, subject_id: str) -> AssessmentTurn:
        """开始测评：推导子领域并抛出第一个问题。

        **进入 ``assessing`` 的时机在结构化调用成功之后**。这样"推导失败"时状态机保持原样，
        不需要任何回滚——而回滚恰恰不可靠（``assessing → draft`` 在迁移表里是非法的，
        强行回滚会让状态机失去约束意义）。
        """
        subject = self._require_subject(subject_id)
        settings = self._settings()

        if self._repos.assessments.get_running(subject_id) is not None:
            raise AssessmentError("已经有一次测评进行中，请先完成它，或点击「跳过测评」。")

        session = self._sessions.open(subject_id, ConversationKind.ASSESS)
        # 每次重新测评都从干净记忆开始，避免上一轮测评的结论干扰本轮判断
        session.memory.replace_with(Memory())
        session.persist()

        try:
            topics_model, _usage = await session.structured.request(
                LlmPurpose.ASSESS,
                system=render(
                    ASSESSOR_TOPICS,
                    concept=subject.title,
                    goal=subject.goal or "（未填写）",
                    background=subject.background or "（未填写）",
                    min_topics=TOPIC_MIN,
                    max_topics=TOPIC_MAX,
                ),
                user="请按上述要求输出 json。",
                schema=AssessmentTopics,
            )
        except StructuredOutputError as exc:
            self._log.warning(f"子领域推导失败（状态保持 {self._state.snapshot(subject_id).state.value}）: "
                              f"{exc.user_message}")
            return AssessmentTurn(
                error_message="没能生成诊断问题，请稍后重试。",
                error_category="structured_output",
                retryable=True,
            )

        topics = _clean_topics(topics_model.topics, limit=settings.assess_max_turns)
        self._repos.assessments.create(subject_id, max_turns=len(topics), topics=topics)
        self._state.transition(subject_id, SubjectState.ASSESSING)

        session.memory.add_round().add(system_message(self._system_prompt(subject, topics, 1)))
        session.memory.add_message_to_last_round(assistant_message(topics_model.first_question))
        session.persist()

        self._log.info(f"测评开始 subject={subject_id} topics={len(topics)}")

        return AssessmentTurn(
            text=topics_model.first_question,
            finished=False,
            turn_number=0,
            total_turns=len(topics),
            current_topic=topics[0] if topics else None,
            topics=topics,
        )

    # ------------------------------------------------------------------ #
    # 作答
    # ------------------------------------------------------------------ #

    async def answer(
        self,
        subject_id: str,
        answer: str,
        *,
        on_delta: DeltaCallback | None = None,
    ) -> AssessmentTurn:
        """提交一次测评作答：继续问下一个子领域，或产出画像并结束。"""
        text = (answer or "").strip()
        if not text:
            return AssessmentTurn(
                error_message="请先填写你的回答，或点击「不了解」跳到下一个方面。",
                error_category="validation",
                retryable=False,
            )

        assessment = self._repos.assessments.get_running(subject_id)
        if assessment is None:
            raise AssessmentError("当前没有进行中的测评，请先开始测评。")

        subject = self._require_subject(subject_id)
        session = self._sessions.open(subject_id, ConversationKind.ASSESS)
        session.memory.add_round().add(user_message(text))
        session.persist()

        turn_number = self._repos.assessments.increment_turn(assessment.id)
        topics = assessment.topics
        if not topics:
            # topics 理论上不会丢；真丢了就让测评收敛为"一轮综合考察"，而不是卡死
            topics = (f"{subject.title}的综合理解",)
            self._repos.assessments.set_topics(assessment.id, topics)
        covered = min(turn_number, len(topics))

        if covered >= len(topics):
            return await self._finish_with_profile(subject_id, assessment.id, session)

        # 每轮就地改写首条 system 消息（见模块文档）
        set_leading_system(session.memory, self._system_prompt(subject, topics, covered + 1))

        result = await session.executor.stream("", on_delta=on_delta, purpose=LlmPurpose.ASSESS)
        session.persist()

        if not result.ok:
            return AssessmentTurn(
                result=result,
                error_message=result.message,
                error_category=result.error_category,
                retryable=result.retryable,
                turn_number=turn_number,
                total_turns=len(topics),
                topics=topics,
            )

        return AssessmentTurn(
            text=result.content,
            finished=False,
            turn_number=turn_number,
            total_turns=len(topics),
            current_topic=topics[covered],
            topics=topics,
            result=result,
        )

    # ------------------------------------------------------------------ #
    # 跳过
    # ------------------------------------------------------------------ #

    def skip(self, subject_id: str) -> AssessmentTurn:
        """跳过测评（FR-205）：按零基础处理，并留下"未测评"的痕迹。"""
        self._require_subject(subject_id)

        assessment = self._repos.assessments.get_running(subject_id)
        if assessment is None:
            assessment = self._repos.assessments.create(subject_id, max_turns=0)
        self._repos.assessments.finish(assessment.id, skipped=True)

        self._state.transition(subject_id, SubjectState.PATH_REVIEW)
        self._log.info(f"用户跳过测评 subject={subject_id}")

        return AssessmentTurn(
            text="已跳过知识基础确认。接下来会按零基础为你生成学习路径。",
            finished=True,
            skipped=True,
            total_turns=0,
        )

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    async def _finish_with_profile(
        self,
        subject_id: str,
        assessment_id: str,
        session: ConversationSession,
    ) -> AssessmentTurn:
        """生成画像、落库并把状态推进到 path_review。"""
        transcript = build_transcript(session)
        try:
            profile, _usage = await session.structured.request(
                LlmPurpose.ASSESS,
                system=render(ASSESSOR_PROFILE, transcript=transcript),
                user="请依据上面的对话输出 json 画像。",
                schema=AssessmentProfile,
            )
        except StructuredOutputError as exc:
            self._log.warning(f"画像生成失败: {exc.user_message}")
            # 不移除进行中的测评：用户可以再提交一次作答重试，而不是从头再来
            return AssessmentTurn(
                error_message="没能整理出你的掌握情况，请再提交一次作答以重试。",
                error_category="structured_output",
                retryable=True,
            )

        levels = [
            ProfileLevel(topic=item.topic, level=item.level, evidence=item.evidence)
            for item in profile.levels
        ]
        self._repos.assessments.save_levels(assessment_id, levels)
        finished = self._repos.assessments.finish(
            assessment_id,
            summary=profile.summary,
            suggested_start_node=profile.suggested_start_node,
        )
        self._state.transition(subject_id, SubjectState.PATH_REVIEW)
        session.persist()

        self._log.info(f"测评完成并产出画像 subject={subject_id} levels={len(levels)}")

        return AssessmentTurn(
            text=profile.summary,
            finished=True,
            profile=profile,
            topics=finished.topics if finished is not None else (),
        )

    @staticmethod
    def _system_prompt(subject: Subject, topics: tuple[str, ...], turn_number: int) -> str:
        current = topics[min(turn_number - 1, len(topics) - 1)] if topics else "综合理解"
        return render(
            ASSESSOR,
            concept=subject.title,
            topics=list(topics),
            turn_number=turn_number,
            total_turns=len(topics),
            current_topic=current,
        )

    def _require_subject(self, subject_id: str) -> Subject:
        subject = self._repos.subjects.get(subject_id)
        if subject is None:
            raise KeyError(f"主题不存在: {subject_id}")
        return subject


def _clean_topics(topics: list[str], *, limit: int) -> tuple[str, ...]:
    """去重、去空、按上限截断。"""
    seen: list[str] = []
    for topic in topics:
        text = str(topic).strip()
        if text and text not in seen:
            seen.append(text)
    if limit > 0:
        seen = seen[:limit]
    return tuple(seen)


def build_transcript(session: ConversationSession, *, limit: int = TRANSCRIPT_LIMIT) -> str:
    """把对话记忆整理成紧凑记录，供画像/路径等分析型调用使用。

    只保留 user / assistant 的正文：思维链与 system 提示对分析没有价值，
    放进去既费 token 又会干扰判断。
    """
    lines: list[str] = []
    for envelope in session.memory.envelopes():
        role = envelope.data.get(ROLE_KEY)
        if role not in {"user", "assistant"}:
            continue
        content = str(envelope.data.get(CONTENT_KEY, "")).strip()
        if not content:
            continue
        speaker = "学习者" if role == "user" else "诊断者"
        lines.append(f"{speaker}：{content}")

    transcript = "\n".join(lines)
    if len(transcript) > limit:
        transcript = transcript[-limit:]
    return transcript or "（无对话记录）"
