"""对话会话封装（p2-assessment 前置）。

**为什么需要这一层**：P1 的 ``RoundExecutor`` 在一次对话里改动的是内存中的 ``Memory`` 对象，
落库需要显式调用。若让测评/教学/复习各自处理"载入 → 执行 → 保存"，
一定会有人忘记保存（表现为"聊了几轮，重启后全没了"）。

这里把三件事绑在一起：会话 id、内存记忆、执行器，并暴露唯一的 :meth:`ConversationSession.persist`。
每个会话对应一个 :class:`ConversationKind`，因此测评与教学的记忆天然隔离（FR-602、FR-209）。
"""

from dataclasses import dataclass

from app.contracts.schemas import ConversationKind, LlmPurpose
from app.infra.deepseek.client import DeepSeekClient
from app.infra.deepseek.protocol import TokenUsage
from app.services.llm.agent import RoundExecutor
from app.services.llm.judge import Judge
from app.services.llm.memory import Memory
from app.services.llm.policy import ModelPolicy
from app.services.llm.repository import MemoryRepository
from app.services.llm.structured import StructuredChannel
from app.services.llm.usage import UsageRecorder


@dataclass
class ConversationSession:
    """一个用途下的对话：会话 id + 内存记忆 + 执行器。"""

    conversation_id: str
    kind: ConversationKind
    memory: Memory
    executor: RoundExecutor
    _repository: MemoryRepository

    def persist(self) -> int:
        """把内存记忆写回数据库；返回写入的轮次数。"""
        return self._repository.save(self.conversation_id, self.memory)

    @property
    def structured(self) -> StructuredChannel:
        """基于本会话执行器的结构化输出通道。"""
        return StructuredChannel(self.executor)

    @property
    def judge(self) -> Judge:
        """基于本会话执行器的判定器（判定本身是独立调用，不读本会话记忆）。"""
        return Judge(self.structured)

    def mark_compressed(self) -> None:
        self._repository.mark_compressed(self.conversation_id)


class ConversationSessionFactory:
    """按"主题 + 用途"打开会话。"""

    def __init__(
        self,
        client: DeepSeekClient,
        memories: MemoryRepository,
        usage: UsageRecorder,
        policy: ModelPolicy,
        *,
        max_retries: int = 2,
        tools_enabled: bool = False,
    ) -> None:
        self._client = client
        self._memories = memories
        self._usage = usage
        self._policy = policy
        self._max_retries = max_retries
        self._tools_enabled = tools_enabled

    def open(self, subject_id: str, kind: ConversationKind) -> ConversationSession:
        """载入（或新建）会话。"""
        conversation_id = self._memories.ensure_conversation(kind, subject_id=subject_id)
        memory = self._memories.load(conversation_id)

        def on_usage(
            purpose: LlmPurpose,
            model: str,
            usage: TokenUsage | None,
            duration_ms: int | None = None,
        ) -> None:
            # 用量按主题聚合（FR-106 的进度概览需要），因此这里显式带上 subject_id
            self._usage.record(
                purpose,
                model,
                usage,
                conversation_id=conversation_id,
                subject_id=subject_id,
                duration_ms=duration_ms,
            )

        executor = RoundExecutor(
            self._client,
            memory,
            self._policy,
            on_usage=on_usage,
            tools_enabled=self._tools_enabled,
        )
        return ConversationSession(
            conversation_id=conversation_id,
            kind=kind,
            memory=memory,
            executor=executor,
            _repository=self._memories,
        )
