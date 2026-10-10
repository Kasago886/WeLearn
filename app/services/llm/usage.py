"""用量落库（FR-705）。

``prompt_cache_hit_tokens`` 与 ``prompt_cache_miss_tokens`` 都记录：
命中率异常偏低往往意味着上下文前缀被改写（例如压缩重排），是排查费用问题的关键线索。
"""

from collections.abc import Mapping
from dataclasses import dataclass

from app.contracts.schemas import LlmPurpose
from app.infra.deepseek.protocol import TokenUsage
from app.infra.storage.db import Database
from app.infra.storage.models import UsageRecordRow


@dataclass(frozen=True)
class UsageSummary:
    """聚合用量。"""

    calls: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_miss_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率；无输入时返回 0。"""
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cached_prompt_tokens / self.prompt_tokens


class UsageRecorder:
    """把每次请求的用量写入 ``usage_record``。"""

    def __init__(
        self,
        db: Database,
        *,
        subject_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self._db = db
        self._subject_id = subject_id
        self._conversation_id = conversation_id

    def record(
        self,
        purpose: LlmPurpose,
        model: str,
        usage: TokenUsage | None,
        *,
        conversation_id: str | None = None,
    ) -> None:
        """写入一条记录；``usage`` 为空时写入全 0 记录，以保留"这次调用发生过的"事实。"""
        payload = usage or TokenUsage()
        with self._db.session() as session:
            session.add(
                UsageRecordRow(
                    subject_id=self._subject_id,
                    conversation_id=conversation_id or self._conversation_id,
                    purpose=purpose.value,
                    model=model,
                    prompt_tokens=payload.prompt_tokens,
                    cached_prompt_tokens=payload.cached_prompt_tokens,
                    cache_miss_tokens=payload.cache_miss_tokens,
                    completion_tokens=payload.completion_tokens,
                    reasoning_tokens=payload.reasoning_tokens,
                    total_tokens=payload.total_tokens,
                )
            )

    def summary(self, *, subject_id: str | None = None) -> UsageSummary:
        """聚合统计，用于主题详情页展示（T3.6）。"""
        from sqlalchemy import func, select

        target = subject_id if subject_id is not None else self._subject_id
        statement = select(
            func.count(UsageRecordRow.id),
            func.coalesce(func.sum(UsageRecordRow.prompt_tokens), 0),
            func.coalesce(func.sum(UsageRecordRow.cached_prompt_tokens), 0),
            func.coalesce(func.sum(UsageRecordRow.cache_miss_tokens), 0),
            func.coalesce(func.sum(UsageRecordRow.completion_tokens), 0),
            func.coalesce(func.sum(UsageRecordRow.reasoning_tokens), 0),
            func.coalesce(func.sum(UsageRecordRow.total_tokens), 0),
        )
        if target is not None:
            statement = statement.where(UsageRecordRow.subject_id == target)

        with self._db.session() as session:
            row = session.execute(statement).one()

        return UsageSummary(
            calls=int(row[0]),
            prompt_tokens=int(row[1]),
            cached_prompt_tokens=int(row[2]),
            cache_miss_tokens=int(row[3]),
            completion_tokens=int(row[4]),
            reasoning_tokens=int(row[5]),
            total_tokens=int(row[6]),
        )

    @staticmethod
    def from_mapping(purpose: LlmPurpose, model: str, usage: Mapping[str, object]) -> UsageRecordRow:
        """从原始字典构造行（测试与导入场景使用）。"""
        parsed = TokenUsage.from_usage_dict(usage)
        payload = parsed or TokenUsage()
        return UsageRecordRow(
            purpose=purpose.value,
            model=model,
            prompt_tokens=payload.prompt_tokens,
            cached_prompt_tokens=payload.cached_prompt_tokens,
            cache_miss_tokens=payload.cache_miss_tokens,
            completion_tokens=payload.completion_tokens,
            reasoning_tokens=payload.reasoning_tokens,
            total_tokens=payload.total_tokens,
        )
