"""上下文压缩（T1.4，FR-604~FR-609）。

移植 ReferenceCode ``ContextCompressor`` 的核心设计，它解决了三件容易做错的事：

1. **范围**：跳过开头的 system 轮次（FR-608：系统提示词永不被压缩掉），并限制单次压缩量；
2. **防过期**：压缩期间用户会继续说话，因此记录待压缩范围的 **SHA-256 内容指纹**，
   应用前重新计算并比对——不一致就丢弃本次结果。这一步让"压缩不阻塞对话"（FR-609）
   在正确性上成立；
3. **摘要注入形态**：摘要以 ``user`` 角色并显式声明"这不是新的用户请求、冲突以后续为准"
   （FR-605），否则模型会把摘要当成新指令。

与 ReferenceCode 的差异：``compressed_context`` 为空时视为"无需保留"（不重试），
格式错误才重试一次。
"""

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from app.contracts.schemas import CompressionSummary, LlmPurpose
from app.infra.llm.errors import LlmError
from app.infra.logging.queue_logger import CHANNEL_LLM, get_logger
from app.prompts import COMPRESSOR, render
from app.services.llm.agent import RoundExecutor
from app.services.llm.memory import Memory
from app.services.llm.structured import ParseOutcome, parse_json_payload


@dataclass(frozen=True)
class CompressionPlan:
    """一次压缩任务的计划。"""

    start_round_index: int
    round_count: int
    signature: str
    payload: str


class CompressionOutcome(StrEnum):
    """压缩结果，便于观测与测试（FR-904 的关键状态留痕）。"""

    APPLIED = "applied"
    NO_PLAN = "no_plan"
    EMPTY_SUMMARY = "empty_summary"
    INVALID_SUMMARY = "invalid_summary"
    STALE = "stale"
    FAILED = "failed"


def compute_signature(messages: list[dict[str, object]]) -> str:
    """范围内容指纹（稳定序列化后取 SHA-256）。"""
    payload = json.dumps(messages, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def try_create_plan(
    memory: Memory,
    *,
    threshold_rounds: int,
    compress_rounds: int,
) -> CompressionPlan | None:
    """生成压缩计划；轮次不足或没有可压缩内容时返回 ``None``。"""
    if compress_rounds <= 0:
        return None

    with memory.lock:
        if memory.round_count < threshold_rounds:
            return None

        skip = memory.count_leading_system_rounds()
        available = memory.round_count - skip
        count = min(compress_rounds, available)
        if count <= 0:
            return None

        target = memory.protocol_range(skip, count)
        if not target:
            return None

        return CompressionPlan(
            start_round_index=skip,
            round_count=count,
            signature=compute_signature(target),
            payload=json.dumps(
                {"complete_messages": memory.outbound_messages(), "messages_to_compress": target},
                ensure_ascii=False,
                indent=2,
            ),
        )


def parse_summary(content: str | None) -> tuple[CompressionOutcome, str]:
    """解析摘要回复（三级容错，与结构化输出通道共用解析器）。

    必须区分两种情况：**字段缺失**属于格式不符（Invalid，可重试）；
    **字段存在但为空**是模型明确表示"无需保留"（Empty，不重试）。
    若把前者也当成 Empty，就会静默跳过本该发生的压缩。
    """
    parsed = parse_json_payload(content)

    if parsed.outcome is ParseOutcome.EMPTY:
        return CompressionOutcome.INVALID_SUMMARY, ""
    if parsed.outcome is ParseOutcome.INVALID or parsed.payload is None:
        return CompressionOutcome.INVALID_SUMMARY, ""
    if "compressed_context" not in parsed.payload:
        return CompressionOutcome.INVALID_SUMMARY, ""

    try:
        model = CompressionSummary.model_validate(parsed.payload)
    except Exception:
        return CompressionOutcome.INVALID_SUMMARY, ""

    summary = model.compressed_context.strip()
    if not summary:
        return CompressionOutcome.EMPTY_SUMMARY, ""
    return CompressionOutcome.APPLIED, summary


def try_apply(memory: Memory, plan: CompressionPlan, summary: str) -> bool:
    """把摘要写回记忆；范围跨出边界或内容已变化（过期）时返回 ``False`` 且不修改记忆。"""
    if not summary.strip():
        return False

    with memory.lock:
        if plan.start_round_index < 0 or plan.round_count <= 0:
            return False
        if plan.start_round_index + plan.round_count > memory.round_count:
            return False

        current = memory.protocol_range(plan.start_round_index, plan.round_count)
        if compute_signature(current) != plan.signature:
            return False

        return memory.replace_rounds_with_summary(plan.start_round_index, plan.round_count, summary)


class ContextCompressor:
    """上下文压缩服务。"""

    def __init__(
        self,
        executor: RoundExecutor,
        *,
        threshold_rounds: int,
        compress_rounds: int,
    ) -> None:
        self._executor = executor
        self._threshold_rounds = threshold_rounds
        self._compress_rounds = compress_rounds
        self._log = get_logger(CHANNEL_LLM)

    async def maybe_compress(self, memory: Memory | None = None) -> CompressionOutcome:
        """按需压缩；不需要压缩时直接返回，不做任何调用。"""
        target = memory if memory is not None else self._executor.memory
        plan = try_create_plan(
            target,
            threshold_rounds=self._threshold_rounds,
            compress_rounds=self._compress_rounds,
        )
        if plan is None:
            return CompressionOutcome.NO_PLAN

        prompt = render(COMPRESSOR, payload=plan.payload)
        try:
            content, _usage = await self._executor.call_blocking(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "请按上述要求输出 json 摘要。"},
                ],
                purpose=LlmPurpose.COMPRESS,
                json_object=True,
            )
        except LlmError as exc:
            self._log.warning(f"压缩请求失败: {exc.category.value} {exc.user_message}")
            return CompressionOutcome.FAILED

        outcome, summary = parse_summary(content)

        if outcome is CompressionOutcome.EMPTY_SUMMARY:
            self._log.info("压缩模型判定无需保留内容，跳过本次压缩")
            return CompressionOutcome.EMPTY_SUMMARY

        if outcome is CompressionOutcome.INVALID_SUMMARY:
            self._log.warning("压缩摘要格式不符，放弃本次压缩（记忆未被修改）")
            return CompressionOutcome.INVALID_SUMMARY

        if not try_apply(target, plan, summary):
            self._log.warning("压缩结果已过期（范围内容在压缩期间被改动），丢弃本次结果（FR-606）")
            return CompressionOutcome.STALE

        self._log.info(
            f"上下文压缩已生效: 轮次 {plan.start_round_index}~{plan.start_round_index + plan.round_count - 1}"
            f"（{plan.round_count} 轮）替换为一条摘要"
        )
        return CompressionOutcome.APPLIED

    def plan_preview(self, memory: Memory | None = None) -> CompressionPlan | None:
        """只生成计划、不调用模型（调试与测试用）。"""
        return try_create_plan(
            memory if memory is not None else self._executor.memory,
            threshold_rounds=self._threshold_rounds,
            compress_rounds=self._compress_rounds,
        )
