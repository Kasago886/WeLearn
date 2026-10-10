"""P1 内核集成测试：验证实施计划第 11 章的 P1 出口条件。

出口条件原文（docs/实施计划.md）：用桩驱动 100 轮对话触发压缩；记忆不脏、失败不入库、并发不串改。

这里把三件事放在一起跑，因为它们只有在**长跑**下才会互相作用：
压缩改写了历史 → 前缀变化 → 下一轮仍要能正常提交；中途失败一轮 → 不能污染后续压缩的范围。
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from app.contracts.schemas import LlmPurpose
from app.infra.deepseek.errors import LlmError
from app.infra.deepseek.protocol import BlockingResult, StreamEvent, TokenUsage
from app.infra.settings import AppSettings
from app.services.llm.agent import RoundExecutor
from app.services.llm.compression import CompressionOutcome, ContextCompressor
from app.services.llm.dispatcher import RoundDispatcher
from app.services.llm.memory import CONTENT_KEY, Memory, system_message
from app.services.llm.policy import ModelPolicy
from tests.stubs import FakeLlmClient

ROUNDS = 100
THRESHOLD = 10
COMPRESS_ROUNDS = 5
SUMMARY = "（历史摘要）此前若干轮讨论了贝叶斯定理的定义与公式。"
USAGE = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)


class SimulationClient(FakeLlmClient):
    """既能回答对话、又能返回合法压缩摘要的桩，并统计调用次数与日志。"""

    def __init__(self, *, fail_calls: set[int] | None = None) -> None:
        super().__init__()
        self.fail_calls = fail_calls or set()
        self.stream_calls = 0
        self.blocking_calls = 0

    async def stream_chat(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        **_: Any,
    ) -> AsyncIterator[StreamEvent]:
        self.stream_calls += 1
        if self.stream_calls in self.fail_calls:
            raise LlmError.network(f"模拟第 {self.stream_calls} 次请求的网络中断")

        question = str(messages[-1].get("content", ""))
        prompt_tokens = sum(len(str(message.get("content", ""))) for message in messages)
        yield StreamEvent(content=f"针对「{question}」的讲解。")
        yield StreamEvent(
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=10,
                total_tokens=prompt_tokens + 10,
            )
        )

    async def chat(
        self,
        *,
        json_object: bool = False,
        **_: Any,
    ) -> BlockingResult:
        self.blocking_calls += 1
        if json_object:
            # 摘要文本里不含花括号，故这里的 f-string 不会与 JSON 的 {} 冲突
            return BlockingResult(
                content=f'{{"compressed_context": "{SUMMARY}"}}',
                reasoning=None,
                usage=USAGE,
            )
        return BlockingResult(content="普通回答", reasoning=None, usage=USAGE)


def build_stack(client: SimulationClient) -> tuple[Memory, RoundDispatcher, ContextCompressor, list[tuple]]:
    memory = Memory()
    memory.add_round().add(system_message("你是耐心的导师，逐点教学。"))

    usage_log: list[tuple] = []
    executor = RoundExecutor(
        client,
        memory,
        ModelPolicy(AppSettings(max_tokens=256)),
        on_usage=lambda *args: usage_log.append(args),
    )
    dispatcher = RoundDispatcher(executor, max_retries=1)
    compressor = ContextCompressor(
        executor,
        threshold_rounds=THRESHOLD,
        compress_rounds=COMPRESS_ROUNDS,
    )
    return memory, dispatcher, compressor, usage_log


async def test_hundred_rounds_converge_without_corruption() -> None:
    """长跑验证：压缩持续生效、轮次有界、system 提示不被压缩掉。"""
    client = SimulationClient()
    memory, dispatcher, compressor, usage_log = build_stack(client)

    applied = 0

    for index in range(ROUNDS):
        result = await dispatcher.submit(f"第 {index} 轮提问")
        assert result.ok, result.message

        outcome = await compressor.maybe_compress()
        if outcome is CompressionOutcome.APPLIED:
            applied += 1
        else:
            assert outcome is CompressionOutcome.NO_PLAN, f"第 {index} 轮出现异常压缩结果: {outcome}"

        # 不变量：system 轮次永远在第一轮，且从未被压缩
        assert memory.rounds[0].is_system_round is True

        # 不变量：轮次始终有界（否则压缩等于没做）
        assert memory.round_count <= THRESHOLD + 1, f"第 {index} 轮后轮次失控: {memory.round_count}"

    assert client.stream_calls == ROUNDS
    # 100 轮、每 4 轮净减 1 轮左右，压缩必然多次生效
    assert applied >= 15, f"压缩生效次数过少: {applied}"
    assert client.blocking_calls == applied

    # 末尾仍是最新一轮的内容
    last = memory.outbound_messages()[-1]
    assert last["role"] == "assistant"
    assert "第 99 轮提问" in str(memory.outbound_messages()[-2][CONTENT_KEY])

    # 摘要仍在记忆里
    assert any("历史摘要" in envelope.content for envelope in memory.envelopes())
    assert any(envelope.is_summary for envelope in memory.envelopes())

    # 每次实际请求都记录了用量（含压缩调用）
    assert len(usage_log) == ROUNDS + applied


async def test_failed_round_in_the_middle_does_not_corrupt_memory() -> None:
    """FR-610 的长跑版：中途整轮失败（含重试仍失败）后，压缩与提交都必须照常工作。"""
    # 第 37 次请求与它的那次重试（第 38 次）都失败 → 这一轮彻底失败
    client = SimulationClient(fail_calls={37, 38})
    memory, dispatcher, compressor, _ = build_stack(client)

    failed_index = 36

    for index in range(60):
        result = await dispatcher.submit(f"第 {index} 轮提问")

        if index == failed_index:
            assert result.error is True, "这一轮的重试也应失败"
            assert result.error_category == "network"
            # 失败那一轮的用户输入不得进入记忆
            assert all(f"第 {index} 轮提问" not in envelope.content for envelope in memory.envelopes())
        else:
            assert result.ok, f"第 {index} 轮不应失败: {result.message}"

        await compressor.maybe_compress()
        assert memory.rounds[0].is_system_round is True

    contents = "\n".join(envelope.content for envelope in memory.envelopes())

    # 失败轮次自始至终没有进入记忆（上面的逐轮断言是关键，这里再复核一次）
    assert f"第 {failed_index} 轮提问" not in contents

    # 压缩继续正常工作：最近的轮次在位，且摘要已生效
    assert "第 59 轮提问" in contents
    assert any(envelope.is_summary for envelope in memory.envelopes())
    # 长跑后轮次仍然有界
    assert memory.round_count <= THRESHOLD + 1


async def test_busy_rejection_never_mixes_into_memory() -> None:
    """并发场景：忙碌被拒绝的那一轮，其输入与输出都不得进入记忆。"""
    import asyncio

    gate = asyncio.Event()

    class GatedClient(SimulationClient):
        async def stream_chat(self, *, messages, **kwargs):  # type: ignore[override]
            question = str(messages[-1].get("content", ""))
            yield StreamEvent(content=f"回答「{question}」")
            await gate.wait()

    client = GatedClient()
    memory, dispatcher, _, _ = build_stack(client)

    slow = asyncio.create_task(dispatcher.submit("慢慢来"))
    await asyncio.sleep(0.05)

    rejected = await dispatcher.submit("别插队")
    assert rejected.error is True
    assert rejected.error_category == "busy"

    gate.set()
    assert (await slow).ok is True

    contents = [envelope.content for envelope in memory.envelopes()]
    assert not any("别插队" in content for content in contents)
    assert any("慢慢来" in content for content in contents)


async def test_compression_reduces_context_size_materially() -> None:
    """压缩的意义在于减小上下文：长跑后请求体积必须显著小于"不压缩"的等价轮次。"""
    client = SimulationClient()
    memory, dispatcher, compressor, _ = build_stack(client)

    for index in range(40):
        await dispatcher.submit(f"第 {index} 轮提问，这里再多写一些字让上下文变长。")
        await compressor.maybe_compress()

    outbound = memory.outbound_messages()
    compressed_chars = sum(len(str(message.get("content", ""))) for message in outbound)

    # 不压缩时，40 轮提问 + 40 轮回答的正文量
    uncompressed_chars = sum(
        len(f"第 {index} 轮提问，这里再多写一些字让上下文变长。")
        + len(f"针对「第 {index} 轮提问，这里再多写一些字让上下文变长。」的讲解。")
        for index in range(40)
    )

    assert compressed_chars < uncompressed_chars * 0.6, (
        f"压缩效果不明显: {compressed_chars} vs 未压缩 {uncompressed_chars}"
    )
    # 摘要确实替代了原始轮次
    assert sum(1 for envelope in memory.envelopes() if envelope.is_summary) >= 1


async def test_summary_keeps_prompt_cache_prefix_observable() -> None:
    """压缩必然改写前缀（缓存会冷启动一次），此后前缀应重新稳定下来。"""
    from app.services.llm.context_policy import build_outbound_messages, stable_prefix_hash

    client = SimulationClient()
    memory, dispatcher, compressor, _ = build_stack(client)

    for index in range(30):
        await dispatcher.submit(f"第 {index} 轮提问")
        await compressor.maybe_compress()

    before = build_outbound_messages(memory, tools_enabled=False)
    hash_before = stable_prefix_hash(before)

    # 仅新增一轮，前缀部分（除最后新增的两条）必须完全不变
    await dispatcher.submit("再来一轮")
    after = build_outbound_messages(memory, tools_enabled=False)

    assert len(after) > len(before)
    assert stable_prefix_hash(after[: len(before) - 1]) == stable_prefix_hash(before[: len(before) - 1])
    assert hash_before != stable_prefix_hash(after)


async def test_purposes_use_expected_models() -> None:
    """FR-703：判定/规划可用 pro 模型、压缩关闭思考，且不影响教学。"""
    from app.services.llm.policy import ModelPolicy, apply_recommended_overrides

    policy = apply_recommended_overrides(ModelPolicy(AppSettings()))

    assert policy.resolve(LlmPurpose.TEACH).model == "deepseek-flash"
    assert policy.resolve(LlmPurpose.TEACH).thinking is True
    assert policy.resolve(LlmPurpose.JUDGE).model == "deepseek-v4-pro"
    assert policy.resolve(LlmPurpose.COMPRESS).thinking is False
