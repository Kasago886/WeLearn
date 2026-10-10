"""上下文压缩测试（T1.4，FR-604~FR-609）。"""

from app.infra.deepseek.errors import LlmError
from app.infra.settings import AppSettings
from app.services.llm.agent import RoundExecutor
from app.services.llm.compression import (
    CompressionOutcome,
    ContextCompressor,
    compute_signature,
    parse_summary,
    try_apply,
    try_create_plan,
)
from app.services.llm.memory import Memory, assistant_message, system_message, user_message
from app.services.llm.policy import ModelPolicy
from tests.stubs import FakeLlmClient, error_response, json_response


def build_memory(rounds: int, *, with_system: bool = True) -> Memory:
    memory = Memory()
    if with_system:
        memory.add_round().add(system_message("你是耐心的导师"))
    for index in range(rounds):
        round_ = memory.add_round()
        round_.add(user_message(f"问题 {index}"))
        round_.add(assistant_message(f"回答 {index}"))
    return memory


def make_compressor(
    client: FakeLlmClient,
    memory: Memory,
    *,
    threshold: int = 4,
    compress: int = 2,
) -> ContextCompressor:
    executor = RoundExecutor(client, memory, ModelPolicy(AppSettings()))
    return ContextCompressor(executor, threshold_rounds=threshold, compress_rounds=compress)


# --------------------------------------------------------------------------- #
# 计划生成
# --------------------------------------------------------------------------- #


def test_no_plan_below_threshold() -> None:
    memory = build_memory(2)

    assert try_create_plan(memory, threshold_rounds=5, compress_rounds=2) is None


def test_plan_skips_leading_system_rounds() -> None:
    """FR-608：系统提示词永不被压缩掉，因此范围必须从 system 之后开始。"""
    memory = build_memory(4)

    plan = try_create_plan(memory, threshold_rounds=4, compress_rounds=2)

    assert plan is not None
    assert plan.start_round_index == 1  # 跳过 1 个 system 轮次
    assert plan.round_count == 2


def test_plan_caps_round_count_at_available() -> None:
    memory = build_memory(2)

    plan = try_create_plan(memory, threshold_rounds=2, compress_rounds=10)

    assert plan is not None
    assert plan.round_count == 2  # 只有 2 轮可压缩


def test_plan_payload_contains_both_sections() -> None:
    memory = build_memory(3)

    plan = try_create_plan(memory, threshold_rounds=3, compress_rounds=1)

    assert plan is not None
    assert "complete_messages" in plan.payload
    assert "messages_to_compress" in plan.payload
    assert plan.signature


def test_plan_is_none_when_only_system_rounds() -> None:
    memory = Memory()
    memory.add_round().add(system_message("只有系统提示"))

    # 还不到阈值
    assert try_create_plan(memory, threshold_rounds=1, compress_rounds=1) is None


def test_signature_is_stable_and_content_sensitive() -> None:
    messages = [{"role": "user", "content": "a"}]

    assert compute_signature(messages) == compute_signature([{"role": "user", "content": "a"}])
    assert compute_signature(messages) != compute_signature([{"role": "user", "content": "b"}])


# --------------------------------------------------------------------------- #
# 摘要解析
# --------------------------------------------------------------------------- #


def test_parse_summary_variants() -> None:
    outcome, summary = parse_summary('{"compressed_context": "摘要内容"}')
    assert outcome is CompressionOutcome.APPLIED
    assert summary == "摘要内容"

    outcome, summary = parse_summary('```json\n{"compressed_context": "代码块里的摘要"}\n```')
    assert outcome is CompressionOutcome.APPLIED
    assert summary == "代码块里的摘要"

    # 空摘要 = 无需保留，不属于错误
    outcome, _ = parse_summary('{"compressed_context": ""}')
    assert outcome is CompressionOutcome.EMPTY_SUMMARY

    outcome, _ = parse_summary('{"compressed_context": "   "}')
    assert outcome is CompressionOutcome.EMPTY_SUMMARY


def test_parse_summary_invalid_variants() -> None:
    assert parse_summary(None)[0] is CompressionOutcome.INVALID_SUMMARY
    assert parse_summary("")[0] is CompressionOutcome.INVALID_SUMMARY
    assert parse_summary("不是 JSON")[0] is CompressionOutcome.INVALID_SUMMARY
    assert parse_summary('{"other": "字段不对"}')[0] is CompressionOutcome.INVALID_SUMMARY
    assert parse_summary('{"compressed_context":')[0] is CompressionOutcome.INVALID_SUMMARY


# --------------------------------------------------------------------------- #
# 应用与防过期
# --------------------------------------------------------------------------- #


def test_apply_replaces_range_with_single_summary_round() -> None:
    memory = build_memory(4)
    plan = try_create_plan(memory, threshold_rounds=4, compress_rounds=2)
    assert plan is not None

    assert try_apply(memory, plan, "前两轮摘要") is True

    # system 轮次保留，被压缩的 2 轮变成 1 轮摘要
    assert memory.round_count == 4
    assert memory.rounds[0].is_system_round is True
    envelopes = memory.envelopes()
    summaries = [envelope for envelope in envelopes if envelope.is_summary]
    assert len(summaries) == 1
    assert "前两轮摘要" in summaries[0].content
    assert "不是新的用户请求" in summaries[0].content

    # 后面的轮次没有被破坏
    remaining = [envelope.content for envelope in envelopes if not envelope.is_summary]
    assert "问题 2" in remaining
    assert "回答 3" in remaining


def test_apply_rejects_stale_plan() -> None:
    """FR-606：压缩期间范围内容被改动时必须丢弃，而不是错位替换。"""
    memory = build_memory(4)
    plan = try_create_plan(memory, threshold_rounds=4, compress_rounds=2)
    assert plan is not None

    # 模拟压缩期间该范围被另一次压缩替换过
    memory.replace_rounds_with_summary(plan.start_round_index, plan.round_count, "别的摘要")
    before = memory.outbound_messages()

    assert try_apply(memory, plan, "本次摘要") is False
    assert memory.outbound_messages() == before


def test_apply_rejects_out_of_bounds() -> None:
    memory = build_memory(2)
    plan = try_create_plan(memory, threshold_rounds=2, compress_rounds=2)
    assert plan is not None

    memory.remove_rounds(1, 2)  # 把可压缩部分删掉

    assert try_apply(memory, plan, "摘要") is False


def test_apply_rejects_empty_summary() -> None:
    memory = build_memory(4)
    plan = try_create_plan(memory, threshold_rounds=4, compress_rounds=1)
    assert plan is not None

    assert try_apply(memory, plan, "   ") is False


def test_apply_succeeds_when_new_rounds_appended_during_compression() -> None:
    """FR-609：压缩不阻塞对话——压缩期间用户新增的轮次不得丢失。"""
    memory = build_memory(4)
    plan = try_create_plan(memory, threshold_rounds=4, compress_rounds=2)
    assert plan is not None

    # 压缩进行中，用户又说了两句
    memory.add_round().add(user_message("新问题"))
    memory.add_message_to_last_round(assistant_message("新回答"))

    assert try_apply(memory, plan, "前两轮摘要") is True

    contents = [envelope.content for envelope in memory.envelopes()]
    assert "前两轮摘要" in "\n".join(contents)
    # 新增轮次仍在
    assert "新问题" in contents
    assert "新回答" in contents


# --------------------------------------------------------------------------- #
# 服务编排
# --------------------------------------------------------------------------- #


async def test_maybe_compress_skips_when_below_threshold() -> None:
    client = FakeLlmClient()
    memory = build_memory(1)

    outcome = await make_compressor(client, memory, threshold=10).maybe_compress()

    assert outcome is CompressionOutcome.NO_PLAN
    assert client.blocking_calls == 0


async def test_maybe_compress_applies_summary() -> None:
    client = FakeLlmClient(blocking_script=[json_response('{"compressed_context": "两轮摘要"}')])
    memory = build_memory(4)

    outcome = await make_compressor(client, memory, threshold=4, compress=2).maybe_compress()

    assert outcome is CompressionOutcome.APPLIED
    assert client.blocking_calls == 1
    assert client.calls[0].json_object is True
    assert client.calls[0].purpose.value == "compress"

    contents = [envelope.content for envelope in memory.envelopes()]
    assert any("两轮摘要" in content for content in contents)
    # system 提示还在
    assert memory.rounds[0].is_system_round is True

    # 提示词必须含 json 字样
    assert "json" in str(client.calls[0].messages[0]["content"]).lower()


async def test_maybe_compress_handles_empty_summary_without_touching_memory() -> None:
    client = FakeLlmClient(blocking_script=[json_response('{"compressed_context": ""}')])
    memory = build_memory(4)
    before = memory.outbound_messages()

    outcome = await make_compressor(client, memory, threshold=4, compress=2).maybe_compress()

    assert outcome is CompressionOutcome.EMPTY_SUMMARY
    assert memory.outbound_messages() == before


async def test_maybe_compress_handles_invalid_summary() -> None:
    client = FakeLlmClient(blocking_script=[json_response("这不是 JSON")])
    memory = build_memory(4)
    before = memory.outbound_messages()

    outcome = await make_compressor(client, memory, threshold=4, compress=2).maybe_compress()

    assert outcome is CompressionOutcome.INVALID_SUMMARY
    assert memory.outbound_messages() == before


async def test_maybe_compress_reports_stale_result() -> None:
    """压缩请求返回时范围已变 → 丢弃并报告 STALE（可观测，便于排查）。"""
    memory = build_memory(4)

    class MutatingClient(FakeLlmClient):
        async def chat(self, **kwargs):  # type: ignore[override]
            # 模拟"压缩尚未返回时，该范围已被另一次压缩替换"
            memory.replace_rounds_with_summary(1, 2, "并发的摘要")
            return await super().chat(**kwargs)

    client = MutatingClient(blocking_script=[json_response('{"compressed_context": "本次摘要"}')])

    outcome = await make_compressor(client, memory, threshold=4, compress=2).maybe_compress()

    assert outcome is CompressionOutcome.STALE


async def test_maybe_compress_reports_model_failure() -> None:
    client = FakeLlmClient(blocking_script=[error_response(LlmError.from_status(503))])
    memory = build_memory(4)
    before = memory.outbound_messages()

    outcome = await make_compressor(client, memory, threshold=4, compress=2).maybe_compress()

    assert outcome is CompressionOutcome.FAILED
    assert memory.outbound_messages() == before


async def test_plan_preview_does_not_call_model() -> None:
    client = FakeLlmClient()
    memory = build_memory(4)
    compressor = make_compressor(client, memory, threshold=4, compress=2)

    plan = compressor.plan_preview()

    assert plan is not None
    assert plan.round_count == 2
    assert client.blocking_calls == 0


async def test_repeated_compression_converges() -> None:
    """反复压缩必须收敛：每一轮都在减少轮次，而不是原地打转。"""
    client = FakeLlmClient(fallback=json_response('{"compressed_context": "摘要"}'))
    memory = build_memory(8)
    compressor = make_compressor(client, memory, threshold=4, compress=2)

    outcomes = [await compressor.maybe_compress() for _ in range(3)]

    assert outcomes == [
        CompressionOutcome.APPLIED,
        CompressionOutcome.APPLIED,
        CompressionOutcome.APPLIED,
    ]
    # 8 轮 + 1 system，压缩 3 次各减 1 轮
    assert memory.round_count == 6
    assert memory.rounds[0].is_system_round is True


async def test_compression_stops_once_below_threshold() -> None:
    """压缩必须收敛：轮次降到阈值以下后不再触发（否则会无限压缩）。"""
    client = FakeLlmClient(fallback=json_response('{"compressed_context": "摘要"}'))
    memory = build_memory(3)  # 1 system + 3 轮 = 4 轮
    compressor = make_compressor(client, memory, threshold=4, compress=2)

    assert await compressor.maybe_compress() is CompressionOutcome.APPLIED
    assert memory.round_count == 3  # 压缩 2 轮 → 1 条摘要，净减 1 轮

    assert await compressor.maybe_compress() is CompressionOutcome.NO_PLAN
    assert memory.round_count == 3
    assert memory.rounds[0].is_system_round is True
