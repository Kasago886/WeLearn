"""用途策略与用量落库测试（T1.7、FR-703、FR-705）。"""

from app.contracts.schemas import LlmPurpose, ReasoningEffort
from app.infra.deepseek.protocol import TokenUsage
from app.infra.settings import AppSettings
from app.infra.storage.models import UsageRecordRow
from app.main_context import AppContext
from app.services.llm.policy import ModelPolicy, apply_recommended_overrides
from app.services.llm.usage import UsageRecorder

# --------------------------------------------------------------------------- #
# ModelPolicy
# --------------------------------------------------------------------------- #


def test_default_policy_inherits_global_settings() -> None:
    settings = AppSettings(
        model_default="deepseek-flash",
        model_pro="deepseek-v4-pro",
        thinking_enabled=True,
        reasoning_effort=ReasoningEffort.MAX,
        max_tokens=2048,
        timeout_seconds=90,
    )
    policy = ModelPolicy(settings)

    for purpose in LlmPurpose:
        choice = policy.resolve(purpose)
        assert choice.model == "deepseek-flash"
        assert choice.thinking is True
        assert choice.reasoning_effort == "max"
        assert choice.max_tokens == 2048
        assert choice.timeout_seconds == 90


def test_override_can_switch_model_and_disable_thinking() -> None:
    policy = ModelPolicy(AppSettings())
    policy.set_override(LlmPurpose.COMPRESS, {"thinking": False})

    compress = policy.resolve(LlmPurpose.COMPRESS)
    assert compress.thinking is False

    assert policy.resolve(LlmPurpose.TEACH).thinking is True


def test_recommended_overrides_use_pro_model_for_judge() -> None:
    policy = apply_recommended_overrides(ModelPolicy(AppSettings()))

    judge = policy.resolve(LlmPurpose.JUDGE)
    assert judge.model == "deepseek-v4-pro"
    assert judge.reasoning_effort == "high"

    assert policy.resolve(LlmPurpose.PLAN).model == "deepseek-v4-pro"
    assert policy.resolve(LlmPurpose.COMPRESS).thinking is False
    # 教学仍用默认模型
    assert policy.resolve(LlmPurpose.TEACH).model == "deepseek-flash"


def test_override_replacement_and_clearing() -> None:
    policy = ModelPolicy(AppSettings())
    policy.set_override(LlmPurpose.PLAN, {"model_field": "model_pro"})
    policy.set_override(LlmPurpose.PLAN, {"max_tokens": 1234})

    plan = policy.resolve(LlmPurpose.PLAN)
    assert plan.model == "deepseek-v4-pro"  # 第一次的设置保留
    assert plan.max_tokens == 1234  # 第二次合并进来

    policy.clear_override(LlmPurpose.PLAN)
    assert policy.resolve(LlmPurpose.PLAN).model == "deepseek-flash"


def test_policy_reflects_updated_settings() -> None:
    policy = ModelPolicy(AppSettings())
    assert policy.resolve(LlmPurpose.TEACH).model == "deepseek-flash"

    policy.update_settings(AppSettings(model_default="deepseek-v4-pro"))

    assert policy.resolve(LlmPurpose.TEACH).model == "deepseek-v4-pro"


def test_choice_description() -> None:
    described = ModelPolicy(AppSettings()).resolve(LlmPurpose.TEACH).describe()
    assert "model=deepseek-flash" in described
    assert "thinking=True" in described


# --------------------------------------------------------------------------- #
# UsageRecorder
# --------------------------------------------------------------------------- #

USAGE = TokenUsage(
    prompt_tokens=100,
    cached_prompt_tokens=64,
    cache_miss_tokens=36,
    completion_tokens=50,
    reasoning_tokens=20,
    total_tokens=150,
)


def test_record_and_summarise(context: AppContext) -> None:
    recorder = UsageRecorder(context.db, subject_id="subject-1")

    recorder.record(LlmPurpose.TEACH, "deepseek-flash", USAGE)
    recorder.record(LlmPurpose.JUDGE, "deepseek-v4-pro", TokenUsage(prompt_tokens=10, total_tokens=10))

    summary = recorder.summary()
    assert summary.calls == 2
    assert summary.total_tokens == 160
    assert summary.prompt_tokens == 110
    assert summary.cached_prompt_tokens == 64
    assert summary.cache_miss_tokens == 36
    assert summary.reasoning_tokens == 20
    assert summary.cache_hit_rate == 64 / 110


def test_records_purpose_and_model_per_row(context: AppContext) -> None:
    recorder = UsageRecorder(context.db)
    recorder.record(LlmPurpose.COMPRESS, "deepseek-flash", USAGE)

    with context.db.session() as session:
        row = session.query(UsageRecordRow).one()

    assert row.purpose == "compress"
    assert row.model == "deepseek-flash"
    assert row.cached_prompt_tokens == 64


def test_records_zero_row_when_usage_missing(context: AppContext) -> None:
    """用量缺失也要留痕：这样"这次调用发生过"这一事实不会丢。"""
    recorder = UsageRecorder(context.db)
    recorder.record(LlmPurpose.TEACH, "deepseek-flash", None)

    with context.db.session() as session:
        row = session.query(UsageRecordRow).one()

    assert row.total_tokens == 0
    assert row.prompt_tokens == 0


def test_summary_is_scoped_by_subject(context: AppContext) -> None:
    UsageRecorder(context.db, subject_id="a").record(LlmPurpose.TEACH, "m", USAGE)
    UsageRecorder(context.db, subject_id="b").record(LlmPurpose.TEACH, "m", TokenUsage(total_tokens=1))

    assert UsageRecorder(context.db, subject_id="a").summary().total_tokens == 150
    assert UsageRecorder(context.db).summary().calls == 2


def test_cache_hit_rate_of_empty_summary_is_zero(context: AppContext) -> None:
    assert UsageRecorder(context.db).summary().cache_hit_rate == 0.0


def test_from_mapping_parses_raw_usage() -> None:
    row = UsageRecorder.from_mapping(
        LlmPurpose.TEACH,
        "deepseek-flash",
        {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    )

    assert row.total_tokens == 12
    assert row.completion_tokens == 7
