"""用途策略与用量落库测试（T1.7、FR-703、FR-705）。"""

from app.contracts.schemas import LlmPurpose, ReasoningEffort
from app.infra.llm.protocol import TokenUsage
from app.infra.settings import AppSettings, PurposeOverride
from app.infra.storage.models import UsageRecordRow
from app.main_context import AppContext
from app.services.learning.repository import DomainRepositories
from app.services.llm.policy import ModelPolicy, apply_recommended_overrides, describe_model_plan
from app.services.llm.usage import UsageRecorder, cache_hint

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
# 模型策略落地（T3.5、FR-703）
# --------------------------------------------------------------------------- #


def test_settings_alone_assemble_the_policy() -> None:
    """构造策略时就按设置装配：调用方不必记得"先建对象再补覆盖项"。"""
    policy = ModelPolicy(AppSettings(use_recommended_overrides=True))

    assert policy.resolve(LlmPurpose.JUDGE).model == "deepseek-v4-pro"
    assert policy.resolve(LlmPurpose.JUDGE).reasoning_effort == "high"
    assert policy.resolve(LlmPurpose.PLAN).model == "deepseek-v4-pro"
    assert policy.resolve(LlmPurpose.COMPRESS).thinking is False
    # 推荐项之外的用途仍继承全局设置
    assert policy.resolve(LlmPurpose.TEACH).model == "deepseek-flash"
    assert policy.resolve(LlmPurpose.ASSESS).reasoning_effort == "high"


def test_explicit_override_beats_recommended() -> None:
    settings = AppSettings(
        use_recommended_overrides=True,
        purpose_overrides={
            LlmPurpose.JUDGE: PurposeOverride(model="default", thinking="off", reasoning_effort="low"),
        },
    )
    judge = ModelPolicy(settings).resolve(LlmPurpose.JUDGE)

    assert judge.model == "deepseek-flash"
    assert judge.thinking is False
    assert judge.reasoning_effort == "low"
    # 没被显式覆盖的用途照旧拿推荐项
    assert ModelPolicy(settings).resolve(LlmPurpose.PLAN).model == "deepseek-v4-pro"


def test_reconfigure_applies_saved_settings_without_restart() -> None:
    policy = ModelPolicy(AppSettings())
    assert policy.resolve(LlmPurpose.JUDGE).model == "deepseek-flash"

    policy.reconfigure(AppSettings(use_recommended_overrides=True))
    assert policy.resolve(LlmPurpose.JUDGE).model == "deepseek-v4-pro"

    # 关掉推荐开关后必须回到继承：留着旧覆盖项就是"界面上关了，实际还在用 pro"
    policy.reconfigure(AppSettings())
    assert policy.resolve(LlmPurpose.JUDGE).model == "deepseek-flash"
    assert policy.resolve(LlmPurpose.JUDGE).reasoning_effort == "high"


def test_explicit_override_is_cleared_when_reset_to_inherit() -> None:
    settings = AppSettings(purpose_overrides={LlmPurpose.TEACH: PurposeOverride(model="pro")})
    policy = ModelPolicy(settings)
    assert policy.resolve(LlmPurpose.TEACH).model == "deepseek-v4-pro"

    policy.reconfigure(AppSettings())
    assert policy.resolve(LlmPurpose.TEACH).model == "deepseek-flash"


def test_describe_model_plan_covers_every_purpose() -> None:
    plan = describe_model_plan(ModelPolicy(AppSettings(use_recommended_overrides=True)))

    assert set(plan) == {purpose.value for purpose in LlmPurpose}
    assert plan["judge"] == {"model": "deepseek-v4-pro", "thinking": True, "reasoning_effort": "high"}
    assert plan["compress"]["thinking"] is False
    assert plan["teach"] == {"model": "deepseek-flash", "thinking": True, "reasoning_effort": "high"}


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


# --------------------------------------------------------------------------- #
# 用量聚合（T3.6、FR-705、FR-720）
# --------------------------------------------------------------------------- #


def test_by_purpose_aggregates_each_purpose(context: AppContext) -> None:
    recorder = UsageRecorder(context.db, subject_id="s1")
    recorder.record(LlmPurpose.TEACH, "deepseek-flash", USAGE, duration_ms=120)
    recorder.record(LlmPurpose.TEACH, "deepseek-flash", TokenUsage(prompt_tokens=10, total_tokens=10), duration_ms=30)
    recorder.record(
        LlmPurpose.JUDGE,
        "deepseek-v4-pro",
        TokenUsage(prompt_tokens=8, cached_prompt_tokens=8, total_tokens=8),
        duration_ms=50,
    )

    by_purpose = recorder.by_purpose()

    assert set(by_purpose) == {"teach", "judge"}
    assert by_purpose["teach"].calls == 2
    assert by_purpose["teach"].total_tokens == 160
    assert by_purpose["teach"].duration_ms == 150
    assert by_purpose["judge"].cache_hit_rate == 1.0
    # 与总量口径一致（同一套聚合表达式）
    assert sum(item.total_tokens for item in by_purpose.values()) == recorder.summary().total_tokens


def test_by_purpose_is_scoped_by_subject(context: AppContext) -> None:
    UsageRecorder(context.db, subject_id="a").record(LlmPurpose.TEACH, "m", USAGE)
    UsageRecorder(context.db, subject_id="b").record(LlmPurpose.TEACH, "m", TokenUsage(total_tokens=1))

    assert UsageRecorder(context.db, subject_id="a").by_purpose()["teach"].total_tokens == 150
    assert UsageRecorder(context.db).by_purpose()["teach"].calls == 2


def test_by_subject_sorts_by_total_tokens_and_resolves_title(context: AppContext) -> None:
    repos = DomainRepositories(context.db)
    big = repos.subjects.create("大主题")
    small = repos.subjects.create("小主题")

    recorder = UsageRecorder(context.db)
    recorder.record(LlmPurpose.TEACH, "m", TokenUsage(total_tokens=300), subject_id=small.id)
    recorder.record(LlmPurpose.TEACH, "m", TokenUsage(total_tokens=900), subject_id=big.id)
    # 没有主题归属的调用不进这张表：归不到主题上的用量放进"按主题"就失去意义了
    recorder.record(LlmPurpose.TEACH, "m", TokenUsage(total_tokens=5000))

    items = recorder.by_subject()

    assert [item.subject_id for item in items] == [big.id, small.id]
    assert items[0].title == "大主题"
    assert items[0].summary.total_tokens == 900
    assert items[0].summary.calls == 1


def test_by_subject_falls_back_to_id_when_title_is_gone(context: AppContext) -> None:
    """主题被删掉（级联删记录前）也不能让标题变成空串。"""
    UsageRecorder(context.db).record(LlmPurpose.TEACH, "m", TokenUsage(total_tokens=10), subject_id="ghost")

    items = UsageRecorder(context.db).by_subject()

    assert items[0].title == "ghost"


def test_by_subject_keeps_only_top_ten(context: AppContext) -> None:
    recorder = UsageRecorder(context.db)
    for index in range(12):
        recorder.record(LlmPurpose.TEACH, "m", TokenUsage(total_tokens=index), subject_id=f"s{index}")

    items = recorder.by_subject()

    assert len(items) == 10
    assert items[0].summary.total_tokens == 11
    assert [item.summary.total_tokens for item in items] == sorted(
        [item.summary.total_tokens for item in items], reverse=True
    )


def test_cache_hint_only_when_there_are_calls_but_no_hits(context: AppContext) -> None:
    recorder = UsageRecorder(context.db)

    assert cache_hint(recorder.summary()) is None  # 还没调用过，谈不上命中率异常

    recorder.record(LlmPurpose.TEACH, "m", TokenUsage(prompt_tokens=100, cache_miss_tokens=100, total_tokens=100))
    hint = cache_hint(recorder.summary())
    assert hint is not None
    assert "缓存" in hint

    recorder.record(LlmPurpose.TEACH, "m", TokenUsage(prompt_tokens=100, cached_prompt_tokens=100, total_tokens=100))
    assert cache_hint(recorder.summary()) is None
