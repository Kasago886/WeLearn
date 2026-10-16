"""设置读写与回退测试（FR-901、FR-902、AC-23、AC-27）。"""

import json
from pathlib import Path

from app.contracts.schemas import LlmPurpose
from app.infra.settings import (
    AppSettings,
    PurposeOverride,
    SettingsService,
    purpose_overrides_payload,
    resolve_data_dir,
)
from app.infra.storage.db import Database
from app.infra.storage.models import AppSettingsRow
from app.main_context import AppContext


def _write_raw(context: AppContext, **values: object) -> None:
    with context.db.session() as session:
        row = session.get(AppSettingsRow, 1) or AppSettingsRow(id=1)
        for key, value in values.items():
            setattr(row, key, value)
        session.add(row)


def test_fresh_database_yields_defaults(context: AppContext) -> None:
    settings = context.settings
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model_default == "deepseek-flash"
    assert settings.model_pro == "deepseek-v4-pro"
    assert (settings.path_min_nodes, settings.path_max_nodes) == (3, 15)
    assert settings.assess_max_turns == 6
    assert settings.reasoning_effort.value == "high"
    assert settings.data_dir == str(Path(context.paths.root))


def test_ensure_defaults_persists_row(context: AppContext) -> None:
    with context.db.session() as session:
        row = session.get(AppSettingsRow, 1)
    assert row is not None
    assert row.model_default == "deepseek-flash"


def test_invalid_values_fall_back_per_field(context: AppContext) -> None:
    """单字段非法只影响该字段（FR-902）：不能因为一个坏值让整份配置失效。"""
    _write_raw(
        context,
        timeout_seconds=5,  # 越界（30~600）
        assess_max_turns=99,  # 越界（3~12）
        reasoning_effort="turbo",  # 非法枚举
        log_level="VERBOSE",  # 非法枚举
        base_url="   ",  # 空值
        max_tokens="不是数字",  # 类型错误
        model_default="deepseek-v4-pro",  # 合法：应当保留
        price_table_json="{坏 JSON",  # 无法解析
    )

    settings, diagnostics = context.settings_service.load_with_diagnostics()

    assert settings.timeout_seconds == 60
    assert settings.assess_max_turns == 6
    assert settings.reasoning_effort.value == "high"
    assert settings.log_level == "INFO"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.max_tokens == 8192
    assert settings.price_table == {}
    assert settings.model_default == "deepseek-v4-pro"

    assert set(diagnostics.rejected_fields) >= {
        "timeout_seconds",
        "assess_max_turns",
        "reasoning_effort",
        "log_level",
        "max_tokens",
        "price_table",
    }


def test_legal_custom_values_survive(context: AppContext) -> None:
    saved = context.settings_service.save(
        AppSettings(timeout_seconds=180, assess_max_turns=10, path_min_nodes=4, path_max_nodes=9)
    )
    assert saved.timeout_seconds == 180

    reloaded = context.settings_service.load()
    assert reloaded.assess_max_turns == 10
    assert (reloaded.path_min_nodes, reloaded.path_max_nodes) == (4, 9)


def test_inverted_node_range_is_repaired(context: AppContext) -> None:
    _write_raw(context, path_min_nodes=15, path_max_nodes=3)

    settings, diagnostics = context.settings_service.load_with_diagnostics()

    assert (settings.path_min_nodes, settings.path_max_nodes) == (3, 15)
    assert diagnostics.repaired_fields == ["path_min_nodes", "path_max_nodes"]


def test_save_refuses_inverted_range(context: AppContext) -> None:
    saved = context.settings_service.save(AppSettings(path_min_nodes=15, path_max_nodes=3))
    assert (saved.path_min_nodes, saved.path_max_nodes) == (3, 15)


def test_env_overrides_take_precedence(context: AppContext, env: dict[str, str]) -> None:
    from app.main_context import build_context

    custom_env = {
        **env,
        "WELEARN_BASE_URL": "https://proxy.example.com/v1",
        "WELEARN_MODEL_DEFAULT": "deepseek-v4-pro",
        "WELEARN_LOG_LEVEL": "WARNING",
    }
    ctx = build_context(custom_env)
    try:
        settings, diagnostics = ctx.settings_service.load_with_diagnostics()
        assert settings.base_url == "https://proxy.example.com/v1"
        assert settings.model_default == "deepseek-v4-pro"
        assert settings.log_level == "WARNING"
        assert set(diagnostics.env_overridden) == {"base_url", "model_default", "log_level"}
    finally:
        ctx.close()


def test_invalid_env_value_is_ignored(env: dict[str, str]) -> None:
    from app.main_context import build_context

    ctx = build_context({**env, "WELEARN_LOG_LEVEL": "NOISY"})
    try:
        assert ctx.settings.log_level == "INFO"
    finally:
        ctx.close()


def test_data_dir_follows_env_not_stored_value(tmp_path: Path) -> None:
    """data_dir 只由环境变量决定，避免"改了数据目录就找不到自己的库"。"""
    data_dir = tmp_path / "custom"
    db = Database.from_path(data_dir / "welearn.db")
    db.init()
    service = SettingsService(db, {"WELEARN_DATA_DIR": str(data_dir)})
    service.ensure_defaults()

    settings = service.save(AppSettings(data_dir="D:/somewhere/else"))
    assert settings.data_dir == str(data_dir)

    db.dispose()


def test_resolve_data_dir_default() -> None:
    assert resolve_data_dir({}) == Path("./data")
    assert resolve_data_dir({"WELEARN_DATA_DIR": "   "}) == Path("./data")
    assert resolve_data_dir({"WELEARN_DATA_DIR": "  ./x "}) == Path("./x")


def test_corrupt_settings_json_does_not_break_startup(context: AppContext) -> None:
    """价格表存的是 JSON 文本，写入坏内容时只能丢该字段（AC-23）。"""
    with context.db.session() as session:
        row = session.get(AppSettingsRow, 1)
        assert row is not None
        row.price_table_json = "[]"  # 合法 JSON 但不是对象
    settings, _ = context.settings_service.load_with_diagnostics()
    assert settings.price_table == {}
    assert json.loads(json.dumps(settings.price_table)) == {}


# --------------------------------------------------------------------------- #
# P3C：采样参数、审计开关与按用途覆盖（FR-703、FR-715、FR-718）
# --------------------------------------------------------------------------- #


def test_sampling_and_audit_settings_round_trip(context: AppContext) -> None:
    context.settings_service.save(
        AppSettings(temperature=0.7, top_p=0.9, audit_enabled=True, use_recommended_overrides=True)
    )

    reloaded = context.settings_service.load()
    assert reloaded.temperature == 0.7
    assert reloaded.top_p == 0.9
    assert reloaded.audit_enabled is True
    assert reloaded.use_recommended_overrides is True


def test_purpose_overrides_round_trip(context: AppContext) -> None:
    """按用途覆盖必须落盘再读回一致（FR-703），否则重启后档位会悄悄变回默认。"""
    context.settings_service.save(
        AppSettings(
            purpose_overrides={
                LlmPurpose.JUDGE: PurposeOverride(model="pro", reasoning_effort="max"),
                LlmPurpose.COMPRESS: PurposeOverride(thinking="off"),
            }
        )
    )

    reloaded = context.settings_service.load()
    assert reloaded.purpose_overrides[LlmPurpose.JUDGE].model == "pro"
    assert reloaded.purpose_overrides[LlmPurpose.JUDGE].reasoning_effort == "max"
    assert reloaded.purpose_overrides[LlmPurpose.COMPRESS].thinking == "off"
    # 没配置的用途就是"继承"，不能凭空多出一项
    assert LlmPurpose.TEACH not in reloaded.purpose_overrides


def test_purpose_overrides_payload_fills_every_purpose(context: AppContext) -> None:
    """对外形状固定：五个用途都在，"继承"显式写出来，前端不必自己补默认值。"""
    payload = purpose_overrides_payload(context.settings_service.load())

    assert set(payload) == {"assess", "plan", "teach", "judge", "compress"}
    assert payload["judge"] == {"model": "inherit", "thinking": "inherit", "reasoning_effort": "inherit"}


def test_dirty_purpose_overrides_fall_back_to_all_inherit(context: AppContext) -> None:
    """脏数据一律回退"全部继承"，且绝不抛异常（FR-902 的既有做法）。"""
    broken_values = [
        "{坏 JSON",
        '["assess"]',  # 合法 JSON 但不是对象
        '{"assess": {"model": "gpt-4"}}',  # 取值非法
        '{"不存在的用途": {"model": "pro"}}',  # 用途非法
        '{"assess": {"reasoning_effort": "turbo"}}',
    ]
    for broken in broken_values:
        _write_raw(context, purpose_overrides_json=broken)
        settings, diagnostics = context.settings_service.load_with_diagnostics()
        assert settings.purpose_overrides == {}, broken
        assert "purpose_overrides" in diagnostics.rejected_fields, broken


def test_out_of_range_sampling_values_fall_back(context: AppContext) -> None:
    """越界的采样参数只影响该字段（FR-718 的取值范围是 0~2 / 0.01~1）。"""
    _write_raw(context, temperature=3.0, top_p=0.0)

    settings, diagnostics = context.settings_service.load_with_diagnostics()
    assert settings.temperature == 1.0
    assert settings.top_p == 1.0
    assert set(diagnostics.rejected_fields) >= {"temperature", "top_p"}
