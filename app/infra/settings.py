"""应用设置（FR-901、FR-902）。

核心约定：**配置损坏不得导致应用不可用**。读取时逐字段校验，任何缺失、空值、
类型错误或越界值都只影响该字段（回退默认），并在日志中记录，绝不让整个应用启动失败。

`data_dir` 是例外：它决定数据库与日志的位置，只能由环境变量或内置默认值决定
（否则"改了数据目录就找不到自己的库"）。读取时始终用生效值覆盖，界面上应展示为只读。
"""

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from sqlalchemy import select

from app.contracts.schemas import ReasoningEffort
from app.infra.logging.queue_logger import CHANNEL_APP, get_logger
from app.infra.storage.db import Database
from app.infra.storage.models import AppSettingsRow

DEFAULT_DATA_DIR = "./data"

ENV_BASE_URL = "WELEARN_BASE_URL"
ENV_MODEL_DEFAULT = "WELEARN_MODEL_DEFAULT"
ENV_MODEL_PRO = "WELEARN_MODEL_PRO"
ENV_LOG_LEVEL = "WELEARN_LOG_LEVEL"
ENV_DATA_DIR = "WELEARN_DATA_DIR"


def resolve_data_dir(env: Mapping[str, str] | None = None) -> Path:
    """解析数据根目录：环境变量优先，其次内置默认值（D3）。"""
    source = env if env is not None else os.environ
    raw = (source.get(ENV_DATA_DIR) or "").strip() or DEFAULT_DATA_DIR
    return Path(raw).expanduser()


class AppSettings(BaseModel):
    """可持久化的应用设置。默认值即 SRS 第 11 章确认后的默认值。"""

    base_url: str = Field(default="https://api.deepseek.com", min_length=1)
    model_default: str = Field(default="deepseek-flash", min_length=1)
    model_pro: str = Field(default="deepseek-v4-pro", min_length=1)
    thinking_enabled: bool = True
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    # 官方要求 JSON Output 必须显式设置 max_tokens 以防截断（FR-716）
    max_tokens: int = Field(default=8192, ge=256, le=393216)
    timeout_seconds: int = Field(default=60, ge=30, le=600)
    max_retries: int = Field(default=2, ge=0, le=5)
    assess_max_turns: int = Field(default=6, ge=3, le=12)
    path_min_nodes: int = Field(default=3, ge=3, le=15)
    path_max_nodes: int = Field(default=15, ge=3, le=15)
    node_max_retries: int = Field(default=2, ge=0, le=5)
    compress_threshold_rounds: int = Field(default=40, ge=4)
    compress_round_count: int = Field(default=10, ge=1)
    # Q8 决策：思维链默认收起
    show_reasoning: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # 生效值，只读（见模块文档）
    data_dir: str = DEFAULT_DATA_DIR
    price_table: dict[str, float] = Field(default_factory=dict)

    def paths_inverted(self) -> bool:
        """路径节点数区间是否颠倒。"""
        return self.path_min_nodes > self.path_max_nodes


@dataclass
class SettingsDiagnostics:
    """一次加载的诊断信息，用于测试与日志（FR-902 的可验证性）。"""

    rejected_fields: list[str] = field(default_factory=list)
    env_overridden: list[str] = field(default_factory=list)
    repaired_fields: list[str] = field(default_factory=list)


_ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    # 字段名 -> (环境变量名, 类型)
    "base_url": (ENV_BASE_URL, "str"),
    "model_default": (ENV_MODEL_DEFAULT, "str"),
    "model_pro": (ENV_MODEL_PRO, "str"),
    "log_level": (ENV_LOG_LEVEL, "str"),
}


def _sanitize(raw: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """逐字段校验：只保留合法值，非法字段名进拒绝清单。"""
    clean: dict[str, Any] = {}
    rejected: list[str] = []

    for name, field_info in AppSettings.model_fields.items():
        if name not in raw or name == "data_dir":
            continue
        value = raw[name]
        if value is None or (isinstance(value, str) and not value.strip()):
            continue

        if name == "price_table" and isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                rejected.append(name)
                continue

        try:
            adapter: TypeAdapter[Any] = TypeAdapter(field_info.rebuild_annotation())
            clean[name] = adapter.validate_python(value)
        except ValidationError:
            rejected.append(name)

    return clean, rejected


class SettingsService:
    """设置的读写入口。"""

    def __init__(self, db: Database, env: Mapping[str, str] | None = None) -> None:
        self._db = db
        self._env = env if env is not None else os.environ
        self._log = get_logger(CHANNEL_APP)

    def load_with_diagnostics(self) -> tuple[AppSettings, SettingsDiagnostics]:
        """读取设置并返回诊断信息。"""
        diagnostics = SettingsDiagnostics()
        raw = self._read_raw()

        clean, rejected = _sanitize(raw)
        diagnostics.rejected_fields = rejected
        settings = AppSettings(**clean)

        if settings.paths_inverted():
            defaults = AppSettings()
            diagnostics.repaired_fields = ["path_min_nodes", "path_max_nodes"]
            self._log.warning("设置中的路径节点数区间颠倒，已回退默认值")
            settings = settings.model_copy(
                update={
                    "path_min_nodes": defaults.path_min_nodes,
                    "path_max_nodes": defaults.path_max_nodes,
                }
            )

        settings, overridden = self._apply_env(settings)
        diagnostics.env_overridden = overridden

        if diagnostics.rejected_fields:
            self._log.warning(f"设置中以下字段非法已回退默认值: {', '.join(diagnostics.rejected_fields)}")

        # data_dir 始终使用生效值
        settings = settings.model_copy(update={"data_dir": str(resolve_data_dir(self._env))})
        return settings, diagnostics

    def load(self) -> AppSettings:
        """读取设置（含所有回退与覆盖规则）。"""
        settings, _ = self.load_with_diagnostics()
        return settings

    def save(self, settings: AppSettings) -> AppSettings:
        """保存设置并返回生效后的值。data_dir 不随保存变更（见模块文档）。"""
        effective_dir = resolve_data_dir(self._env)
        to_store = settings.model_copy(update={"data_dir": str(effective_dir)})

        if to_store.paths_inverted():
            defaults = AppSettings()
            self._log.warning("拒绝保存颠倒的路径节点数区间，已回退默认值")
            to_store = to_store.model_copy(
                update={
                    "path_min_nodes": defaults.path_min_nodes,
                    "path_max_nodes": defaults.path_max_nodes,
                }
            )

        with self._db.session() as session:
            row = session.get(AppSettingsRow, 1) or AppSettingsRow(id=1)
            row.base_url = to_store.base_url
            row.model_default = to_store.model_default
            row.model_pro = to_store.model_pro
            row.thinking_enabled = to_store.thinking_enabled
            row.reasoning_effort = to_store.reasoning_effort.value
            row.max_tokens = to_store.max_tokens
            row.timeout_seconds = to_store.timeout_seconds
            row.max_retries = to_store.max_retries
            row.assess_max_turns = to_store.assess_max_turns
            row.path_min_nodes = to_store.path_min_nodes
            row.path_max_nodes = to_store.path_max_nodes
            row.node_max_retries = to_store.node_max_retries
            row.compress_threshold_rounds = to_store.compress_threshold_rounds
            row.compress_round_count = to_store.compress_round_count
            row.show_reasoning = to_store.show_reasoning
            row.log_level = to_store.log_level
            row.data_dir = to_store.data_dir
            row.price_table_json = json.dumps(to_store.price_table, ensure_ascii=False)
            session.add(row)

        self._log.info("设置已保存")
        return self.load()

    def ensure_defaults(self) -> AppSettings:
        """首次启动时确保设置行存在（对应 ReferenceCode 的"文件不存在则写入默认"）。"""
        if not self._read_raw():
            return self.save(AppSettings())
        return self.load()

    # ------------------------------------------------------------------ #

    def _read_raw(self) -> dict[str, Any]:
        with self._db.session() as session:
            row = session.scalar(select(AppSettingsRow).where(AppSettingsRow.id == 1))
            if row is None:
                return {}
            return {
                "base_url": row.base_url,
                "model_default": row.model_default,
                "model_pro": row.model_pro,
                "thinking_enabled": row.thinking_enabled,
                "reasoning_effort": row.reasoning_effort,
                "max_tokens": row.max_tokens,
                "timeout_seconds": row.timeout_seconds,
                "max_retries": row.max_retries,
                "assess_max_turns": row.assess_max_turns,
                "path_min_nodes": row.path_min_nodes,
                "path_max_nodes": row.path_max_nodes,
                "node_max_retries": row.node_max_retries,
                "compress_threshold_rounds": row.compress_threshold_rounds,
                "compress_round_count": row.compress_round_count,
                "show_reasoning": row.show_reasoning,
                "log_level": row.log_level,
                "price_table": row.price_table_json,
            }

    def _apply_env(self, settings: AppSettings) -> tuple[AppSettings, list[str]]:
        """环境变量覆盖（FR-707 对 Key 的优先级规则，对连接配置同样适用）。"""
        updates: dict[str, Any] = {}
        overridden: list[str] = []

        for name, (env_name, _kind) in _ENV_OVERRIDES.items():
            raw = (self._env.get(env_name) or "").strip()
            if not raw:
                continue
            try:
                adapter: TypeAdapter[Any] = TypeAdapter(
                    AppSettings.model_fields[name].rebuild_annotation()
                )
                value = adapter.validate_python(raw)
            except ValidationError:
                self._log.warning(f"环境变量 {env_name} 的值非法，已忽略")
                continue
            if getattr(settings, name) != value:
                updates[name] = value
                overridden.append(name)

        if not overridden:
            return settings, overridden

        self._log.info(f"以下设置被环境变量覆盖: {', '.join(overridden)}")
        return settings.model_copy(update=updates), overridden
