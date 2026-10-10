"""按用途解析模型与思考参数（FR-703、FR-717）。

P1 的行为是"全部继承全局设置"，但把**按用途覆盖**的扩展点先留好：
T3.5 只需往 ``overrides`` 里填配置，不必改动调用方。

默认建议（FR-703）：路径规划与判定用 `deepseek-v4-pro` + `effort=high`；
上下文压缩关闭思考以省时省钱。这些默认值在 :data:`RECOMMENDED_OVERRIDES` 中给出，
但**默认不启用**——是否启用属于设置项，由 T3.5 决定。
"""

from dataclasses import dataclass, replace

from app.contracts.schemas import LlmPurpose, ReasoningEffort
from app.infra.settings import AppSettings


@dataclass(frozen=True)
class ModelChoice:
    """一次调用使用的模型与思考参数。"""

    model: str
    thinking: bool
    reasoning_effort: str
    max_tokens: int
    timeout_seconds: int

    def describe(self) -> str:
        return f"model={self.model} thinking={self.thinking} effort={self.reasoning_effort}"


RECOMMENDED_OVERRIDES: dict[LlmPurpose, dict[str, object]] = {
    LlmPurpose.JUDGE: {"model_field": "model_pro", "reasoning_effort": ReasoningEffort.HIGH},
    LlmPurpose.PLAN: {"model_field": "model_pro", "reasoning_effort": ReasoningEffort.HIGH},
    # 压缩是机械性摘要，关闭思考能显著降低延迟与费用
    LlmPurpose.COMPRESS: {"thinking": False},
}


class ModelPolicy:
    """把全局设置 + 按用途覆盖解析为一次调用的具体参数。"""

    def __init__(
        self,
        settings: AppSettings,
        overrides: dict[LlmPurpose, dict[str, object]] | None = None,
    ) -> None:
        self._settings = settings
        self._overrides = dict(overrides or {})

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def update_settings(self, settings: AppSettings) -> None:
        """设置页保存后同步（配合 AppContext.refresh_settings 使用）。"""
        self._settings = settings

    def set_override(self, purpose: LlmPurpose, values: dict[str, object]) -> None:
        """设置某用途的覆盖项（T3.5 的接入点）。"""
        merged = dict(self._overrides.get(purpose, {}))
        merged.update(values)
        self._overrides[purpose] = merged

    def clear_override(self, purpose: LlmPurpose) -> None:
        self._overrides.pop(purpose, None)

    def resolve(self, purpose: LlmPurpose) -> ModelChoice:
        """解析某用途的模型参数。"""
        settings = self._settings
        override = self._overrides.get(purpose, {})

        model_field = override.get("model_field", "model_default")
        model = settings.model_pro if model_field == "model_pro" else settings.model_default

        effort = override.get("reasoning_effort", settings.reasoning_effort)
        effort_value = effort.value if isinstance(effort, ReasoningEffort) else str(effort)

        thinking = bool(override.get("thinking", settings.thinking_enabled))
        max_tokens = _as_int(override.get("max_tokens"), settings.max_tokens)
        timeout_seconds = _as_int(override.get("timeout_seconds"), settings.timeout_seconds)

        return ModelChoice(
            model=model,
            thinking=thinking,
            reasoning_effort=effort_value,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def with_settings(self, settings: AppSettings) -> "ModelPolicy":
        return ModelPolicy(settings, self._overrides)


def apply_recommended_overrides(policy: ModelPolicy) -> ModelPolicy:
    """启用推荐覆盖（T3.5 在设置项打开时调用）。"""
    for purpose, values in RECOMMENDED_OVERRIDES.items():
        policy.set_override(purpose, dict(values))
    return policy


def replace_choice(choice: ModelChoice, **changes: object) -> ModelChoice:
    """便于测试与设置页做局部调整。"""
    return replace(choice, **changes)  # type: ignore[arg-type]


def _as_int(value: object, fallback: int) -> int:
    """覆盖项来自字典（值类型不受静态检查约束），非法值一律回退，避免运行期崩。"""
    if isinstance(value, bool):
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return fallback
