"""按用途解析模型与思考参数（FR-703、FR-717）。

默认行为是"全部继承全局设置"，按用途覆盖只有两类来源：
推荐项（:data:`RECOMMENDED_OVERRIDES`，由设置开关 ``use_recommended_overrides`` 决定是否启用）
与用户在设置页显式选择的覆盖项。**显式项压过推荐项**：推荐只是起点，用户选了就得听用户的。
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


def overrides_from_settings(settings: AppSettings) -> dict[LlmPurpose, dict[str, object]]:
    """按设置装配覆盖项：推荐项打底，显式覆盖压顶（FR-703）。"""
    overrides: dict[LlmPurpose, dict[str, object]] = {}
    if settings.use_recommended_overrides:
        overrides.update({purpose: dict(values) for purpose, values in RECOMMENDED_OVERRIDES.items()})

    for purpose, override in settings.purpose_overrides.items():
        values = override.to_policy_values()
        if values:
            overrides.setdefault(purpose, {}).update(values)

    return overrides


class ModelPolicy:
    """把全局设置 + 按用途覆盖解析为一次调用的具体参数。"""

    def __init__(
        self,
        settings: AppSettings,
        overrides: dict[LlmPurpose, dict[str, object]] | None = None,
    ) -> None:
        self._settings = settings
        # 不传 overrides 时按设置推导：调用方（stack / AppContext）在构造处就不必记得"还要装配一次"
        self._overrides = dict(overrides) if overrides is not None else overrides_from_settings(settings)

    @property
    def settings(self) -> AppSettings:
        return self._settings

    def update_settings(self, settings: AppSettings) -> None:
        """只同步全局设置，保留已装配的覆盖项。"""
        self._settings = settings

    def reconfigure(self, settings: AppSettings) -> None:
        """按新设置重新装配覆盖项（设置保存后调用，无需重启，FR-703）。

        覆盖项**整体重建**而不是增量合并：关掉推荐开关、或把某项改回"继承"之后，
        旧覆盖项若还留着，界面显示与实际行为就会不一致。
        """
        self._settings = settings
        self._overrides = overrides_from_settings(settings)

    def set_override(self, purpose: LlmPurpose, values: dict[str, object]) -> None:
        """设置某用途的覆盖项（推荐项与显式覆盖都经由此处）。"""
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
    """启用推荐覆盖（设置项 ``use_recommended_overrides`` 打开时调用）。"""
    for purpose, values in RECOMMENDED_OVERRIDES.items():
        policy.set_override(purpose, dict(values))
    return policy


def describe_model_plan(policy: ModelPolicy) -> dict[str, dict[str, object]]:
    """每个用途**实际**会用的参数（FR-703 的可核对性）。

    只读展示是必需的：覆盖项写进设置里不等于生效（可能被推荐项压过、也可能因为"继承"
    而回落到全局），前端不把解析结果摊开，用户就只能靠猜。
    """
    plan: dict[str, dict[str, object]] = {}
    for purpose in LlmPurpose:
        choice = policy.resolve(purpose)
        plan[purpose.value] = {
            "model": choice.model,
            "thinking": choice.thinking,
            "reasoning_effort": choice.reasoning_effort,
        }
    return plan


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
