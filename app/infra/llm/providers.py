"""供应商能力档案（P6）：同一份 OpenAI 兼容协议，各服务的可选能力并不相同。

**为什么需要它**：`/chat/completions` 的形状是通用的，但各家在这几件事上分叉——

| 能力 | DeepSeek | OpenAI 官方 / 多数兼容服务 |
| --- | --- | --- |
| 思考开关 `thinking` | 有（`{"type": "enabled"}`） | 没有；开启思考靠模型本身 |
| `reasoning_effort` 取值 | low / high / max | minimal / low / medium / high |
| 思维链字段 | `reasoning_content` | `reasoning_content`（vLLM 系）或 `reasoning`（OpenRouter 系） |
| 缓存命中统计 | `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` | `prompt_tokens_details.cached_tokens` |
| 余额接口 | `/user/balance` | 无 |
| 模型元数据 | `/models` 带上下文窗口与档位 | 通常只有 id |
| 空闲时段半价 | 有 | 无（按固定价） |
| 带 tools 时回传思维链 | 硬性要求 | 不要求；回传反而可能被拒 |

把差异集中在一张档案里，请求组装与响应解析就都只读这张表，
而不是散落成一堆 `if provider == ...`。

**默认值选 DeepSeek**：它保持既有安装的行为完全不变（升级后一个字都不变）；
想接别的服务，在设置页把「接口类型」切成 OpenAI 兼容即可，其余能力自动降级。
"""

from dataclasses import dataclass
from enum import StrEnum

from app.contracts.schemas import ReasoningEffort


class ProviderKind(StrEnum):
    """接口类型。取值即设置项 ``provider`` 的合法值。"""

    OPENAI = "openai"
    DEEPSEEK = "deepseek"


@dataclass(frozen=True)
class ProviderProfile:
    """一个供应商的完整能力描述。"""

    kind: ProviderKind
    label: str
    default_base_url: str
    # ---- 请求侧 ----
    send_thinking: bool
    """是否下发 ``thinking: {"type": "enabled"|"disabled"}``。"""
    send_reasoning_effort: bool
    """是否下发 ``reasoning_effort``（仅在"需要更深的推理"时才有意义）。"""
    effort_map: tuple[tuple[str, str], ...]
    """本项目取值 → 该服务取值。缺项表示该档位不受支持，按 high 处理。"""
    echo_reasoning_with_tools: bool
    """请求携带 ``tools`` 时是否必须回传历史 ``reasoning_content``。"""
    # ---- 响应侧 ----
    reasoning_fields: tuple[str, ...]
    """思维链字段名，按顺序尝试（各家不一致，第一个取到值就算）。"""
    cache_hit_fields: tuple[str, ...]
    """顶层缓存命中字段（``usage.<name>``）。"""
    cache_miss_fields: tuple[str, ...]
    """顶层缓存未命中字段；为空时按 ``prompt - cached`` 推算。"""
    cache_details_field: str | None
    """``usage.prompt_tokens_details.<name>`` 形式的缓存命中字段。

    两家都给同一个名字（``cached_tokens``），保留这层是为了在字段形状再次变化时只改一处。
    """
    # ---- 附加能力 ----
    balance_path: str | None
    """余额接口路径；``None`` 表示该服务没有这个接口。"""
    models_metadata: bool
    """``/models`` 是否带上下文窗口、输出上限与档位等元数据。"""
    off_peak_multiplier: float
    """空闲时段的价格系数。DeepSeek 空闲半价，其他服务一律 1.0。"""

    def map_effort(self, effort: str) -> str:
        """把本项目档位映射成该服务接受的取值。"""
        table = dict(self.effort_map)
        return table.get(effort, table.get(ReasoningEffort.HIGH.value, effort))

    def capabilities(self) -> dict[str, bool]:
        """对外（前端、健康检查）暴露的能力开关，用于灰掉不生效的控件。"""
        return {
            "thinking_switch": self.send_thinking,
            "reasoning_effort": self.send_reasoning_effort,
            "balance": self.balance_path is not None,
            "cache_stats": bool(self.cache_hit_fields or self.cache_details_field),
            "model_metadata": self.models_metadata,
            "off_peak_pricing": self.off_peak_multiplier != 1.0,
            "reasoning_echo_with_tools": self.echo_reasoning_with_tools,
        }


_DEEPSEEK = ProviderProfile(
    kind=ProviderKind.DEEPSEEK,
    label="DeepSeek（含思考开关、缓存统计、余额与空闲半价）",
    default_base_url="https://api.deepseek.com",
    send_thinking=True,
    send_reasoning_effort=True,
    effort_map=(
        (ReasoningEffort.LOW.value, "low"),
        (ReasoningEffort.HIGH.value, "high"),
        (ReasoningEffort.MAX.value, "max"),
    ),
    echo_reasoning_with_tools=True,
    reasoning_fields=("reasoning_content",),
    cache_hit_fields=("prompt_cache_hit_tokens",),
    cache_miss_fields=("prompt_cache_miss_tokens",),
    # 网关可能把 DeepSeek 的缓存字段改写成 OpenAI 写法，因此这里也留一个兜底
    cache_details_field="cached_tokens",
    balance_path="/user/balance",
    models_metadata=True,
    off_peak_multiplier=0.5,
)

_OPENAI = ProviderProfile(
    kind=ProviderKind.OPENAI,
    label="OpenAI 兼容（通用：vLLM / Ollama / LM Studio / 各类网关）",
    default_base_url="https://api.openai.com/v1",
    # 通用接口不下发 thinking：这个字段只有 DeepSeek 认，发给别人轻则被忽略、重则 400
    send_thinking=False,
    # 但 reasoning_effort 是 OpenAI 自己的字段，思考模型接受它，因此保留
    send_reasoning_effort=True,
    # 本项目有 low/high/max 三档，OpenAI 没有 max：映射到 high（再高就要靠模型自身的预算）
    effort_map=(
        (ReasoningEffort.LOW.value, "low"),
        (ReasoningEffort.HIGH.value, "high"),
        (ReasoningEffort.MAX.value, "high"),
    ),
    # OpenAI 不接受输入消息里的 reasoning_content；vLLM 等多数实现也只在自己产出的轮次里认
    echo_reasoning_with_tools=False,
    # 两种写法都覆盖：vLLM / Moonshot 用 reasoning_content，OpenRouter 系用 reasoning
    reasoning_fields=("reasoning_content", "reasoning"),
    cache_hit_fields=(),
    cache_miss_fields=(),
    cache_details_field="cached_tokens",
    balance_path=None,
    models_metadata=False,
    off_peak_multiplier=1.0,
)

PROFILES: dict[ProviderKind, ProviderProfile] = {
    ProviderKind.DEEPSEEK: _DEEPSEEK,
    ProviderKind.OPENAI: _OPENAI,
}

DEFAULT_PROVIDER = ProviderKind.DEEPSEEK


def profile_for(kind: ProviderKind | str | None) -> ProviderProfile:
    """按接口类型取档案；未知/空值回退默认（配置损坏不得让应用不可用）。"""
    if isinstance(kind, ProviderKind):
        return PROFILES.get(kind, _DEEPSEEK)
    try:
        return PROFILES[ProviderKind(str(kind or "").strip().lower())]
    except (ValueError, KeyError):
        return PROFILES[DEFAULT_PROVIDER]


def provider_options() -> list[dict[str, object]]:
    """给设置页的选项清单（含能力开关，便于前端灰掉不生效的控件）。"""
    return [
        {
            "value": profile.kind.value,
            "label": profile.label,
            "default_base_url": profile.default_base_url,
            "capabilities": profile.capabilities(),
        }
        for profile in PROFILES.values()
    ]
