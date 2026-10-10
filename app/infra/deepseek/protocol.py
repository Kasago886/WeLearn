"""chat/completions 请求与响应协议（FR-701、FR-705、FR-716、FR-717）。

请求体构建规则来自官方文档，有两条容易踩坑、这里显式处理：

1. **`tools` 只在确有工具时下发**——官方语义是"无 tool 时 tool_choice 默认为 none"，
   但部分兼容端点不接受 `"tools": null`；
2. **`max_tokens` 在 JSON Output 场景必须显式设置**——否则 JSON 可能被中途截断（FR-716）。

另有两条官方限制由上层负责（P1/P3 落地，见 FR-714、FR-718）：
思考模式下不支持 `tool_choice=required` 与指定具体 tool；
思考模式下 `temperature`/`presence_penalty`/`frequency_penalty` 不生效、`top_p` 被钳制到 0.95~1.0。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.contracts.events import UsagePayload
from app.infra.deepseek.errors import LlmError

CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"
BALANCE_PATH = "/user/balance"


def build_endpoint(base_url: str, path: str) -> str:
    """拼接端点；直接拼接而非用 BaseAddress，以保留 base_url 中的子路径（如 /v1）。"""
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise LlmError.config("未配置 DeepSeek API 地址，请在设置中填写。")
    if not url.startswith(("http://", "https://")):
        raise LlmError.config("DeepSeek API 地址必须以 http:// 或 https:// 开头。")
    return f"{url}{path}"


@dataclass(frozen=True)
class TokenUsage:
    """token 用量。字段缺失按 0 处理。"""

    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_miss_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_usage_dict(cls, usage: Mapping[str, Any] | None) -> "TokenUsage | None":
        if not isinstance(usage, Mapping):
            return None

        details = usage.get("completion_tokens_details")

        def as_int(value: Any) -> int:
            return int(value) if isinstance(value, int | float) else 0

        return cls(
            prompt_tokens=as_int(usage.get("prompt_tokens")),
            cached_prompt_tokens=as_int(usage.get("prompt_cache_hit_tokens")),
            cache_miss_tokens=as_int(usage.get("prompt_cache_miss_tokens")),
            completion_tokens=as_int(usage.get("completion_tokens")),
            reasoning_tokens=as_int(details.get("reasoning_tokens")) if isinstance(details, Mapping) else 0,
            total_tokens=as_int(usage.get("total_tokens")),
        )

    def to_payload(self) -> UsagePayload:
        return UsagePayload(
            prompt_tokens=self.prompt_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens,
            cache_miss_tokens=self.cache_miss_tokens,
            completion_tokens=self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens,
            total_tokens=self.total_tokens,
        )


@dataclass(frozen=True)
class StreamEvent:
    """流式事件。

    - 增量事件：`content` 与/或 `reasoning` 有值，`usage` 为 None；
    - 收尾事件：`content` 与 `reasoning` 均为 None，`usage` 有值。
    """

    content: str | None = None
    reasoning: str | None = None
    usage: TokenUsage | None = None

    @property
    def is_usage_only(self) -> bool:
        return self.content is None and self.reasoning is None and self.usage is not None


def build_chat_request(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    stream: bool,
    thinking: bool,
    reasoning_effort: str,
    max_tokens: int,
    json_object: bool = False,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """构建 chat/completions 请求体。"""
    if not model.strip():
        raise LlmError.config("未配置模型名，请在设置中填写。")

    body: dict[str, Any] = {
        "model": model.strip(),
        "messages": list(messages),
        "stream": stream,
        "thinking": {"type": "enabled" if thinking else "disabled"},
    }

    # 思考关闭时下发 effort 没有意义
    if thinking and reasoning_effort:
        body["reasoning_effort"] = reasoning_effort

    if max_tokens > 0:
        body["max_tokens"] = max_tokens

    if json_object:
        body["response_format"] = {"type": "json_object"}

    if tools:
        body["tools"] = list(tools)

    return body


def extract_delta(chunk: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """从流式数据块中取出正文增量与思维链增量。"""
    choices = chunk.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
        return None, None

    first = choices[0]
    if not isinstance(first, Mapping):
        return None, None

    delta = first.get("delta")
    if not isinstance(delta, Mapping):
        return None, None

    content = delta.get("content")
    reasoning = delta.get("reasoning_content")
    return (
        content if isinstance(content, str) and content else None,
        reasoning if isinstance(reasoning, str) and reasoning else None,
    )


def extract_usage(chunk: Mapping[str, Any]) -> TokenUsage | None:
    """从流式数据块中取出用量（仅在最后一个数据块出现）。"""
    usage = chunk.get("usage")
    return TokenUsage.from_usage_dict(usage if isinstance(usage, Mapping) else None)


@dataclass(frozen=True)
class BlockingResult:
    """非流式响应。"""

    content: str
    reasoning: str | None
    usage: TokenUsage | None


def parse_blocking_response(body: Mapping[str, Any]) -> BlockingResult:
    """解析非流式响应；结构不符时抛可诊断的协议错误。"""
    choices = body.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
        raise LlmError.protocol("响应中缺少 choices[0]，无法解析模型输出。")

    first = choices[0]
    message = first.get("message") if isinstance(first, Mapping) else None
    if not isinstance(message, Mapping):
        raise LlmError.protocol("响应中缺少 choices[0].message，无法解析模型输出。")

    raw_content = message.get("content")
    # 官方已知问题：JSON Output 有概率返回空 content（FR-716）；空串按空内容返回，由上层重试
    content = raw_content if isinstance(raw_content, str) else ""

    raw_reasoning = message.get("reasoning_content")
    reasoning = raw_reasoning if isinstance(raw_reasoning, str) and raw_reasoning else None

    usage = TokenUsage.from_usage_dict(body.get("usage") if isinstance(body.get("usage"), Mapping) else None)
    return BlockingResult(content=content, reasoning=reasoning, usage=usage)


@dataclass(frozen=True)
class ModelInfo:
    """`GET /models` 返回的模型元数据（FR-702）。"""

    id: str
    name: str
    context_window: int | None = None
    max_output_tokens: int | None = None
    effort_levels: tuple[str, ...] = ()
    default_effort: str | None = None


def parse_models_response(body: Mapping[str, Any]) -> list[ModelInfo]:
    """解析模型列表。"""
    data = body.get("data")
    if not isinstance(data, Sequence) or isinstance(data, str):
        raise LlmError.protocol("`/models` 响应结构不符合预期。")

    models: list[ModelInfo] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue

        raw_effort = item.get("effort")
        effort: Mapping[str, Any] = raw_effort if isinstance(raw_effort, Mapping) else {}
        levels = effort.get("supported_levels")
        display = item.get("name")

        effort_levels: tuple[str, ...] = ()
        if isinstance(levels, Sequence) and not isinstance(levels, str):
            effort_levels = tuple(str(level) for level in levels)

        models.append(
            ModelInfo(
                id=model_id,
                name=display if isinstance(display, str) and display else model_id,
                context_window=item.get("context_window") if isinstance(item.get("context_window"), int) else None,
                max_output_tokens=(
                    item.get("max_output_tokens") if isinstance(item.get("max_output_tokens"), int) else None
                ),
                effort_levels=effort_levels,
                default_effort=effort.get("default_level") if isinstance(effort.get("default_level"), str) else None,
            )
        )

    return models
