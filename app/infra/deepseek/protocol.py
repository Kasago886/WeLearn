"""chat/completions 请求与响应协议（FR-701、FR-705、FR-716、FR-717、FR-718）。

请求体构建规则来自官方文档，有两条容易踩坑、这里显式处理：

1. **`tools` 只在确有工具时下发**——官方语义是"无 tool 时 tool_choice 默认为 none"，
   但部分兼容端点不接受 `"tools": null`；
2. **`max_tokens` 在 JSON Output 场景必须显式设置**——否则 JSON 可能被中途截断（FR-716）。

采样参数（FR-718）按官方限制处理，这些限制体现在"下发什么"上：

- 思考模式下 `temperature` / `presence_penalty` / `frequency_penalty` **不生效**
  （设置不报错但会被忽略）——仍然照常下发 `temperature`，因为同一个值在关闭思考模式的
  用途（如上下文压缩）下是有效的；
- `top_p` **仅在思考模式下生效**，且有效范围被官方钳制到 `0.95`–`1.0`（低于 0.95 按 0.95 处理），
  因此界面必须标注"当前配置下该参数不生效或已被钳制"，否则用户会误以为调参无效是 bug；
- `presence_penalty` / `frequency_penalty` **本项目不使用**，一律不下发。

工具调用（FR-714）在这两层处理：

- **请求侧**：给了非空 `tools` 才下发 `tools`，**永远不下发 `tool_choice`**。
  原因是官方限制——思考模式下 `tool_choice="required"` 与"指定具体某个工具"都会返回 400，
  而本项目的教学调用可能就处在思考模式，写死 `tool_choice` 等于让一条正常请求随机 400；
  不下发时默认值是 `auto`，模型自己决定是否调用，这恰好也是我们要的语义。
- **响应侧**：流式按 `index` 累积 `delta.tool_calls`（`id` / `name` 只取首个非空块，
  `arguments` 逐块拼接），非流式直接读 `message.tool_calls`，两者都归一成
  :class:`ToolCall`，上层因此不必区分两种传输方式。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.contracts.events import UsagePayload
from app.infra.deepseek.errors import LlmError

CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"
BALANCE_PATH = "/user/balance"

# 官方默认值：不显式配置时保持与"不下发"等价的行为
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0


@dataclass(frozen=True)
class SamplingOptions:
    """采样参数（FR-718）。默认值即官方默认，避免"顺手下发 0 温度"这类意外。"""

    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P


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
class ToolCall:
    """模型请求的一次工具调用（FR-714）。

    ``arguments`` 保留**原始 JSON 字符串**：官方返回的就是"未解析的字符串"，
    在协议层解析会把"模型给了非法 JSON"变成"协议解析失败"，而那种情况应该作为
    工具结果回给模型让它自己纠正，不是让整轮失败。
    """

    id: str
    name: str
    arguments: str = ""

    def to_payload(self) -> dict[str, Any]:
        """还原成放回 history 的 `assistant.tool_calls[]` 元素。"""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class StreamEvent:
    """流式事件。

    - 增量事件：`content` 与/或 `reasoning` 有值，`usage` 为 None；
    - 工具调用事件：`tool_calls` 有值（流结束时一次性给出累积结果）；
    - 收尾事件：`content` / `reasoning` / `tool_calls` 均为 None，`usage` 有值。
    """

    content: str | None = None
    reasoning: str | None = None
    usage: TokenUsage | None = None
    tool_calls: tuple[ToolCall, ...] | None = None

    @property
    def is_usage_only(self) -> bool:
        return (
            self.content is None
            and self.reasoning is None
            and self.tool_calls is None
            and self.usage is not None
        )


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
    sampling: SamplingOptions | None = None,
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

    # 采样参数：只在调用方给出时下发（思考模式下会被官方忽略/钳制，见模块文档）
    if sampling is not None:
        body["temperature"] = sampling.temperature
        body["top_p"] = sampling.top_p

    if json_object:
        body["response_format"] = {"type": "json_object"}

    if tools:
        # 只下发 tools，**不下发 tool_choice**（见模块文档：思考模式下 required/指定工具会 400）
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


def _first_delta(chunk: Mapping[str, Any]) -> Mapping[str, Any] | None:
    choices = chunk.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, str) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    delta = first.get("delta")
    return delta if isinstance(delta, Mapping) else None


def extract_tool_call_deltas(chunk: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """取出一个数据块里的 `delta.tool_calls` 增量项（可能为空）。"""
    delta = _first_delta(chunk)
    if delta is None:
        return []
    raw = delta.get("tool_calls")
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


class ToolCallAccumulator:
    """按 `index` 累积流式工具调用。

    官方把一次调用拆成若干块：首块带 `id` 与 `name`，后续块只带 `arguments` 片段。
    因此 `id` / `name` **只取首个非空值**（后续块可能是空串或重复值），`arguments` 逐块拼接。
    """

    def __init__(self) -> None:
        self._parts: dict[int, dict[str, str]] = {}

    def feed(self, chunk: Mapping[str, Any]) -> bool:
        """累积一个数据块；返回该块是否含有工具调用增量。"""
        deltas = extract_tool_call_deltas(chunk)
        for position, raw in enumerate(deltas):
            index = raw.get("index")
            key = index if isinstance(index, int) else position
            entry = self._parts.setdefault(key, {"id": "", "name": "", "arguments": ""})

            call_id = raw.get("id")
            if isinstance(call_id, str) and call_id and not entry["id"]:
                entry["id"] = call_id

            function = raw.get("function")
            if isinstance(function, Mapping):
                name = function.get("name")
                if isinstance(name, str) and name and not entry["name"]:
                    entry["name"] = name
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    entry["arguments"] += arguments
        return bool(deltas)

    def finish(self) -> tuple[ToolCall, ...]:
        """产出累积结果；缺 `name` 的块无法执行，直接丢弃。"""
        calls: list[ToolCall] = []
        for key in sorted(self._parts):
            entry = self._parts[key]
            if not entry["name"]:
                continue
            calls.append(ToolCall(id=entry["id"], name=entry["name"], arguments=entry["arguments"]))
        return tuple(calls)


def parse_tool_calls(message: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    """解析非流式响应里的 `message.tool_calls`。"""
    raw = message.get("tool_calls")
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return ()

    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments")
        call_id = item.get("id")
        calls.append(
            ToolCall(
                id=call_id if isinstance(call_id, str) else "",
                name=name,
                arguments=arguments if isinstance(arguments, str) else "",
            )
        )
    return tuple(calls)


@dataclass(frozen=True)
class BlockingResult:
    """非流式响应。"""

    content: str
    reasoning: str | None
    usage: TokenUsage | None
    tool_calls: tuple[ToolCall, ...] = ()


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
    return BlockingResult(
        content=content,
        reasoning=reasoning,
        usage=usage,
        tool_calls=parse_tool_calls(message),
    )


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
