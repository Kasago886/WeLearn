"""工具调用注册表（T4.6，FR-714）。

三个工具让模型能主动读写学习档案：读档案、写笔记、记薄弱点。

**官方限制决定了这里的形状**：

1. 思考模式下不支持 `tool_choice = "required"`，也不支持指定具体某个工具（服务端返回 400），
   因此请求体里**从不携带 `tool_choice`**（见 :func:`app.infra.llm.protocol.build_chat_request`）。
   代价是模型只能"自愿"调用工具，所以每个工具的 ``description`` 必须把
   "什么时候该用"说清楚，否则模型不会去用；
2. 请求携带 `tools` 时，历史轮次的 `reasoning_content` 必须回传并会拼进上下文——
   由 :mod:`app.services.llm.context_policy` 的 ``tools_enabled`` 开关负责，
   执行器在持有工具时会强制打开它（见 :class:`app.services.llm.agent.RoundExecutor`）。

**错误处理原则**：工具错误是给**模型**看的信息，不是给用户的异常。
参数 JSON 非法、缺参数、节点不存在、甚至 handler 抛异常，都转成一段可读文本回给模型，
让它自己纠正；只有"模型调用本身失败"才让整轮失败。否则模型偶发一次参数笔误，
用户就会看到整轮报错——而这本来是可以自愈的。
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.contracts.schemas import WeakPointReason
from app.infra.llm.protocol import ToolCall
from app.services.learning.entities import KnowledgePoint
from app.services.learning.repository import DomainRepositories

READ_ARCHIVE = "read_archive"
WRITE_NOTE = "write_note"
RECORD_WEAK_POINT = "record_weak_point"

ARCHIVE_TEXT_LIMIT = 4000
"""档案文本上限：工具结果会原样进入下一轮上下文，无上限会让上下文被档案挤爆。"""

NOTE_PREVIEW_LIMIT = 300
NOTE_CONTENT_LIMIT = 4000

_WEAK_POINT_REASONS: dict[str, WeakPointReason] = {reason.value: reason for reason in WeakPointReason}

ToolHandler = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的定义：给模型看的描述 + 本地真正的处理函数。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def to_payload(self) -> dict[str, Any]:
        """转成请求体 ``tools[]`` 的元素（OpenAI 兼容的 function 形式）。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """按名字索引的工具集合。

    保持**注册顺序**不变：``tools`` 数组每轮都一样，才不会破坏缓存前缀（FR-720）。
    """

    def __init__(self, specs: Sequence[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self._specs[spec.name] = spec

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def specs(self) -> list[dict[str, Any]]:
        """请求体的 ``tools`` 数组。"""
        return [spec.to_payload() for spec in self._specs.values()]

    def execute(self, call: ToolCall) -> str:
        """执行一次工具调用；**任何错误都返回文本**，绝不抛出。"""
        spec = self._specs.get(call.name)
        if spec is None:
            available = ", ".join(self._specs) or "（无）"
            return _invalid(f"不存在名为 {call.name!r} 的工具。可用工具：{available}。")

        try:
            arguments = _parse_arguments(call.arguments)
        except ValueError as exc:
            return _invalid(f"工具 {spec.name} 的参数不是合法的 JSON 对象：{exc}。请重新给出符合 schema 的 JSON。")

        missing = _missing_required(spec.parameters, arguments)
        if missing:
            return _invalid(f"工具 {spec.name} 缺少必填参数：{', '.join(missing)}。请补齐后重试。")

        try:
            result = spec.handler(arguments)
        except Exception as exc:  # 工具内部的任何异常都不能让整轮失败
            return _invalid(f"工具 {spec.name} 执行失败：{type(exc).__name__}: {exc}。请检查参数后重试。")

        return result if isinstance(result, str) else str(result)


def build_teaching_tools(repos: DomainRepositories, subject_id: str) -> ToolRegistry:
    """教学场景的工具集。

    作用域绑死当前主题：模型不需要（也不应该）传 ``subject_id``——
    一旦让它自己填，就可能写到别的主题里去，而用户看到的是"我的笔记不见了"。
    """

    def read_archive(_: dict[str, Any]) -> str:
        return _read_archive(repos, subject_id)

    def write_note(arguments: dict[str, Any]) -> str:
        return _write_note(repos, subject_id, arguments)

    def record_weak_point(arguments: dict[str, Any]) -> str:
        return _record_weak_point(repos, subject_id, arguments)

    return ToolRegistry(
        [
            ToolSpec(
                name=READ_ARCHIVE,
                description=(
                    "读取当前学习主题的档案：路径里每个节点的序号、标题、状态、目标，以及已有笔记摘要。"
                    "当用户问“我学到哪了”“我有哪些笔记”“这个主题还有哪些节点”时使用。"
                ),
                parameters={"type": "object", "properties": {}, "additionalProperties": False},
                handler=read_archive,
            ),
            ToolSpec(
                name=WRITE_NOTE,
                description=(
                    "把一个节点的笔记写入学习档案（覆盖该节点原有笔记，来源标记为用户）。"
                    "当用户说“帮我记一下”“记到笔记里”并给出明确内容时使用；"
                    "不要替用户总结他没说过的内容。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "节点 id，来自 read_archive 的 node_id"},
                        "content": {"type": "string", "description": f"笔记正文，不超过 {NOTE_CONTENT_LIMIT} 字"},
                    },
                    "required": ["node_id", "content"],
                    "additionalProperties": False,
                },
                handler=write_note,
            ),
            ToolSpec(
                name=RECORD_WEAK_POINT,
                description=(
                    "把一个节点标记为薄弱点，供之后的复习安排使用。"
                    "当用户明确表示自己没弄懂、跳过或直接看了答案时使用。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "节点 id，来自 read_archive 的 node_id"},
                        "reason": {
                            "type": "string",
                            "enum": sorted(_WEAK_POINT_REASONS),
                            "description": (
                                "成因：partial=部分掌握, skipped=跳过, "
                                "show_answer=直接看答案, advanced=强行推进"
                            ),
                        },
                    },
                    "required": ["node_id", "reason"],
                    "additionalProperties": False,
                },
                handler=record_weak_point,
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #


def _read_archive(repos: DomainRepositories, subject_id: str) -> str:
    subject = repos.subjects.get(subject_id)
    path = repos.paths.get_active(subject_id)

    lines = [f"学习档案（主题：{subject.title if subject is not None else subject_id}）"]
    if path is None:
        lines.append("（还没有生成学习路径）")
    else:
        lines.append(f"路径 v{path.version}，共 {path.node_count} 个节点：")
        for node in path.ordered_nodes():
            lines.extend(_describe_node(node))

    notes = repos.notes.list_for_subject(subject_id)
    lines.append("")
    if notes:
        lines.append(f"笔记摘要（{len(notes)} 条）：")
        for note in notes:
            target = "主题级" if note.node_id is None else f"节点 {note.node_id}"
            lines.append(f"- [{target}] {_truncate(note.content, NOTE_PREVIEW_LIMIT)}")
    else:
        lines.append("笔记摘要：（暂无笔记）")

    return _truncate_with_notice("\n".join(lines), ARCHIVE_TEXT_LIMIT)


def _describe_node(node: KnowledgePoint) -> list[str]:
    state = node.state.value if hasattr(node.state, "value") else str(node.state)
    return [
        f"{node.order_index + 1}. [{state}] {node.title}（node_id={node.id}）",
        f"   目标：{_oneline(node.objective)}",
    ]


def _write_note(repos: DomainRepositories, subject_id: str, arguments: dict[str, Any]) -> str:
    node_id, error = _require_node(repos, arguments)
    if error is not None:
        return error
    assert node_id is not None

    raw_content = arguments.get("content")
    if not isinstance(raw_content, str) or not raw_content.strip():
        return _invalid("参数 content 必须是非空字符串。")

    content = raw_content.strip()
    if len(content) > NOTE_CONTENT_LIMIT:
        return _invalid(f"笔记内容 {len(content)} 字，超过 {NOTE_CONTENT_LIMIT} 字上限，请精简后再写入。")

    note = repos.notes.upsert(subject_id, node_id, content, source="user")
    node = repos.paths.node(node_id)
    title = node.title if node is not None else node_id
    return f"已写入笔记：节点「{title}」（{len(note.content)} 字，来源 user）。"


def _record_weak_point(repos: DomainRepositories, subject_id: str, arguments: dict[str, Any]) -> str:
    node_id, error = _require_node(repos, arguments)
    if error is not None:
        return error
    assert node_id is not None

    raw_reason = arguments.get("reason")
    if not isinstance(raw_reason, str) or raw_reason not in _WEAK_POINT_REASONS:
        allowed = ", ".join(sorted(_WEAK_POINT_REASONS))
        return _invalid(f"参数 reason 必须是 {allowed} 之一（收到 {raw_reason!r}）。")

    weak_point = repos.weak_points.record(subject_id, node_id, _WEAK_POINT_REASONS[raw_reason])
    node = repos.paths.node(node_id)
    title = node.title if node is not None else node_id
    return f"已记录薄弱点：节点「{title}」，成因 {weak_point.reason.value}，累计 {weak_point.fail_count} 次。"


def _require_node(repos: DomainRepositories, arguments: dict[str, Any]) -> tuple[str | None, str | None]:
    """校验 ``node_id``：必须是非空字符串且确实存在于当前主题的路径里。"""
    raw = arguments.get("node_id")
    if not isinstance(raw, str) or not raw.strip():
        return None, _invalid("缺少参数 node_id（字符串）。请先用 read_archive 查看节点的 node_id。")

    node_id = raw.strip()
    if repos.paths.node(node_id) is None:
        return None, _invalid(f"节点 {node_id} 不存在。请先用 read_archive 查看可用的 node_id。")
    return node_id, None


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def _invalid(message: str) -> str:
    return f"错误：{message}"


def _parse_arguments(raw: str) -> dict[str, Any]:
    """把模型给的原始 JSON 字符串解析成参数对象。

    ``""`` 视为"无参数"：部分端点对空参数返回空串而不是 ``{}``。
    """
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{exc.msg}（位置 {exc.pos}）") from exc
    if not isinstance(parsed, dict):
        raise ValueError("参数必须是 JSON 对象")
    return parsed


def _missing_required(parameters: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    required = parameters.get("required")
    if not isinstance(required, Sequence) or isinstance(required, str):
        return []
    return [str(name) for name in required if name not in arguments]


def _oneline(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, limit: int) -> str:
    collapsed = _oneline(text)
    return collapsed if len(collapsed) <= limit else f"{collapsed[:limit]}…"


def _truncate_with_notice(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n……（档案内容过长，已截断，仅显示前 {limit} 字）"
