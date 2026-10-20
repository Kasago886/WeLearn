"""工具调用（T4.6，FR-714）测试：协议解析、注册表容错、执行循环与开关行为。

三组不变量：

- **协议层**：流式 tool_calls 按 `index` 累积（`id`/`name` 取首块、`arguments` 拼接），
  非流式同构；请求体**永不含 `tool_choice`**（思考模式下会 400）；
- **注册表**：参数 JSON 非法、缺参数、handler 抛异常都变成"给模型看的结果文本"，
  绝不让整轮失败；
- **执行循环**：工具真正落地（笔记写进库）、中间轮正文不外泄、轮次有上限，
  且 `tools=None` 时行为与改造前完全一致。
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from app.contracts.schemas import WeakPointReason
from app.infra.deepseek.protocol import (
    BlockingResult,
    StreamEvent,
    ToolCall,
    ToolCallAccumulator,
    build_chat_request,
    parse_blocking_response,
)
from app.infra.settings import AppSettings
from app.main_context import AppContext
from app.services.learning.session import ConversationSessionFactory
from app.services.llm.agent import MAX_TOOL_ROUNDS, RoundExecutor
from app.services.llm.memory import Memory, system_message
from app.services.llm.tools import (
    ARCHIVE_TEXT_LIMIT,
    READ_ARCHIVE,
    RECORD_WEAK_POINT,
    WRITE_NOTE,
    ToolRegistry,
    ToolSpec,
    build_teaching_tools,
)
from app.services.llm.usage import UsageRecorder
from tests.conftest import ClientFactory
from tests.learning_harness import LearningHarness, make_harness
from tests.stubs import FakeLlmClient, ScriptedResponse, text_events
from tests.test_teaching_service import enter_teaching, node_payload

NOTE_CONTENT = "贝叶斯：先验 × 似然 → 后验。"


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


class ToolAwareClient(FakeLlmClient):
    """记录 ``tools`` 参数的桩。

    真实客户端会把它写进请求体的 ``tools`` 字段；桩只需记下来，
    测试据此判断"有没有下发工具"以及"最后一轮是否收走了工具"。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tools_seen: list[list[dict[str, Any]] | None] = []

    async def stream_chat(
        self, *, tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> AsyncIterator[StreamEvent]:
        self.tools_seen.append(tools)
        async for event in super().stream_chat(**kwargs):
            yield event

    async def chat(self, *, tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> BlockingResult:
        self.tools_seen.append(tools)
        return await super().chat(**kwargs)


class BlockingToolClient(FakeLlmClient):
    """非流式桩：按脚本返回 ``tool_calls``。

    ``FakeLlmClient.chat`` 只回正文，因此这里单独覆写；脚本项是 ``(正文, 工具调用)``。
    """

    def __init__(self, script: list[tuple[str, tuple[ToolCall, ...]]]) -> None:
        super().__init__()
        self._script = list(script)
        self.tools_seen: list[list[dict[str, Any]] | None] = []

    async def chat(self, *, tools: list[dict[str, Any]] | None = None, **_: Any) -> BlockingResult:
        self.tools_seen.append(tools)
        content, calls = self._script.pop(0)
        return BlockingResult(content=content, reasoning=None, usage=None, tool_calls=calls)


def tool_events(*calls: ToolCall, content: str | None = None) -> list[StreamEvent]:
    """一轮"只调工具"的响应（可带一段不该被展示的过渡正文）。"""
    events: list[StreamEvent] = []
    if content:
        events.append(StreamEvent(content=content))
    events.append(StreamEvent(tool_calls=tuple(calls)))
    return events


def prepare(context: AppContext) -> tuple[LearningHarness, str, str]:
    """建一个主题 + 两个节点的路径，返回 (脚手架, 主题 id, 第一个节点 id)。"""
    harness = make_harness(context)
    subject_id = harness.new_subject("贝叶斯定理")
    path = harness.repos.paths.create(
        subject_id,
        rationale="从基础开始逐点推进。",
        model="deepseek-flash",
        nodes=[node_payload("节点 0", 1), node_payload("节点 1", 1)],
    )
    return harness, subject_id, path.nodes[0].id


def make_executor(
    harness: LearningHarness,
    client: FakeLlmClient,
    subject_id: str,
    *,
    with_tools: bool = True,
) -> RoundExecutor:
    memory = Memory()
    memory.add_round().add(system_message("你是耐心的导师。"))
    registry = build_teaching_tools(harness.repos, subject_id) if with_tools else None
    return RoundExecutor(client, memory, harness.policy, tools=registry)


def tool_messages(executor: RoundExecutor) -> list[dict[str, Any]]:
    return [message for message in executor.memory.outbound_messages() if message.get("role") == "tool"]


# --------------------------------------------------------------------------- #
# 协议层
# --------------------------------------------------------------------------- #


def tool_chunk(*calls: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"tool_calls": list(calls)}}]}


def test_accumulator_merges_fragments_by_index() -> None:
    """官方把一次调用拆成多块：id/name 只在首块，arguments 逐块拼接。"""
    accumulator = ToolCallAccumulator()
    assert accumulator.feed(
        tool_chunk(
            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "write_note", "arguments": ""}},
            {"index": 1, "id": "call_2", "type": "function", "function": {"name": "read_archive", "arguments": ""}},
        )
    )
    # 后续块只有 arguments 片段，且 id/name 可能是空串
    accumulator.feed(tool_chunk({"index": 0, "id": "", "function": {"name": "", "arguments": '{"node_id": "n1",'}}))
    accumulator.feed(tool_chunk({"index": 1, "function": {"arguments": "{}"}}))
    # 末块补齐第一个调用的参数
    accumulator.feed(tool_chunk({"index": 0, "function": {"arguments": ' "content": "记一下"}'}}))
    assert accumulator.feed({"choices": [{"delta": {"content": "正文"}}]}) is False

    assert accumulator.finish() == (
        ToolCall(id="call_1", name="write_note", arguments='{"node_id": "n1", "content": "记一下"}'),
        ToolCall(id="call_2", name="read_archive", arguments="{}"),
    )


def test_accumulator_drops_calls_without_name() -> None:
    accumulator = ToolCallAccumulator()
    accumulator.feed(tool_chunk({"index": 0, "id": "call_1", "function": {"arguments": "{}"}}))
    assert accumulator.finish() == ()


def test_blocking_response_parses_tool_calls() -> None:
    body = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "type": "function",
                            "function": {"name": "read_archive", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    result = parse_blocking_response(body)
    assert result.tool_calls == (ToolCall(id="call_9", name="read_archive", arguments="{}"),)
    assert result.content == ""


def build_tools_body_sample() -> list[dict[str, Any]]:
    """不依赖数据库的 tools 数组样例（注册表本身在别的用例里覆盖）。"""
    registry = ToolRegistry(
        [
            ToolSpec(
                name=READ_ARCHIVE,
                description="读取档案",
                parameters={"type": "object", "properties": {}},
                handler=lambda _: "",
            )
        ]
    )
    return registry.specs()


@pytest.mark.parametrize("thinking", [True, False])
def test_request_body_never_sends_tool_choice(thinking: bool) -> None:
    """官方限制：思考模式下 required / 指定工具都会 400，所以从来不下发 tool_choice。"""
    tools = build_tools_body_sample()
    body = build_chat_request(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        thinking=thinking,
        reasoning_effort="high",
        max_tokens=512,
        tools=tools,
    )
    assert body["tools"] == tools
    assert "tool_choice" not in body


def test_request_body_omits_tools_when_none() -> None:
    body = build_chat_request(
        model="deepseek-flash",
        messages=[],
        stream=False,
        thinking=False,
        reasoning_effort="high",
        max_tokens=128,
        tools=None,
    )
    assert "tools" not in body
    assert "tool_choice" not in body


async def test_client_streams_tool_calls_and_sends_tools(make_client: ClientFactory) -> None:
    """端到端（协议 + 客户端）：SSE 分块累积成一次调用，请求体带 tools 但不带 tool_choice。"""
    captured: list[dict[str, Any]] = []
    chunks = [
        tool_chunk(
            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "write_note", "arguments": ""}}
        ),
        tool_chunk({"index": 0, "function": {"arguments": '{"node_id": "n1", '}}),
        tool_chunk({"index": 0, "function": {"arguments": '"content": "记一下"}'}}),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    ]
    body = "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=body.encode("utf-8"))

    client = make_client(handler)
    tools = build_tools_body_sample()
    events = [
        event
        async for event in client.stream_chat(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "记一下"}],
            thinking=True,
            tools=tools,
        )
    ]

    with_calls = [event for event in events if event.tool_calls]
    assert len(with_calls) == 1
    assert with_calls[0].tool_calls == (
        ToolCall(id="call_1", name="write_note", arguments='{"node_id": "n1", "content": "记一下"}'),
    )
    # 只有 tool_calls、没有正文，不能被当成"空回复"而重试
    assert len(captured) == 1
    assert captured[0]["tools"] == tools
    assert "tool_choice" not in captured[0]
    assert events[-1].usage is not None


# --------------------------------------------------------------------------- #
# 注册表容错
# --------------------------------------------------------------------------- #


def test_registry_success_writes_note(context: AppContext) -> None:
    harness, subject_id, node_id = prepare(context)
    registry = build_teaching_tools(harness.repos, subject_id)

    text = registry.execute(
        ToolCall(id="call_1", name=WRITE_NOTE, arguments=json.dumps({"node_id": node_id, "content": NOTE_CONTENT}))
    )

    assert "已写入笔记" in text
    note = harness.repos.notes.get(subject_id, node_id)
    assert note is not None
    assert note.content == NOTE_CONTENT
    assert note.source == "user"


def test_registry_invalid_json_returns_text() -> None:
    registry = ToolRegistry(
        [ToolSpec(name="echo", description="", parameters={"type": "object"}, handler=lambda args: str(args))]
    )
    text = registry.execute(ToolCall(id="c1", name="echo", arguments="{不是 JSON"))
    assert text.startswith("错误：")
    assert "JSON" in text


def test_registry_missing_parameter_returns_text() -> None:
    called: list[dict[str, Any]] = []

    def handler(arguments: dict[str, Any]) -> str:
        called.append(arguments)
        return "不该被调用"

    registry = ToolRegistry(
        [
            ToolSpec(
                name="needs_node",
                description="",
                parameters={"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]},
                handler=handler,
            )
        ]
    )
    text = registry.execute(ToolCall(id="c1", name="needs_node", arguments="{}"))
    assert "缺少必填参数" in text
    assert called == []


def test_registry_handler_exception_returns_text() -> None:
    def boom(_: dict[str, Any]) -> str:
        raise RuntimeError("数据库炸了")

    registry = ToolRegistry([ToolSpec(name="boom", description="", parameters={"type": "object"}, handler=boom)])
    text = registry.execute(ToolCall(id="c1", name="boom", arguments="{}"))
    assert text.startswith("错误：")
    assert "数据库炸了" in text


def test_registry_unknown_tool_returns_text() -> None:
    registry = ToolRegistry([ToolSpec(name="echo", description="", parameters={}, handler=lambda _: "")])
    text = registry.execute(ToolCall(id="c1", name="nope", arguments="{}"))
    assert "不存在名为" in text


def test_read_archive_lists_nodes_and_notes(context: AppContext) -> None:
    harness, subject_id, node_id = prepare(context)
    harness.repos.notes.upsert(subject_id, node_id, "旧笔记内容", source="user")
    registry = build_teaching_tools(harness.repos, subject_id)

    text = registry.execute(ToolCall(id="c1", name=READ_ARCHIVE, arguments="{}"))

    assert "贝叶斯定理" in text
    assert "节点 0" in text
    assert node_id in text
    assert "旧笔记内容" in text
    assert "笔记摘要" in text


def test_read_archive_truncates_long_output(context: AppContext) -> None:
    """档案文本有上限：工具结果会进上下文，无上限会把窗口挤爆。"""
    harness = make_harness(context)
    subject_id = harness.new_subject("长档案")
    harness.repos.paths.create(
        subject_id,
        rationale="节点很多。",
        model=None,
        nodes=[node_payload(f"节点 {index}", 1) for index in range(60)],
    )
    registry = build_teaching_tools(harness.repos, subject_id)

    text = registry.execute(ToolCall(id="c1", name=READ_ARCHIVE, arguments="{}"))

    assert "已截断" in text
    assert len(text) > ARCHIVE_TEXT_LIMIT
    assert len(text) < ARCHIVE_TEXT_LIMIT + 200


def test_record_weak_point_validates_reason(context: AppContext) -> None:
    harness, subject_id, node_id = prepare(context)
    registry = build_teaching_tools(harness.repos, subject_id)

    ok = registry.execute(
        ToolCall(
            id="c1",
            name=RECORD_WEAK_POINT,
            arguments=json.dumps({"node_id": node_id, "reason": WeakPointReason.SKIPPED.value}),
        )
    )
    bad = registry.execute(
        ToolCall(id="c2", name=RECORD_WEAK_POINT, arguments=json.dumps({"node_id": node_id, "reason": "瞎写"}))
    )

    assert "已记录薄弱点" in ok
    assert bad.startswith("错误：")
    assert WeakPointReason.SKIPPED.value in bad  # 错误文本里列出可选值，模型才能自我纠正
    open_points = harness.repos.weak_points.list_open(subject_id)
    assert [point.reason for point in open_points] == [WeakPointReason.SKIPPED]


def test_write_note_rejects_unknown_node(context: AppContext) -> None:
    harness, subject_id, _ = prepare(context)
    registry = build_teaching_tools(harness.repos, subject_id)

    text = registry.execute(
        ToolCall(id="c1", name=WRITE_NOTE, arguments=json.dumps({"node_id": "不存在的节点", "content": "x"}))
    )

    assert text.startswith("错误：")
    assert "不存在" in text
    assert harness.repos.notes.list_for_subject(subject_id) == []


# --------------------------------------------------------------------------- #
# 执行循环
# --------------------------------------------------------------------------- #


async def test_stream_executes_tool_then_answers(context: AppContext) -> None:
    harness, subject_id, node_id = prepare(context)
    arguments = json.dumps({"node_id": node_id, "content": NOTE_CONTENT}, ensure_ascii=False)
    client = ToolAwareClient(
        stream_script=[
            ScriptedResponse(
                events=tool_events(
                    ToolCall(id="call_1", name=WRITE_NOTE, arguments=arguments),
                    content="我先查一下档案。",
                )
            ),
            ScriptedResponse(events=text_events("已记到笔记里了。")),
        ]
    )
    executor = make_executor(harness, client, subject_id)
    deltas: list[str] = []

    result = await executor.stream(
        "帮我把结论记到笔记里", on_delta=lambda event: deltas.append(event.content or "")
    )

    assert result.ok
    assert result.content == "已记到笔记里了。"
    # 工具真的落地了
    note = harness.repos.notes.get(subject_id, node_id)
    assert note is not None and note.content == NOTE_CONTENT and note.source == "user"
    # 对话里留下 tool 结果，供下一轮模型回读
    messages = tool_messages(executor)
    assert len(messages) == 1
    assert messages[0]["tool_call_id"] == "call_1"
    assert "已写入笔记" in str(messages[0]["content"])
    # 中间轮的过渡正文不展示、也不进入最终结果
    assert "我先查一下档案。" not in result.content
    assert "我先查一下档案。" not in "".join(deltas)
    assert "已记到笔记里了。" in "".join(deltas)


async def test_stream_keeps_tool_errors_inside_the_round(context: AppContext) -> None:
    """工具参数非法只是"模型需要纠正的问题"，整轮不应失败。"""
    harness, subject_id, _ = prepare(context)
    client = ToolAwareClient(
        stream_script=[
            ScriptedResponse(
                events=[StreamEvent(tool_calls=(ToolCall(id="call_bad", name=WRITE_NOTE, arguments="{不是 JSON"),))]
            ),
            ScriptedResponse(events=text_events("抱歉，我重新记一次。")),
        ]
    )
    executor = make_executor(harness, client, subject_id)

    result = await executor.stream("记一下")

    assert result.ok
    assert result.content == "抱歉，我重新记一次。"
    messages = tool_messages(executor)
    assert len(messages) == 1
    assert messages[0]["tool_call_id"] == "call_bad"
    assert "错误" in str(messages[0]["content"])
    assert harness.repos.notes.list_for_subject(subject_id) == []


async def test_stream_stops_tool_loop_at_limit(context: AppContext) -> None:
    """模型反复要工具时必须有上限，且最后一轮不带工具逼它收尾。"""
    harness, subject_id, _ = prepare(context)
    looping = [
        ScriptedResponse(
            events=[StreamEvent(tool_calls=(ToolCall(id=f"call_{index}", name=READ_ARCHIVE, arguments="{}"),))]
        )
        for index in range(MAX_TOOL_ROUNDS)
    ]
    client = ToolAwareClient(
        stream_script=[*looping, ScriptedResponse(events=text_events("档案里一共两个节点。"))]
    )
    executor = make_executor(harness, client, subject_id)

    result = await executor.stream("我学到哪了")

    assert result.ok
    assert client.stream_calls == MAX_TOOL_ROUNDS + 1
    assert client.tools_seen[-1] is None  # 上限后用"无工具"请求收尾
    assert all(seen is not None for seen in client.tools_seen[:-1])


async def test_stream_without_tools_keeps_incremental_deltas(context: AppContext) -> None:
    """``tools=None`` 必须与改造前完全一致：逐块实时推送，不携带 tools。"""
    harness, subject_id, _ = prepare(context)
    client = ToolAwareClient(stream_script=[ScriptedResponse(events=text_events("第一块", "第二块"))])
    executor = make_executor(harness, client, subject_id, with_tools=False)
    deltas: list[StreamEvent] = []

    result = await executor.stream("讲一下", on_delta=deltas.append)

    assert result.ok
    assert [event.content for event in deltas if event.content] == ["第一块", "第二块"]
    assert client.tools_seen[0] is None
    assert executor.tools_enabled is False


async def test_run_executes_tools_too(context: AppContext) -> None:
    """非流式路径同样支持工具（判定/压缩以外的调用也可能带上它）。"""
    harness, subject_id, node_id = prepare(context)
    arguments = json.dumps({"node_id": node_id, "content": NOTE_CONTENT}, ensure_ascii=False)
    client = BlockingToolClient(
        [
            ("", (ToolCall(id="call_1", name=WRITE_NOTE, arguments=arguments),)),
            ("好的，已记下。", ()),
        ]
    )
    executor = make_executor(harness, client, subject_id)

    result = await executor.run("记一下")

    assert result.ok
    assert result.content == "好的，已记下。"
    assert harness.repos.notes.get(subject_id, node_id) is not None


# --------------------------------------------------------------------------- #
# 教学接线：开关决定是否挂工具
# --------------------------------------------------------------------------- #


def wire_harness(context: AppContext, *, tools_enabled: bool) -> tuple[LearningHarness, ToolAwareClient]:
    """把脚手架的客户端换成能记录 tools 的桩（会话工厂也要一起换）。"""
    harness = make_harness(context)
    client = ToolAwareClient()
    harness.client = client
    harness.sessions = ConversationSessionFactory(
        client,
        context.memories,
        UsageRecorder(context.db),
        harness.policy,
    )
    harness.settings = AppSettings(tools_enabled=tools_enabled)
    return harness, client


async def test_teaching_attaches_tools_and_replays_reasoning(context: AppContext) -> None:
    """开关打开时：请求带 tools，且历史 reasoning_content 必须回传（官方硬性要求）。"""
    harness, client = wire_harness(context, tools_enabled=True)
    _, _, teaching, subject_id = enter_teaching(harness)

    client.stream_script = [ScriptedResponse(events=text_events("讲解正文。", reasoning="先想清楚再讲。"))]
    first = await teaching.start(subject_id)
    assert first.ok

    client.stream_script = [ScriptedResponse(events=text_events("再补一句。"))]
    client.stream_calls = 0
    await teaching.hint(subject_id)

    first_tools = client.tools_seen[0]
    assert first_tools is not None
    assert {spec["function"]["name"] for spec in first_tools} == {READ_ARCHIVE, WRITE_NOTE, RECORD_WEAK_POINT}
    # 工具开启 → 历史思维链回传（context_policy 的 tools_enabled 生效）
    assert any(message.get("reasoning_content") == "先想清楚再讲。" for message in client.calls[-1].messages)


async def test_teaching_without_flag_sends_no_tools(context: AppContext) -> None:
    harness, client = wire_harness(context, tools_enabled=False)
    _, _, teaching, subject_id = enter_teaching(harness)

    client.stream_script = [ScriptedResponse(events=text_events("讲解正文。", reasoning="先想清楚再讲。"))]
    await teaching.start(subject_id)
    client.stream_script = [ScriptedResponse(events=text_events("再补一句。"))]
    client.stream_calls = 0
    await teaching.hint(subject_id)

    assert client.tools_seen[0] is None
    assert all("reasoning_content" not in message for message in client.calls[-1].messages)
