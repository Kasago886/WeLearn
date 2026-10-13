"""后端闭环集成测试（p2-backend-tests）。

这是 P2A 的出口条件：**用桩 LLM 驱动 HTTP 接口跑完一个主题的完整闭环**——

    创建主题 → 测评（提问/作答）→ 画像 → 生成路径 → 确认 → 教学（讲解/提问/判定）→
    未通过重讲 → 重试耗尽三选一 → 推进 → 完成

全部通过 ``TestClient`` + 注入的桩客户端完成，不消耗真实额度。
SSE 响应按帧解析后逐事件断言，确保"前端能拿到什么"是被真正验证过的。
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from app.contracts.schemas import NodeDecision
from app.infra.settings import AppSettings
from app.main import create_app
from tests.stubs import FakeLlmClient, ScriptedResponse, json_response, text_events

TOPICS = ["定义与直觉", "关键公式"]
TOTAL_TURNS = len(TOPICS)

TOPICS_JSON = json.dumps(
    {"topics": TOPICS, "first_question": "先用你自己的话说说它解决什么问题？"},
    ensure_ascii=False,
)


def path_nodes(count: int) -> list[dict]:
    return [
        {
            "title": f"节点 {index}",
            "objective": f"能解释 节点 {index}",
            "prerequisites": [],
            "key_points": ["要点一", "要点二", "要点三"],
            "checkpoints": [
                {
                    "question": f"节点 {index} 的检查点？",
                    "expected_points": ["判定要点"],
                    "reference_answer": "参考答案",
                }
            ],
            "is_quick_review": False,
        }
        for index in range(count)
    ]


def profile_json(level: str = "none") -> str:
    return json.dumps(
        {
            "levels": [{"topic": TOPICS[0], "level": level, "evidence": "证据"}],
            "summary": "需要从基础开始。",
            "suggested_start_node": "从定义开始。",
        },
        ensure_ascii=False,
    )


def path_json(count: int = 3) -> str:
    return json.dumps({"rationale": "依据画像从基础推进。", "nodes": path_nodes(count)}, ensure_ascii=False)


def node_json(title: str, *, checkpoints: int = 1) -> dict:
    """单节点重生成时模型应返回的结构（与 planner_node 提示词的样例一致）。"""
    return {
        "title": title,
        "objective": f"能解释 {title}",
        "prerequisites": [],
        "key_points": ["要点一", "要点二", "要点三"],
        "checkpoints": [
            {
                "question": f"{title} 的检查点 {index}？",
                "expected_points": ["判定要点"],
                "reference_answer": "参考答案",
            }
            for index in range(checkpoints)
        ],
        "is_quick_review": False,
    }


def verdict_json(verdict: str) -> str:
    if verdict == "pass":
        payload = {
            "verdict": "pass",
            "confidence": 0.9,
            "feedback": "答对了。",
            "hit_points": ["判定要点"],
            "missing_points": [],
        }
    else:
        payload = {
            "verdict": verdict,
            "confidence": 0.3,
            "feedback": "还差一点。",
            "hit_points": [],
            "missing_points": ["判定要点"],
        }
    return json.dumps(payload, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# SSE 解析与夹具
# --------------------------------------------------------------------------- #


@dataclass
class SseEvent:
    type: str
    data: dict


def parse_sse(body: str) -> list[SseEvent]:
    """把 SSE 响应体解析为事件列表。"""
    events: list[SseEvent] = []
    for block in body.split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        event_type = ""
        payload: dict = {}
        for line in lines:
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[len("data:") :].strip())
        if event_type:
            events.append(SseEvent(type=event_type, data=payload))
    return events


def token_text(events: list[SseEvent]) -> str:
    return "".join(str(event.data.get("text", "")) for event in events if event.type == "token")


def events_of(events: list[SseEvent], event_type: str) -> list[SseEvent]:
    return [event for event in events if event.type == event_type]


@dataclass
class ApiHarness:
    client: TestClient
    stub: FakeLlmClient
    app: object


@pytest.fixture
def api(env: dict[str, str]) -> Iterator[ApiHarness]:
    """构造带桩 LLM 的应用；脚本由测试按阶段注入。"""
    app = create_app(env)
    stub = FakeLlmClient()
    app.state.ctx.llm = stub  # type: ignore[assignment]  # 关键：接口层转而使用桩
    app.state.stack = _rebuild_stack(app)
    with TestClient(app) as client:
        yield ApiHarness(client=client, stub=stub, app=app)


def _rebuild_stack(app: object) -> object:
    """用被替换过的 ctx.llm 重新装配领域服务栈。"""
    from app.services.learning.stack import build_learning_stack

    return build_learning_stack(app.state.ctx)  # type: ignore[attr-defined]


def script(harness: ApiHarness, blocking: list[str] | None = None, stream: list[str] | None = None) -> None:
    """设置当前阶段的脚本（会重置调用计数，避免阶段之间相互干扰）。"""
    harness.stub.blocking_script = [json_response(item) for item in (blocking or [])]
    harness.stub.stream_script = [ScriptedResponse(events=text_events(item)) for item in (stream or [])]
    harness.stub.blocking_calls = 0
    harness.stub.stream_calls = 0


def create_subject(harness: ApiHarness, title: str = "贝叶斯定理", **extra: str) -> str:
    response = harness.client.post("/api/subjects", json={"title": title, **extra})
    assert response.status_code == 201, response.text
    return response.json()["subject"]["id"]


def run_assessment(harness: ApiHarness, subject_id: str, *, level: str = "none") -> None:
    """跑完测评（2 个子领域 → 2 次作答）。"""
    script(harness, blocking=[TOPICS_JSON, profile_json(level)], stream=["第二个方面的问题？"])
    assert harness.client.post(f"/api/subjects/{subject_id}/assessment/start").status_code == 200
    assert (
        harness.client.post(f"/api/subjects/{subject_id}/assessment/answer", json={"answer": "回答一"}).status_code
        == 200
    )
    last = harness.client.post(f"/api/subjects/{subject_id}/assessment/answer", json={"answer": "回答二"})
    assert last.status_code == 200


def run_path(harness: ApiHarness, subject_id: str, *, nodes: int = 3) -> None:
    script(harness, blocking=[path_json(nodes)])
    assert harness.client.post(f"/api/subjects/{subject_id}/path/generate").status_code == 200
    assert harness.client.post(f"/api/subjects/{subject_id}/path/confirm").status_code == 200


# --------------------------------------------------------------------------- #
# 主题接口
# --------------------------------------------------------------------------- #


def test_create_and_list_subjects(api: ApiHarness) -> None:
    subject_id = create_subject(api, "贝叶斯定理", goal="通过面试")

    listed = api.client.get("/api/subjects").json()["items"]
    assert len(listed) == 1
    assert listed[0]["id"] == subject_id
    assert listed[0]["title"] == "贝叶斯定理"
    assert listed[0]["status"] == "draft"
    assert listed[0]["progress"]["total_nodes"] == 0
    assert listed[0]["duplicate_title_hint"] is False


def test_create_subject_rejects_invalid_title(api: ApiHarness) -> None:
    response = api.client.post("/api/subjects", json={"title": "!!!"})
    assert response.status_code == 422
    assert "不像是可以学习的内容" in response.json()["detail"]


def test_duplicate_title_is_a_hint_not_an_error(api: ApiHarness) -> None:
    """FR-107：允许同名，只提示。"""
    create_subject(api, "同名概念")
    response = api.client.post("/api/subjects", json={"title": "同名概念"})

    assert response.status_code == 201
    assert response.json()["duplicate_title_hint"] is True


def test_get_update_archive_delete_subject(api: ApiHarness) -> None:
    subject_id = create_subject(api)

    detail = api.client.get(f"/api/subjects/{subject_id}").json()
    assert detail["subject"]["title"] == "贝叶斯定理"
    assert "progress" in detail

    updated = api.client.patch(f"/api/subjects/{subject_id}", json={"title": "新标题"}).json()
    assert updated["subject"]["title"] == "新标题"

    archived = api.client.post(f"/api/subjects/{subject_id}/archive", json={"archived": True}).json()
    assert archived["subject"]["archived"] is True
    assert api.client.get("/api/subjects").json()["items"] == []
    assert len(api.client.get("/api/subjects?include_archived=true").json()["items"]) == 1

    assert api.client.delete(f"/api/subjects/{subject_id}").status_code == 204
    assert api.client.get(f"/api/subjects/{subject_id}").status_code == 404


def test_unknown_subject_returns_404(api: ApiHarness) -> None:
    assert api.client.get("/api/subjects/nope").status_code == 404
    assert api.client.get("/api/subjects/nope/state").status_code == 404
    assert api.client.post("/api/subjects/nope/teach/start").status_code == 404


def test_state_endpoint_guides_first_step(api: ApiHarness) -> None:
    """FR-410：刷新恢复的入口要告诉前端"下一步做什么"。"""
    subject_id = create_subject(api)

    body = api.client.get(f"/api/subjects/{subject_id}/state").json()
    assert body["session"]["state"] == "draft"
    assert body["session"]["next_action"] == "assess"
    assert body["can_answer"] is False
    assert body["path"] is None


def test_action_in_wrong_state_is_rejected(api: ApiHarness) -> None:
    """还没测评就暂停/继续 → 409（请求合法，但当前状态不允许）。"""
    subject_id = create_subject(api)

    assert api.client.post(f"/api/subjects/{subject_id}/teach/pause").status_code == 409
    assert api.client.post(f"/api/subjects/{subject_id}/teach/resume").status_code == 409
    assert api.client.get(f"/api/subjects/{subject_id}/state").json()["session"]["state"] == "draft"


# --------------------------------------------------------------------------- #
# 测评接口
# --------------------------------------------------------------------------- #


def test_assessment_start_streams_question_and_emits_event(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    script(api, blocking=[TOPICS_JSON])

    response = api.client.post(f"/api/subjects/{subject_id}/assessment/start")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)

    assert token_text(events) == "先用你自己的话说说它解决什么问题？"
    assessment_events = events_of(events, "assessment")
    assert len(assessment_events) == 1
    assert assessment_events[0].data["status"] == "question"
    assert assessment_events[0].data["topics"] == TOPICS
    assert assessment_events[0].data["total_turns"] == TOTAL_TURNS
    assert events_of(events, "done")


def test_assessment_stream_carries_streamed_question(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    script(api, blocking=[TOPICS_JSON], stream=["第二方面的问题？"])
    api.client.post(f"/api/subjects/{subject_id}/assessment/start")

    events = parse_sse(
        api.client.post(f"/api/subjects/{subject_id}/assessment/answer", json={"answer": "回答"}).text
    )

    assert token_text(events) == "第二方面的问题？"
    payload = events_of(events, "assessment")[0].data
    assert payload["turn_number"] == 1
    assert payload["current_topic"] == TOPICS[1]


def test_assessment_completes_with_profile(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    script(api, blocking=[TOPICS_JSON, profile_json("basic")], stream=["第二个方面的问题？"])
    api.client.post(f"/api/subjects/{subject_id}/assessment/start")
    api.client.post(f"/api/subjects/{subject_id}/assessment/answer", json={"answer": "回答一"})

    # 子领域已全部覆盖 → 本次作答产出画像
    events = parse_sse(
        api.client.post(f"/api/subjects/{subject_id}/assessment/answer", json={"answer": "回答二"}).text
    )
    payload = events_of(events, "assessment")[0].data
    assert payload["status"] == "profile"
    assert payload["profile"]["summary"] == "需要从基础开始。"
    # 轮次字段必须照实填：前端用它渲染"已考察 n/N"，缺省 0 会显示成 "2/0"
    assert payload["turn_number"] == TOTAL_TURNS
    assert payload["total_turns"] == TOTAL_TURNS

    body = api.client.get(f"/api/subjects/{subject_id}/assessment").json()
    assert body["assessment"]["status"] == "completed"
    assert body["assessment"]["levels"][0]["level"] == "basic"

    state = api.client.get(f"/api/subjects/{subject_id}/state").json()
    assert state["session"]["state"] == "path_review"
    assert state["session"]["next_action"] == "confirm_path"


def test_assessment_empty_reply_surfaces_an_error(api: ApiHarness) -> None:
    """模型只给思维链、正文为空时必须是**错误事件**，不能当成成功。

    真实观测到的失效（2026-10-13）：`completion=411 reasoning=411`，正文一个字都没有。
    以前这条路径被当成成功——落一条空回复、返回 200，界面上用户提交完回答后什么都不发生，
    看着像卡死。同样的"空内容"在结构化输出通道早有处理，流式通道一直漏着。
    """
    subject_id = create_subject(api)
    script(api, blocking=[TOPICS_JSON], stream=["第一个方面的问题？"])
    api.client.post(f"/api/subjects/{subject_id}/assessment/start")
    script(api, stream=[""])  # 正文为空

    events = parse_sse(
        api.client.post(f"/api/subjects/{subject_id}/assessment/answer", json={"answer": "我的回答"}).text
    )

    assert token_text(events) == ""
    errors = events_of(events, "error")
    assert errors, [event.type for event in events]
    assert errors[0].data["category"] == "empty_content"
    assert errors[0].data["retryable"] is True
    # 空回复不得落进对话记录，否则刷新后会上来一个空气泡
    transcript = api.client.get(f"/api/subjects/{subject_id}/assessment").json()["transcript"]
    assert all(item["content"].strip() for item in transcript), transcript


def test_assessment_transcript_survives_refresh(api: ApiHarness) -> None:
    """FR-410：刷新后必须能重新看到"正在问的那个问题"与已有的问答。"""
    subject_id = create_subject(api)
    script(api, blocking=[TOPICS_JSON], stream=["第二个方面的问题？"])
    api.client.post(f"/api/subjects/{subject_id}/assessment/start")
    api.client.post(f"/api/subjects/{subject_id}/assessment/answer", json={"answer": "回答一"})

    body = api.client.get(f"/api/subjects/{subject_id}/assessment").json()

    assert body["pending_question"] == "第二个方面的问题？"
    assert body["transcript"] == [
        {"role": "assistant", "content": "先用你自己的话说说它解决什么问题？"},
        {"role": "user", "content": "回答一"},
        {"role": "assistant", "content": "第二个方面的问题？"},
    ]

    # 未开始测评时不给噪声数据
    fresh = create_subject(api, "另一个概念")
    empty = api.client.get(f"/api/subjects/{fresh}/assessment").json()
    assert empty["assessment"] is None
    assert empty["transcript"] == []
    assert empty["pending_question"] is None


def test_assessment_skip(api: ApiHarness) -> None:
    subject_id = create_subject(api)

    body = api.client.post(f"/api/subjects/{subject_id}/assessment/skip").json()

    assert body["skipped"] is True
    assert api.client.get(f"/api/subjects/{subject_id}/state").json()["session"]["state"] == "path_review"
    assert api.client.get(f"/api/subjects/{subject_id}/assessment").json()["assessment"]["skipped"] is True


def test_override_level_asks_for_path_regeneration(api: ApiHarness) -> None:
    """FR-207：修正画像后必须重新生成路径（否则路径与画像不一致）。"""
    subject_id = create_subject(api)
    run_assessment(api, subject_id, level="none")

    body = api.client.post(
        f"/api/subjects/{subject_id}/assessment/level", json={"topic": TOPICS[0], "level": "proficient"}
    ).json()

    assert body["regenerate_path"] is True
    level = body["assessment"]["levels"][0]
    assert level["level"] == "proficient"
    assert level["user_overridden"] is True


def test_override_unknown_topic_returns_404(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    run_assessment(api, subject_id)

    response = api.client.post(
        f"/api/subjects/{subject_id}/assessment/level", json={"topic": "不存在", "level": "basic"}
    )
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# 路径接口
# --------------------------------------------------------------------------- #


def test_path_generate_emits_path_event(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    run_assessment(api, subject_id)
    script(api, blocking=[path_json(3)])

    events = parse_sse(api.client.post(f"/api/subjects/{subject_id}/path/generate").text)

    payload = events_of(events, "path")[0].data
    assert payload["version"] == 1
    assert "依据画像" in payload["rationale"]
    assert len(payload["nodes"]) == 3
    assert payload["nodes"][0]["checkpoints"][0]["question"] == "节点 0 的检查点？"

    state = api.client.get(f"/api/subjects/{subject_id}/state").json()
    assert state["session"]["state"] == "path_review"
    assert len(state["path"]["nodes"]) == 3


def test_path_view_includes_profile_and_versions(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    run_assessment(api, subject_id)
    run_path(api, subject_id, nodes=4)

    body = api.client.get(f"/api/subjects/{subject_id}/path").json()

    assert body["path"]["version"] == 1
    assert body["current_index"] == 1
    assert body["completed"] is False
    assert body["profile"][0]["topic"] == TOPICS[0]
    assert len(body["versions"]) == 1


def test_path_generate_before_assessment_is_rejected(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    script(api, blocking=[])

    events = parse_sse(api.client.post(f"/api/subjects/{subject_id}/path/generate").text)

    error = events_of(events, "error")
    assert len(error) == 1
    assert "知识基础确认" in error[0].data["message"]
    assert error[0].data["retryable"] is False


def test_path_generate_failure_is_retryable_error(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    run_assessment(api, subject_id)
    script(api, blocking=["坏数据", "坏数据"])

    events = parse_sse(api.client.post(f"/api/subjects/{subject_id}/path/generate").text)

    error = events_of(events, "error")[0]
    assert error.data["category"] == "structured_output"
    assert error.data["retryable"] is True
    assert events_of(events, "path") == []


def test_path_confirm_gate(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    run_assessment(api, subject_id)

    # 还没生成路径就确认 → 409（请求合法，但当前状态不允许）
    response = api.client.post(f"/api/subjects/{subject_id}/path/confirm")
    assert response.status_code == 409

    run_path(api, subject_id, nodes=3)
    confirmed = api.client.post(f"/api/subjects/{subject_id}/path/confirm").json()
    assert confirmed["state"] == "teaching"


# --------------------------------------------------------------------------- #
# 路径编辑接口（FR-304、FR-305）
# --------------------------------------------------------------------------- #


def fetch_nodes(api: ApiHarness, subject_id: str) -> list[dict]:
    body = api.client.get(f"/api/subjects/{subject_id}/path").json()
    return body["path"]["nodes"]


def test_path_edit_round_trip(api: ApiHarness) -> None:
    """改内容 → 插节点 → 移动 → 删除，每一步都返回最新路径且序号连续。"""
    subject_id = prepare_teaching(api)
    nodes = fetch_nodes(api, subject_id)
    assert [node["index"] for node in nodes] == [1, 2, 3]

    updated = api.client.patch(
        f"/api/subjects/{subject_id}/path/nodes/{nodes[0]['id']}",
        json={"title": "改写后的节点", "key_points": ["新要点 A", "新要点 B"]},
    ).json()["path"]
    assert updated["nodes"][0]["title"] == "改写后的节点"
    assert updated["nodes"][0]["key_points"] == ["新要点 A", "新要点 B"]

    added = api.client.post(
        f"/api/subjects/{subject_id}/path/nodes", json={"after_node_id": nodes[0]["id"]}
    ).json()["path"]
    assert len(added["nodes"]) == 4
    assert added["nodes"][1]["title"] == "新的知识点"
    assert [node["index"] for node in added["nodes"]] == [1, 2, 3, 4]

    moved = api.client.post(
        f"/api/subjects/{subject_id}/path/nodes/{nodes[0]['id']}/move", json={"offset": 1}
    ).json()["path"]
    assert [node["title"] for node in moved["nodes"]][:2] == ["新的知识点", "改写后的节点"]

    removed = api.client.delete(f"/api/subjects/{subject_id}/path/nodes/{nodes[0]['id']}").json()["path"]
    assert len(removed["nodes"]) == 3
    assert [node["index"] for node in removed["nodes"]] == [1, 2, 3]

    # 编辑落盘：重新读取仍是编辑后的内容
    assert len(fetch_nodes(api, subject_id)) == 3


def test_path_edit_rejects_started_node(api: ApiHarness) -> None:
    subject_id = prepare_teaching(api)
    script(api, stream=["讲解。"])
    api.client.post(f"/api/subjects/{subject_id}/teach/start")
    current = api.client.get(f"/api/subjects/{subject_id}/teach/current").json()
    node_id = current["node"]["id"]

    deleted = api.client.delete(f"/api/subjects/{subject_id}/path/nodes/{node_id}")
    moved = api.client.post(f"/api/subjects/{subject_id}/path/nodes/{node_id}/move", json={"offset": 1})

    assert deleted.status_code == 409
    assert moved.status_code == 409
    assert "已经进入学习" in deleted.json()["detail"]
    # 但改内容允许（纠错不该被学习进度挡住）
    patched = api.client.patch(f"/api/subjects/{subject_id}/path/nodes/{node_id}", json={"title": "纠错"})
    assert patched.status_code == 200


def test_path_edit_unknown_node_returns_conflict(api: ApiHarness) -> None:
    subject_id = prepare_teaching(api)

    response = api.client.patch(f"/api/subjects/{subject_id}/path/nodes/nope", json={"title": "x"})

    assert response.status_code == 409
    assert "节点不存在" in response.json()["detail"]


def test_path_regenerate_node_emits_path_event(api: ApiHarness) -> None:
    subject_id = prepare_teaching(api)
    nodes = fetch_nodes(api, subject_id)
    script(api, blocking=[json.dumps(node_json("重生成后的节点"), ensure_ascii=False)])

    events = parse_sse(
        api.client.post(f"/api/subjects/{subject_id}/path/nodes/{nodes[0]['id']}/regenerate", json={}).text
    )

    payload = events_of(events, "path")[0].data
    updated = payload["nodes"][0]
    assert updated["id"] == nodes[0]["id"]  # id 不变，作答记录仍挂得上
    assert updated["title"] == "重生成后的节点"
    assert len(payload["nodes"]) == 3


def test_path_regenerate_node_accepts_missing_body(api: ApiHarness) -> None:
    """请求体可省略：前端不带 requirement 时不该 422。"""
    subject_id = prepare_teaching(api)
    nodes = fetch_nodes(api, subject_id)
    script(api, blocking=[json.dumps(node_json("重生成后的节点"), ensure_ascii=False)])

    response = api.client.post(f"/api/subjects/{subject_id}/path/nodes/{nodes[0]['id']}/regenerate")

    assert response.status_code == 200
    payload = events_of(parse_sse(response.text), "path")[0].data
    assert payload["nodes"][0]["title"] == "重生成后的节点"


def test_path_regenerate_node_carries_requirement(api: ApiHarness) -> None:
    subject_id = prepare_teaching(api)
    nodes = fetch_nodes(api, subject_id)
    script(api, blocking=[json.dumps(node_json("更基础的讲法"), ensure_ascii=False)])

    api.client.post(
        f"/api/subjects/{subject_id}/path/nodes/{nodes[0]['id']}/regenerate",
        json={"requirement": "讲得更基础一些"},
    )

    assert "讲得更基础一些" in api.stub.calls[-1].messages[0]["content"]


def test_path_regenerate_whole_path_creates_new_version(api: ApiHarness) -> None:
    subject_id = prepare_teaching(api)
    script(api, blocking=[path_json(4)])

    events = parse_sse(api.client.post(f"/api/subjects/{subject_id}/path/regenerate").text)

    payload = events_of(events, "path")[0].data
    assert payload["version"] == 2
    assert len(payload["nodes"]) == 4

    body = api.client.get(f"/api/subjects/{subject_id}/path").json()
    assert body["path"]["version"] == 2
    assert len(body["versions"]) == 2
    # 重新生成后回到待确认：新版本要重新确认才能继续教学（FR-306）
    assert api.client.get(f"/api/subjects/{subject_id}/state").json()["session"]["state"] == "path_review"


def test_path_regenerate_failure_keeps_current_path(api: ApiHarness) -> None:
    subject_id = prepare_teaching(api)
    script(api, blocking=["坏数据", "坏数据"])

    events = parse_sse(api.client.post(f"/api/subjects/{subject_id}/path/regenerate").text)

    assert events_of(events, "path") == []
    assert events_of(events, "error")[0].data["retryable"] is True
    assert api.client.get(f"/api/subjects/{subject_id}/path").json()["path"]["version"] == 1


# --------------------------------------------------------------------------- #
# 教学接口
# --------------------------------------------------------------------------- #


def prepare_teaching(api: ApiHarness, *, nodes: int = 3) -> str:
    subject_id = create_subject(api)
    run_assessment(api, subject_id)
    run_path(api, subject_id, nodes=nodes)
    return subject_id


def test_teach_start_explains_and_asks(api: ApiHarness) -> None:
    subject_id = prepare_teaching(api)
    script(api, stream=["这是讲解内容。"])

    events = parse_sse(api.client.post(f"/api/subjects/{subject_id}/teach/start").text)

    text = token_text(events)
    assert "这是讲解内容。" in text
    assert "节点 0 的检查点？" in text

    node_state = events_of(events, "node_state")
    assert node_state[0].data["state"] == "learning"
    assert node_state[0].data["index"] == 1
    assert node_state[0].data["total"] == 3

    # 待回答问题落库，刷新后可恢复
    state = api.client.get(f"/api/subjects/{subject_id}/state").json()
    assert state["can_answer"] is True
    assert state["session"]["pending_question"] == "节点 0 的检查点？"
    assert state["session"]["next_action"] == "answer"

    current = api.client.get(f"/api/subjects/{subject_id}/teach/current").json()
    assert current["node"]["title"] == "节点 0"
    assert current["checkpoint"]["question"] == "节点 0 的检查点？"
    assert current["answers"] == []  # 还没有作答记录


def test_teach_answers_survive_refresh(api: ApiHarness) -> None:
    """FR-410：刷新后判定卡片与作答历史要能重建。"""
    subject_id = prepare_teaching(api)
    script(api, stream=["讲解。"])
    api.client.post(f"/api/subjects/{subject_id}/teach/start")
    script(api, blocking=[verdict_json("fail")])
    api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": "错答"})

    current = api.client.get(f"/api/subjects/{subject_id}/teach/current").json()
    assert current["can_answer"] is True
    assert len(current["answers"]) == 1

    record = current["answers"][0]
    assert record["question"] == "节点 0 的检查点？"
    assert record["answer"] == "错答"
    assert record["verdict"] == "fail"
    assert record["missing_points"] == ["判定要点"]
    assert record["attempt_index"] == 1


def test_teach_answer_pass_advances(api: ApiHarness) -> None:
    subject_id = prepare_teaching(api)
    script(api, stream=["讲解。"])
    api.client.post(f"/api/subjects/{subject_id}/teach/start")
    script(api, stream=["小结。"], blocking=[verdict_json("pass")])

    events = parse_sse(
        api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": "我的回答"}).text
    )

    verdicts = events_of(events, "verdict")
    assert verdicts[0].data["verdict"] == "pass"
    assert verdicts[0].data["hit_points"] == ["判定要点"]

    node_state = events_of(events, "node_state")[0].data
    assert node_state["state"] == "mastered"

    # 还有后续节点 → 不结束；下一步是继续讲解
    state = api.client.get(f"/api/subjects/{subject_id}/state").json()
    assert state["session"]["state"] == "teaching"
    assert state["session"]["next_action"] == "teach"
    assert state["node"]["state"] == "mastered"


def test_teach_reteach_then_options_then_decision(api: ApiHarness) -> None:
    """完整走一遍 FR-406 + FR-408：重讲 → 重试耗尽 → 三选一。"""
    app_settings = api.app.state.ctx.settings_service.save(AppSettings(node_max_retries=1))  # type: ignore[attr-defined]
    api.app.state.ctx.refresh_settings(app_settings)  # type: ignore[attr-defined]

    subject_id = prepare_teaching(api)
    script(api, stream=["讲解。"])
    api.client.post(f"/api/subjects/{subject_id}/teach/start")

    script(api, stream=["重讲。"], blocking=[verdict_json("fail")])
    first = parse_sse(
        api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": "错答一"}).text
    )
    assert events_of(first, "verdict")[0].data["verdict"] == "fail"
    assert events_of(first, "options") == []
    assert "节点 0 的检查点？" in token_text(first)  # 重讲后再次提问

    script(api, blocking=[verdict_json("fail")])
    second = parse_sse(
        api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": "错答二"}).text
    )
    options = events_of(second, "options")
    assert len(options) == 1
    assert options[0].data["kind"] == "exhausted"
    assert options[0].data["choices"] == ["reteach", "show_answer", "advance"]

    # 停止自动推进：状态机进入待决策，且不再有待回答问题
    state = api.client.get(f"/api/subjects/{subject_id}/state").json()
    assert state["needs_decision"] is True
    assert state["can_answer"] is False
    assert state["session"]["next_action"] == "decide"

    # 此时提交作答必须被拒绝（否则用户的选项会被顶掉）
    blocked = parse_sse(
        api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": "插队"}).text
    )
    error = events_of(blocked, "error")[0]
    assert error.data["category"] == "awaiting_decision"
    assert error.data["retryable"] is False

    # 选择「直接进入下一节点」
    decided = parse_sse(
        api.client.post(
            f"/api/subjects/{subject_id}/teach/decision", json={"decision": NodeDecision.ADVANCE.value}
        ).text
    )
    assert "我们先进入下一个节点" in token_text(decided)
    moved = events_of(decided, "node_state")[0].data
    assert moved["state"] == "not_started"
    assert moved["index"] == 2

    # 会话已经落在下一个节点上，状态接口与 node_state 事件一致
    state_after = api.client.get(f"/api/subjects/{subject_id}/state").json()
    assert state_after["needs_decision"] is False
    assert state_after["session"]["next_action"] == "teach"
    assert state_after["node"]["state"] == "not_started"
    assert state_after["node"]["index"] == 2

    # 放弃的节点留在路径里，标记为部分掌握并记入薄弱点
    nodes = api.client.get(f"/api/subjects/{subject_id}/path").json()["path"]["nodes"]
    assert [node["state"] for node in nodes] == ["partial", "not_started", "not_started"]
    weak = api.client.get(f"/api/subjects/{subject_id}/weak-points").json()["items"]
    assert len(weak) == 1
    assert weak[0]["reason"] == "advanced"


def test_teach_decision_show_answer(api: ApiHarness) -> None:
    app_settings = api.app.state.ctx.settings_service.save(AppSettings(node_max_retries=0))  # type: ignore[attr-defined]
    api.app.state.ctx.refresh_settings(app_settings)  # type: ignore[attr-defined]

    subject_id = prepare_teaching(api)
    script(api, stream=["讲解。"])
    api.client.post(f"/api/subjects/{subject_id}/teach/start")
    script(api, blocking=[verdict_json("fail")])
    api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": "错答"})
    script(api, stream=["参考答案与简化讲解。"])

    events = parse_sse(
        api.client.post(
            f"/api/subjects/{subject_id}/teach/decision", json={"decision": NodeDecision.SHOW_ANSWER.value}
        ).text
    )

    assert "参考答案与简化讲解。" in token_text(events)
    weak = api.client.get(f"/api/subjects/{subject_id}/weak-points").json()["items"]
    assert weak[0]["reason"] == "show_answer"
    assert api.client.get(f"/api/subjects/{subject_id}/state").json()["node"]["state"] == "partial"


def test_teach_decision_reteach_with_direction(api: ApiHarness) -> None:
    app_settings = api.app.state.ctx.settings_service.save(AppSettings(node_max_retries=0))  # type: ignore[attr-defined]
    api.app.state.ctx.refresh_settings(app_settings)  # type: ignore[attr-defined]

    subject_id = prepare_teaching(api)
    script(api, stream=["讲解。"])
    api.client.post(f"/api/subjects/{subject_id}/teach/start")
    script(api, blocking=[verdict_json("fail")])
    api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": "错答"})
    script(api, stream=["换个更基础的说法。"])

    events = parse_sse(
        api.client.post(
            f"/api/subjects/{subject_id}/teach/decision",
            json={"decision": NodeDecision.RETEACH.value, "direction": "simpler"},
        ).text
    )

    assert "换个更基础的说法。" in token_text(events)
    state = api.client.get(f"/api/subjects/{subject_id}/state").json()
    assert state["needs_decision"] is False
    assert state["can_answer"] is True  # 重新登记了同一个检查点


def test_teach_skip_node(api: ApiHarness) -> None:
    subject_id = prepare_teaching(api)
    script(api, stream=["讲解。"])
    api.client.post(f"/api/subjects/{subject_id}/teach/start")

    events = parse_sse(api.client.post(f"/api/subjects/{subject_id}/teach/skip").text)

    assert "已跳过该节点" in token_text(events)
    assert events_of(events, "node_state")[0].data["state"] == "skipped"
    weak = api.client.get(f"/api/subjects/{subject_id}/weak-points").json()["items"]
    assert weak[0]["reason"] == "skipped"


def test_teach_pause_blocks_answering(api: ApiHarness) -> None:
    subject_id = prepare_teaching(api)
    script(api, stream=["讲解。"])
    api.client.post(f"/api/subjects/{subject_id}/teach/start")
    api.client.post(f"/api/subjects/{subject_id}/teach/pause")

    events = parse_sse(api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": "x"}).text)
    error = events_of(events, "error")[0]
    assert error.data["category"] == "paused"

    api.client.post(f"/api/subjects/{subject_id}/teach/resume")
    assert api.client.get(f"/api/subjects/{subject_id}/state").json()["session"]["state"] == "teaching"


def test_teach_answer_without_question_is_rejected(api: ApiHarness) -> None:
    """FR-410 的核心不变量：没有待回答问题时的输入不能被当成答案。"""
    subject_id = prepare_teaching(api)

    events = parse_sse(api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": "随便说"}).text)

    error = events_of(events, "error")[0]
    assert error.data["category"] == "no_pending_question"
    assert "没有等待回答的问题" in error.data["message"]


def test_judgement_failure_keeps_pending_question(api: ApiHarness) -> None:
    subject_id = prepare_teaching(api)
    script(api, stream=["讲解。"])
    api.client.post(f"/api/subjects/{subject_id}/teach/start")
    script(api, blocking=["坏数据", "坏数据"])

    events = parse_sse(
        api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": "我的回答"}).text
    )

    error = events_of(events, "error")[0]
    assert error.data["retryable"] is True
    state = api.client.get(f"/api/subjects/{subject_id}/state").json()
    assert state["can_answer"] is True  # 仍可重试
    assert state["session"]["node_attempts"] == 0  # 技术故障不计次


def test_full_loop_completes_subject(api: ApiHarness) -> None:
    """端到端：3 个节点全部通过后主题进入 completed。"""
    subject_id = prepare_teaching(api, nodes=3)

    for index in range(3):
        script(api, stream=[f"讲解 {index}。"])
        api.client.post(f"/api/subjects/{subject_id}/teach/start")
        script(api, stream=[f"小结 {index}。"], blocking=[verdict_json("pass")])
        events = parse_sse(
            api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": f"回答 {index}"}).text
        )
        assert events_of(events, "verdict")[0].data["verdict"] == "pass"

    state = api.client.get(f"/api/subjects/{subject_id}/state").json()
    assert state["session"]["state"] == "completed"
    assert state["session"]["next_action"] == "completed"

    path = api.client.get(f"/api/subjects/{subject_id}/path").json()
    assert path["completed"] is True
    assert all(node["state"] == "mastered" for node in path["path"]["nodes"])
    assert all(node["summary_note"] for node in path["path"]["nodes"])

    progress = api.client.get(f"/api/subjects/{subject_id}").json()["progress"]
    assert progress["mastered_nodes"] == 3
    assert progress["mastery_ratio"] == 1.0


# --------------------------------------------------------------------------- #
# 设置与运维接口
# --------------------------------------------------------------------------- #


def test_settings_round_trip(api: ApiHarness) -> None:
    body = api.client.get("/api/settings").json()["settings"]
    assert body["model_default"] == "deepseek-flash"
    assert body["path_min_nodes"] == 3
    assert body["api_key"]["configured"] is False
    assert "data_dir" in body["read_only_fields"]

    body["node_max_retries"] = 1
    body["assess_max_turns"] = 5
    saved = api.client.put("/api/settings", json=body).json()["settings"]
    assert saved["node_max_retries"] == 1

    reloaded = api.client.get("/api/settings").json()["settings"]
    assert reloaded["assess_max_turns"] == 5


def test_settings_reject_out_of_range(api: ApiHarness) -> None:
    body = api.client.get("/api/settings").json()["settings"]
    body["timeout_seconds"] = 5

    assert api.client.put("/api/settings", json=body).status_code == 422


def test_api_key_is_masked_everywhere(api: ApiHarness) -> None:
    """FR-708：接口与日志里都不能出现明文。"""
    secret = "sk-integration-secret-123456"
    response = api.client.put("/api/settings/api-key", json={"api_key": secret})

    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is True
    assert body["masked"] == "sk-****3456"
    assert secret not in response.text

    settings_text = api.client.get("/api/settings").text
    assert secret not in settings_text
    assert "sk-****3456" in settings_text


def test_usage_endpoint_reports_cache_hit_rate(api: ApiHarness) -> None:
    subject_id = create_subject(api)
    run_assessment(api, subject_id)

    body = api.client.get("/api/usage").json()
    assert body["calls"] > 0
    assert 0.0 <= body["cache_hit_rate"] <= 1.0


def test_health_still_works_with_stack_wired(api: ApiHarness) -> None:
    body = api.client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["api_key"]["configured"] is False


# --------------------------------------------------------------------------- #
# 验收项（AC）
# --------------------------------------------------------------------------- #


def test_missing_api_key_is_actionable(env: dict[str, str]) -> None:
    """AC-02：没配 Key 时要给出"缺什么 + 去哪儿配"，且不崩溃、不推进状态机。

    这里刻意不注入桩客户端：要验证的正是真实客户端在缺配置时的行为，
    而它在发请求**之前**就会拦下来，因此不会产生任何网络调用。
    """
    from app.main import create_app

    app = create_app(env)
    with TestClient(app) as client:
        created = client.post("/api/subjects", json={"title": "贝叶斯定理"})
        assert created.status_code == 201
        subject_id = created.json()["subject"]["id"]

        response = client.post(f"/api/subjects/{subject_id}/assessment/start")

        assert response.status_code == 200  # 流式端点用 error 事件承载业务失败
        error = events_of(parse_sse(response.text), "error")[0].data
        assert error["category"] == "config"
        assert error["retryable"] is False
        assert "API Key" in response.text
        # 状态机不该被推进到 assessing
        assert client.get(f"/api/subjects/{subject_id}/state").json()["session"]["state"] == "draft"


def test_partial_verdict_triggers_reteach(api: ApiHarness) -> None:
    """AC-09：部分通过 → 返回缺失要点，并换角度重讲后再次提问。"""
    subject_id = prepare_teaching(api)
    script(api, stream=["讲解。"])
    api.client.post(f"/api/subjects/{subject_id}/teach/start")

    # 桩的 partial 分支：missing_points 非空、hit_points 为空
    script(api, stream=["换角度再讲一遍。"], blocking=[verdict_json("partial")])
    events = parse_sse(
        api.client.post(f"/api/subjects/{subject_id}/teach/answer", json={"answer": "答了一半"}).text
    )

    verdict = events_of(events, "verdict")[0].data
    assert verdict["verdict"] == "partial"
    assert verdict["missing_points"] == ["判定要点"]
    assert "换角度再讲一遍。" in token_text(events)
    assert "节点 0 的检查点？" in token_text(events)  # 重讲后再次提问

    state = api.client.get(f"/api/subjects/{subject_id}/state").json()
    assert state["can_answer"] is True  # 可以再答一次
    assert state["node"]["state"] == "learning"  # 未通过不推进节点状态
    assert state["needs_decision"] is False  # 还没到重试上限
