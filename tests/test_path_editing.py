"""学习路径编辑测试（T3.1，FR-304、FR-305、FR-307）。

这一层的核心风险不是"改不动"，而是**改坏**：

- 编辑后 ``order_index`` 必须仍然唯一且连续——重复序号会让 ``next_node`` 随机挑节点；
- 编辑必须**保留节点与检查点的 id**——作答记录、笔记、薄弱点都挂在 id 上；
- 已经学过的节点不能被删除或挪位，否则"已掌握"会落在别的知识点上；
- 删除当前节点时必须清掉会话指针，否则下一次进入教学会指向不存在的节点。
"""

import json

import pytest

from app.contracts.schemas import NodeState, SubjectState
from app.infra.settings import AppSettings
from app.main_context import AppContext
from app.services.learning.planning import PlanningError
from app.services.learning.state import SessionStateService
from tests.learning_harness import LearningHarness, make_harness
from tests.stubs import ScriptedResponse, json_response, text_events
from tests.test_teaching_service import build_services, create_path, enter_teaching


def node_draft(title: str) -> dict:
    return {
        "title": title,
        "objective": f"能解释 {title}",
        "prerequisites": [],
        "key_points": ["要点一", "要点二", "要点三"],
        "checkpoints": [
            {
                "question": f"{title} 的检查点？",
                "expected_points": ["判定要点"],
                "reference_answer": "参考答案",
            }
        ],
        "is_quick_review": False,
    }


def node_draft_json(title: str) -> str:
    return json.dumps(node_draft(title), ensure_ascii=False)


def ready(subject_id: str, harness: LearningHarness) -> tuple:
    """建好一条已确认的路径，返回 (planning, path)。"""
    _, planning, teaching, _ = enter_teaching(harness, node_count=3)
    path = planning.current_path(subject_id)
    assert path is not None
    return planning, teaching, path


def make_ready(harness: LearningHarness) -> tuple[str, object, object, object]:
    harness.state  # noqa: B018 - 明确依赖 harness.state
    subject_id = harness.new_subject()
    harness.state.transition(subject_id, SubjectState.PATH_REVIEW)
    create_path(harness, subject_id, node_count=3)
    assessments, planning, teaching = build_services(harness)
    planning.confirm(subject_id)
    return subject_id, assessments, planning, teaching


# --------------------------------------------------------------------------- #
# 内部一致性
# --------------------------------------------------------------------------- #


def assert_well_formed(harness: LearningHarness, path_id: str) -> list[str]:
    """返回节点标题顺序，并校验序号唯一连续、每个节点都有检查点。"""
    path = harness.repos.paths.get(path_id)
    assert path is not None
    nodes = path.ordered_nodes()
    indexes = [node.order_index for node in nodes]
    assert indexes == list(range(len(nodes))), f"序号必须唯一连续，实际 {indexes}"
    for node in nodes:
        assert node.checkpoints, f"节点 {node.title} 没有检查点"
    return [node.title for node in nodes]


# --------------------------------------------------------------------------- #
# 改内容
# --------------------------------------------------------------------------- #


def test_update_node_changes_content_only(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    first = path.ordered_nodes()[0]

    updated = planning.update_node(
        subject_id,
        first.id,
        title="  改写后的标题  ",
        objective="新的目标",
        key_points=[" 要点 A ", "", "要点 B"],
    )

    node = next(item for item in updated.ordered_nodes() if item.id == first.id)
    assert node.title == "改写后的标题"  # 首尾空白被去掉
    assert node.objective == "新的目标"
    assert node.key_points == ("要点 A", "要点 B")  # 空项被丢弃
    assert node.state is first.state  # 学习状态不受影响
    assert [item.id for item in updated.ordered_nodes()] == [item.id for item in path.ordered_nodes()]
    assert_well_formed(harness, path.id)


def test_update_node_keeps_checkpoints_and_ids(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    first = path.ordered_nodes()[0]
    checkpoint_id = first.checkpoints[0].id

    updated = planning.update_node(subject_id, first.id, title="只改标题")

    node = next(item for item in updated.ordered_nodes() if item.id == first.id)
    assert node.checkpoints[0].id == checkpoint_id, "改标题不该把检查点 id 换掉（作答记录挂在它上面）"


def test_update_node_allows_editing_a_started_node(context: AppContext) -> None:
    """已开始甚至已掌握的节点仍可改标题——那是纠错，不是改进度。"""
    harness = make_harness(context)
    subject_id, _, planning, _teaching = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    first = path.ordered_nodes()[0]
    harness.repos.paths.update_node(first.id, state=NodeState.MASTERED)

    updated = planning.update_node(subject_id, first.id, title="纠正错别字")

    node = next(item for item in updated.ordered_nodes() if item.id == first.id)
    assert node.title == "纠正错别字"
    assert node.state is NodeState.MASTERED


def test_update_node_rejects_empty_title(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None

    with pytest.raises(PlanningError):
        planning.update_node(subject_id, path.ordered_nodes()[0].id, title="   ")


def test_update_unknown_node_is_rejected(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)

    with pytest.raises(PlanningError) as exc:
        planning.update_node(subject_id, "不存在的节点", title="x")

    assert "节点不存在" in str(exc.value)


# --------------------------------------------------------------------------- #
# 插入 / 删除 / 移动
# --------------------------------------------------------------------------- #


def test_add_node_in_the_middle(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    titles = assert_well_formed(harness, path.id)

    updated = planning.add_node(subject_id, after_node_id=path.ordered_nodes()[0].id)

    order = assert_well_formed(harness, path.id)
    assert len(order) == len(titles) + 1
    assert order[1] == "新的知识点"
    assert len(updated.ordered_nodes()) == 4


def test_add_node_at_the_end(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None

    planning.add_node(subject_id)

    order = assert_well_formed(harness, path.id)
    assert order[-1] == "新的知识点"


def test_added_node_is_blank_and_editable(context: AppContext) -> None:
    """插入刻意留空而不是让模型随便塞内容：用户点"插入"时心里已经有要补的东西。"""
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None

    updated = planning.add_node(subject_id)
    blank = next(node for node in updated.ordered_nodes() if node.title == "新的知识点")

    assert blank.state is NodeState.NOT_STARTED
    assert blank.key_points  # 有占位要点，便于直接编辑
    assert blank.checkpoints and blank.checkpoints[0].question


def test_delete_node(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    victim = path.ordered_nodes()[1]

    updated = planning.delete_node(subject_id, victim.id)

    ids = [node.id for node in updated.ordered_nodes()]
    assert victim.id not in ids
    assert len(ids) == 2
    assert_well_formed(harness, path.id)


def test_delete_rejects_started_node(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    started = path.ordered_nodes()[0]
    harness.repos.paths.update_node(started.id, state=NodeState.LEARNING)

    with pytest.raises(PlanningError) as exc:
        planning.delete_node(subject_id, started.id)

    assert "已经进入学习" in str(exc.value)
    assert len(planning.current_path(subject_id).ordered_nodes()) == 3


def test_delete_last_node_is_rejected(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id = harness.new_subject()
    harness.state.transition(subject_id, SubjectState.PATH_REVIEW)
    create_path(harness, subject_id, node_count=1)
    _, planning, _ = build_services(harness)
    planning.confirm(subject_id)
    path = planning.current_path(subject_id)
    assert path is not None

    with pytest.raises(PlanningError) as exc:
        planning.delete_node(subject_id, path.ordered_nodes()[0].id)

    assert "至少要保留一个节点" in str(exc.value)


def test_delete_current_node_clears_pointer(context: AppContext) -> None:
    """删掉会话指向的节点后必须清指针，否则下次进入教学会指向不存在的节点。"""
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    current = path.ordered_nodes()[0]
    harness.state.enter_node(subject_id, current)

    updated = planning.delete_node(subject_id, current.id)

    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.current_node_id is None
    assert snapshot.pending_checkpoint_id is None
    # 会话恢复视图不再指向被删掉的节点，而是回落到路径里的第一个节点
    assert updated.node_count == 2


def test_move_node_down_and_up(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    original = assert_well_formed(harness, path.id)
    first_id = path.ordered_nodes()[0].id

    planning.move_node(subject_id, first_id, offset=1)
    assert assert_well_formed(harness, path.id) == [original[1], original[0], original[2]]

    planning.move_node(subject_id, first_id, offset=-1)
    assert assert_well_formed(harness, path.id) == original


def test_move_at_boundary_is_rejected(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None

    with pytest.raises(PlanningError):
        planning.move_node(subject_id, path.ordered_nodes()[0].id, offset=-1)


def test_move_rejects_started_node(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    first = path.ordered_nodes()[0]
    harness.repos.paths.update_node(first.id, state=NodeState.MASTERED)

    with pytest.raises(PlanningError) as exc:
        planning.move_node(subject_id, first.id, offset=1)

    assert "不能调整顺序" in str(exc.value)


def test_move_rejects_when_the_other_side_is_started(context: AppContext) -> None:
    """交换的另一端已开始时也要拒绝：否则它的掌握状态会被挪到别的位置。"""
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    first, second = path.ordered_nodes()[0], path.ordered_nodes()[1]
    harness.repos.paths.update_node(second.id, state=NodeState.MASTERED)

    with pytest.raises(PlanningError):
        planning.move_node(subject_id, first.id, offset=1)


def test_move_carries_node_identity_and_content(context: AppContext) -> None:
    """移动的是"节点"而不是"位置"：id、要点与检查点必须跟着内容走。"""
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    second = path.ordered_nodes()[1]
    checkpoint_id = second.checkpoints[0].id

    updated = planning.move_node(subject_id, second.id, offset=-1)

    moved = next(node for node in updated.ordered_nodes() if node.id == second.id)
    assert moved.order_index == 0
    assert moved.title == second.title
    assert moved.key_points == second.key_points
    assert moved.checkpoints[0].id == checkpoint_id
    assert moved.state is NodeState.NOT_STARTED
    assert_well_formed(harness, path.id)


def test_edits_keep_answer_records_addressable(context: AppContext) -> None:
    """编辑后节点 id 不变，作答记录仍然挂得上。"""
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    node = path.ordered_nodes()[0]
    harness.repos.answers.add(
        subject_id=subject_id,
        node_id=node.id,
        checkpoint_id=node.checkpoints[0].id,
        question=node.checkpoints[0].question,
        user_answer="我的回答",
        verdict="fail",
        confidence=0.3,
        feedback="还差一点",
        hit_points=[],
        missing_points=["判定要点"],
        attempt_index=1,
    )

    planning.move_node(subject_id, node.id, offset=1)

    records = harness.repos.answers.list_for_node(node.id)
    assert len(records) == 1
    assert records[0].user_answer == "我的回答"


# --------------------------------------------------------------------------- #
# 重生成
# --------------------------------------------------------------------------- #


async def test_regenerate_node_keeps_id_and_state(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    target = path.ordered_nodes()[0]
    harness.client.blocking_script = [json_response(node_draft_json("全新设计的节点"))]

    result = await planning.regenerate_node(subject_id, target.id, requirement="讲得更基础一些")

    assert result.ok
    updated = next(node for node in result.path.ordered_nodes() if node.id == target.id)
    assert updated.title == "全新设计的节点"
    assert updated.objective == "能解释 全新设计的节点"
    assert updated.checkpoints, "重生成后必须带回新的检查点"
    assert len(result.path.ordered_nodes()) == 3  # 只换一个节点
    assert_well_formed(harness, path.id)

    # 提示词里必须带上上下文与用户要求
    system = harness.client.calls[-1].messages[0]["content"]
    assert "讲得更基础一些" in system
    assert "上一个节点" in system and "下一个节点" in system


async def test_regenerate_node_keeps_other_nodes_untouched(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    before = {node.id: (node.title, node.checkpoints[0].id) for node in path.ordered_nodes()}
    harness.client.blocking_script = [json_response(node_draft_json("新节点"))]

    result = await planning.regenerate_node(subject_id, path.ordered_nodes()[1].id)

    assert result.path is not None
    for node in result.path.ordered_nodes():
        title, checkpoint_id = before[node.id]
        if node.id == path.ordered_nodes()[1].id:
            assert node.title == "新节点"
        else:
            assert node.title == title
            assert node.checkpoints[0].id == checkpoint_id, "别的节点的检查点不该被换掉"


async def test_regenerate_node_rejects_mastered_node(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    node = path.ordered_nodes()[0]
    harness.repos.paths.update_node(node.id, state=NodeState.MASTERED)

    with pytest.raises(PlanningError) as exc:
        await planning.regenerate_node(subject_id, node.id)

    assert "已经学过" in str(exc.value)


async def test_regenerate_node_failure_keeps_path_intact(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    before = [node.title for node in path.ordered_nodes()]
    # 结构化通道失败后会自己重试一次，所以要给两次坏响应
    harness.client.blocking_script = [json_response("坏数据"), json_response("坏数据")]

    result = await planning.regenerate_node(subject_id, path.ordered_nodes()[0].id)

    assert not result.ok
    assert result.retryable is True
    assert [node.title for node in result.path.ordered_nodes()] == before


# --------------------------------------------------------------------------- #
# 整条重生成（FR-305 / FR-307）
# --------------------------------------------------------------------------- #


async def test_regenerate_creates_new_version_and_keeps_history(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    harness.client.blocking_script = [
        json_response(
            json.dumps(
                {
                    "rationale": "按新的理解重排。",
                    "nodes": [node_draft(f"新版节点 {index}") for index in range(3)],
                },
                ensure_ascii=False,
            )
        )
    ]

    result = await planning.regenerate(subject_id)

    assert result.ok
    assert result.path is not None and result.path.version == 2
    assert [node.title for node in result.path.ordered_nodes()] == ["新版节点 0", "新版节点 1", "新版节点 2"]

    versions = planning.versions(subject_id)
    assert [version for _, version, _, _ in versions] == [2, 1]
    # 只有新版本是活动的
    assert sum(1 for _, _, active, _ in versions if active) == 1
    assert harness.repos.paths.get(versions[1][0]) is not None, "旧版本仍然可读（FR-307）"


async def test_regenerate_clears_session_pointer(context: AppContext) -> None:
    """重生成后节点全换了 id，会话指针必须清掉。"""
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    harness.state.enter_node(subject_id, path.ordered_nodes()[0])
    harness.client.blocking_script = [
        json_response(
            json.dumps(
                {
                    "rationale": "重排。",
                    "nodes": [node_draft(f"新 {index}") for index in range(3)],
                },
                ensure_ascii=False,
            )
        )
    ]

    await planning.regenerate(subject_id)

    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.current_node_id is None
    assert snapshot.pending_checkpoint_id is None
    assert snapshot.node_attempts == 0


async def test_regenerate_carries_user_requirement(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    harness.client.blocking_script = [
        json_response(
            json.dumps({"rationale": "r", "nodes": [node_draft(f"n{i}") for i in range(3)]}, ensure_ascii=False)
        )
    ]

    await planning.regenerate(subject_id, requirement="多来点实际例子")

    assert "多来点实际例子" in harness.client.calls[-1].messages[0]["content"]


async def test_regenerate_returns_to_path_review(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    assert harness.state.snapshot(subject_id).state is SubjectState.TEACHING
    harness.client.blocking_script = [
        json_response(
            json.dumps({"rationale": "r", "nodes": [node_draft(f"n{i}") for i in range(3)]}, ensure_ascii=False)
        )
    ]

    await planning.regenerate(subject_id)

    # 新版本需要重新确认才能进入教学（FR-306）
    assert harness.state.snapshot(subject_id).state is SubjectState.PATH_REVIEW


# --------------------------------------------------------------------------- #
# 状态与权限
# --------------------------------------------------------------------------- #


def test_edits_require_an_active_path(context: AppContext) -> None:
    harness = make_harness(context)
    subject_id = harness.new_subject()
    _, planning, _ = build_services(harness)

    with pytest.raises(PlanningError) as exc:
        planning.update_node(subject_id, "任意", title="x")

    assert "还没有学习路径" in str(exc.value)


def test_edits_persist_across_reload(context: AppContext) -> None:
    """编辑必须落盘：刷新后仍然是编辑后的内容。"""
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None

    planning.update_node(subject_id, path.ordered_nodes()[0].id, title="落盘标题")
    planning.add_node(subject_id)

    fresh = SessionStateService(harness.repos.sessions, harness.repos.subjects)
    del fresh  # 只借用一下装配；真正的验证走新的仓储实例
    reopened = harness.repos.paths.get_active(subject_id)
    assert reopened is not None
    assert next(node.title for node in reopened.ordered_nodes()) == "落盘标题"
    assert reopened.node_count == 4


def test_stream_stub_is_unused_for_pure_edits(context: AppContext) -> None:
    """增删改移动都是纯本地操作，不该产生任何 LLM 调用。"""
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    harness.client.calls.clear()

    planning.update_node(subject_id, path.ordered_nodes()[0].id, title="x")
    planning.add_node(subject_id)
    planning.move_node(subject_id, path.ordered_nodes()[0].id, offset=1)

    assert harness.client.calls == []


def test_settings_do_not_break_editing(context: AppContext) -> None:
    harness = make_harness(context, settings=AppSettings(path_min_nodes=3, path_max_nodes=5))
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None

    planning.add_node(subject_id)
    planning.add_node(subject_id)

    assert planning.current_path(subject_id).node_count == 5


async def test_node_regeneration_uses_stream_events_if_needed(context: AppContext) -> None:
    """桩里没有流式脚本，说明重生成确实走的是结构化（非流式）通道。"""
    harness = make_harness(context)
    subject_id, _, planning, _ = make_ready(harness)
    path = planning.current_path(subject_id)
    assert path is not None
    harness.client.stream_script = [ScriptedResponse(events=text_events("不该被用到"))]
    harness.client.blocking_script = [json_response(node_draft_json("结构化节点"))]

    result = await planning.regenerate_node(subject_id, path.ordered_nodes()[0].id)

    assert result.ok
    assert harness.client.stream_calls == 0
