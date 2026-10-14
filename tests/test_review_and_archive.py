"""复习与学习档案（T3.3/T3.4，FR-414、FR-502~FR-504、FR-507、FR-801、FR-802、FR-804、FR-417）。"""


import pytest

from app.contracts.schemas import ConversationKind, NodeState, WeakPointReason
from app.main_context import AppContext
from app.services.learning.review import ReviewError, ReviewService
from tests.learning_harness import LearningHarness, make_harness
from tests.stubs import ScriptedResponse, text_events
from tests.test_teaching_service import (
    enter_teaching,
    verdict_json,
)


def script(harness: LearningHarness, *, blocking: list[str] | None = None, stream: list[str] | None = None) -> None:
    """设置脚本并**重置调用计数**，避免阶段之间互相干扰。"""
    from tests.stubs import json_response

    harness.client.blocking_script = [json_response(item) for item in (blocking or [])]
    harness.client.stream_script = [ScriptedResponse(events=text_events(item)) for item in (stream or [])]
    harness.client.blocking_calls = 0
    harness.client.stream_calls = 0


def make_review(harness: LearningHarness, planning) -> ReviewService:
    return ReviewService(harness.repos, harness.state, harness.sessions, planning)


def open_weak_point(harness: LearningHarness, subject_id: str, node_id: str) -> None:
    """把某个节点变成"部分掌握 + 有未解决薄弱点"，也就是复习页上的样子。"""
    harness.repos.paths.update_node(node_id, state=NodeState.PARTIAL)
    harness.repos.weak_points.record(subject_id, node_id, WeakPointReason.SHOW_ANSWER)


def first_node(harness: LearningHarness, subject_id: str):
    path = harness.repos.paths.get_active(subject_id)
    assert path is not None
    return path.ordered_nodes()[0]


# --------------------------------------------------------------------------- #
# 薄弱点清单（FR-502）
# --------------------------------------------------------------------------- #


async def test_overview_lists_open_weak_points_with_node_titles(context: AppContext) -> None:
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = first_node(harness, subject_id)
    open_weak_point(harness, subject_id, node.id)

    overview = reviews.overview(subject_id)

    assert overview.open_count == 1
    assert overview.node_total == 2
    assert overview.items[0].node_title == node.title
    assert overview.items[0].node_index == 1


async def test_resolved_weak_point_leaves_the_list(context: AppContext) -> None:
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = first_node(harness, subject_id)
    open_weak_point(harness, subject_id, node.id)

    harness.repos.weak_points.resolve(subject_id, node.id)

    assert reviews.overview(subject_id).open_count == 0


# --------------------------------------------------------------------------- #
# 复习会话（FR-503、FR-504、FR-507；AC-22）
# --------------------------------------------------------------------------- #


async def test_review_start_reuses_an_existing_checkpoint_without_calling_the_model(
    context: AppContext,
) -> None:
    """复习题的来源是节点已有的检查点，因此开局不该产生任何模型调用。"""
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = first_node(harness, subject_id)
    open_weak_point(harness, subject_id, node.id)
    before = (harness.client.stream_calls, harness.client.blocking_calls)

    turn = reviews.start(subject_id, node.id)

    assert turn.ok
    assert turn.phase == "question"
    assert turn.checkpoint is not None
    assert turn.checkpoint.question in turn.text
    assert (harness.client.stream_calls, harness.client.blocking_calls) == before
    # 复习状态要落库，否则刷新后就丢掉"正在复习哪个节点"
    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.reviewing
    assert snapshot.review_node_id == node.id


async def test_review_pass_promotes_node_and_resolves_weak_point(context: AppContext) -> None:
    """AC-22：复习通过 → 节点升级为 mastered，薄弱点标记为已解决。"""
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = first_node(harness, subject_id)
    open_weak_point(harness, subject_id, node.id)
    reviews.start(subject_id, node.id)

    script(harness, blocking=[verdict_json("pass")])
    turn = await reviews.answer(subject_id, "这次我讲清楚了因果")

    assert turn.ok
    assert turn.phase == "passed"
    assert turn.promoted is True
    assert turn.resolved is True
    promoted = harness.repos.paths.node(node.id)
    assert promoted is not None and promoted.state is NodeState.MASTERED
    assert harness.repos.weak_points.count_open(subject_id) == 0
    # 复习结束，状态清干净
    assert harness.state.snapshot(subject_id).reviewing is False


async def test_review_fail_keeps_weak_point_and_counts_failures(context: AppContext) -> None:
    """FR-504：复习不通过要保留薄弱点、累计失败次数，并给出更基础的建议。"""
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = first_node(harness, subject_id)
    open_weak_point(harness, subject_id, node.id)
    reviews.start(subject_id, node.id)

    script(harness, blocking=[verdict_json("fail")])
    turn = await reviews.answer(subject_id, "还是没讲清")

    assert turn.ok
    assert turn.phase == "failed"
    assert turn.promoted is False
    assert turn.fail_count == 2  # 建薄弱点时算一次，这次复习失败再累加一次
    assert turn.suggestion
    assert "判定要点 0" in turn.suggestion  # 建议里点名了缺什么
    still = harness.repos.paths.node(node.id)
    assert still is not None and still.state is NodeState.PARTIAL  # 状态不被改动
    assert harness.repos.weak_points.count_open(subject_id) == 1


async def test_review_records_are_marked_as_review(context: AppContext) -> None:
    """FR-507：复习作答要能跟教学作答区分开，否则统计不出"复习了几次"。"""
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = first_node(harness, subject_id)
    open_weak_point(harness, subject_id, node.id)
    reviews.start(subject_id, node.id)

    script(harness, blocking=[verdict_json("fail")])
    await reviews.answer(subject_id, "答得不好")

    history = reviews.history(subject_id, node.id)
    assert len(history) == 1
    assert history[0].source == "review"
    # 教学作答不受影响
    everything = harness.repos.answers.list_for_node(node.id)
    assert [record.source for record in everything] == ["review"]


async def test_review_answer_without_a_started_review_is_rejected(context: AppContext) -> None:
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)

    with pytest.raises(ReviewError):
        await reviews.answer(subject_id, "随手一句")


async def test_review_uses_its_own_memory_container(context: AppContext) -> None:
    """FR-602：复习记忆与教学记忆相互隔离。"""
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = first_node(harness, subject_id)
    open_weak_point(harness, subject_id, node.id)
    reviews.start(subject_id, node.id)

    script(harness, blocking=[verdict_json("fail")])
    await reviews.answer(subject_id, "复习里的回答")

    assert harness.ctx.memories.conversation_id_for(subject_id, ConversationKind.REVIEW) is not None
    teach_id = harness.ctx.memories.conversation_id_for(subject_id, ConversationKind.TEACH)
    review_id = harness.ctx.memories.conversation_id_for(subject_id, ConversationKind.REVIEW)
    assert teach_id != review_id
    teach_contents = [
        message["content"] for message in harness.ctx.memories.load(teach_id).outbound_messages()
    ]
    assert "复习里的回答" not in teach_contents


async def test_review_judgement_failure_writes_nothing(context: AppContext) -> None:
    """判定不可用是技术故障：不写记录、不改状态，用户直接重试。"""
    harness = make_harness(context)
    _, planning, _, subject_id = enter_teaching(harness, node_count=2)
    reviews = make_review(harness, planning)
    node = first_node(harness, subject_id)
    open_weak_point(harness, subject_id, node.id)
    reviews.start(subject_id, node.id)
    before = harness.state.snapshot(subject_id)

    script(harness, blocking=["坏数据", "坏数据"])  # 结构化通道会自己重试一次
    turn = await reviews.answer(subject_id, "回答")

    assert turn.ok is False
    assert turn.retryable is True
    assert harness.repos.answers.list_for_node(node.id) == []
    after = harness.state.snapshot(subject_id)
    assert after.review_node_id == before.review_node_id  # 复习仍在进行中，可直接重试


# --------------------------------------------------------------------------- #
# 相邻复习（FR-414、FR-501）
# --------------------------------------------------------------------------- #


async def test_first_node_has_no_adjacent_review(context: AppContext) -> None:
    """第一个节点没有"上一节点"可回顾。"""
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    script(harness, stream=["讲解一。"])

    turn = await teaching.start(subject_id)

    assert turn.phase == "question"
    assert turn.text.startswith("讲解一。")
    assert harness.state.snapshot(subject_id).reviewing is False


async def test_adjacent_review_is_asked_before_the_next_node(context: AppContext) -> None:
    """通过第一个节点后进入第二个节点前，应先问一道关于上一节点的题。"""
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    await pass_current_node(harness, teaching, subject_id)

    script(harness, stream=["不该被调用"])  # 回顾环节本身不产生讲解
    turn = await teaching.start(subject_id)

    assert turn.phase == "review_question"
    assert "回顾" in turn.text
    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.review_mode == "adjacent"
    assert snapshot.has_review_question
    assert harness.client.stream_calls == 0


async def test_adjacent_review_pass_then_teaches_the_new_node(context: AppContext) -> None:
    """答对回顾 → 简述带过并接着讲解新节点。"""
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    await pass_current_node(harness, teaching, subject_id)
    script(harness, stream=["回顾问题?"])
    await teaching.start(subject_id)

    script(harness, blocking=[verdict_json("pass")], stream=["节点 1 的讲解。"])
    turn = await teaching.answer(subject_id, "回顾回答")

    assert turn.ok
    assert "回顾" in turn.text and "通过" in turn.text
    assert "节点 1 的讲解。" in turn.text
    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.reviewing is False
    assert snapshot.has_pending_question  # 新节点的问题已经挂上
    node = harness.repos.paths.node(snapshot.current_node_id)
    assert node is not None and node.title == "节点 1"


async def test_adjacent_review_fail_goes_back_to_the_previous_node(context: AppContext) -> None:
    """答错回顾 → 回到上一节点补强，并记下薄弱点。"""
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    previous = await pass_current_node(harness, teaching, subject_id)
    script(harness, stream=["回顾问题?"])
    await teaching.start(subject_id)

    script(harness, blocking=[verdict_json("fail")], stream=["再讲一遍上一节。"])
    turn = await teaching.answer(subject_id, "答错了")

    assert turn.ok
    assert "回到" in turn.text
    snapshot = harness.state.snapshot(subject_id)
    assert snapshot.current_node_id == previous.id  # 退回上一节点
    assert snapshot.reviewing is False
    assert harness.repos.weak_points.count_open(subject_id) == 1


async def test_adjacent_review_is_not_asked_twice(context: AppContext) -> None:
    """通过回顾后刷新再点「继续学习」，不该再问一遍同一道回顾题。"""
    harness = make_harness(context)
    _, _, teaching, subject_id = enter_teaching(harness, node_count=2)
    await pass_current_node(harness, teaching, subject_id)
    script(harness, stream=["回顾问题?"])
    await teaching.start(subject_id)
    script(harness, blocking=[verdict_json("pass")], stream=["节点 1 的讲解。"])
    await teaching.answer(subject_id, "回顾回答")

    # 模拟刷新后重新进入教学页
    script(harness, stream=["不该再讲一遍"])
    turn = await teaching.start(subject_id)

    assert turn.phase == "question"
    assert "回顾" not in turn.text
    assert harness.client.stream_calls == 0  # 讲解已经生成过，不重复生成


# --------------------------------------------------------------------------- #
# 档案、历史、笔记与完成报告（FR-801、FR-802、FR-804、FR-417）
async def pass_current_node(harness: LearningHarness, teaching, subject_id: str):
    """让当前节点一次通过，返回该节点。"""
    script(harness, stream=["讲解。"])
    await teaching.start(subject_id)
    node_id = harness.state.snapshot(subject_id).current_node_id
    assert node_id is not None
    script(harness, blocking=[verdict_json("pass")], stream=["小结。"])
    await teaching.answer(subject_id, "回答")
    return harness.repos.paths.node(node_id)
