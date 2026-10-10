"""领域仓储测试（T2.1）。

覆盖 FR-101~FR-106、FR-301~FR-309、FR-404、FR-410、FR-502、FR-801 的持久化行为。
"""

import pytest

from app.contracts.schemas import (
    AssessmentStatus,
    MasteryLevel,
    NodeState,
    SubjectState,
    WeakPointReason,
)
from app.main_context import AppContext
from app.services.learning.entities import ProfileLevel
from app.services.learning.repository import DomainRepositories


@pytest.fixture
def repos(context: AppContext) -> DomainRepositories:
    return DomainRepositories(context.db)


def seed_path(repos: DomainRepositories, subject_id: str, *, node_count: int = 3) -> str:
    nodes = [
        {
            "title": f"节点 {index}",
            "objective": f"目标 {index}",
            "prerequisites": [f"节点 {index - 1}"] if index else [],
            "key_points": ["要点一", "要点二", "要点三"],
            "checkpoints": [
                {
                    "question": f"问题 {index}",
                    "expected_points": ["判定要点 A", "判定要点 B"],
                    "reference_answer": f"参考答案 {index}",
                }
            ],
        }
        for index in range(node_count)
    ]
    path = repos.paths.create(subject_id, rationale="因为画像显示基础较弱", model="deepseek-flash", nodes=nodes)
    return path.id


# --------------------------------------------------------------------------- #
# 主题
# --------------------------------------------------------------------------- #


def test_subject_create_and_get(repos: DomainRepositories) -> None:
    created = repos.subjects.create("贝叶斯定理", goal="通过面试", background="学过概率论", depth="深入")

    fetched = repos.subjects.get(created.id)
    assert fetched is not None
    assert fetched.title == "贝叶斯定理"
    assert fetched.goal == "通过面试"
    assert fetched.status is SubjectState.DRAFT
    assert fetched.archived is False


def test_subject_update_and_archive(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("旧标题")

    updated = repos.subjects.update(subject.id, title="新标题", goal="新目标")
    assert updated is not None
    assert updated.title == "新标题"
    assert updated.goal == "新目标"

    assert repos.subjects.archive(subject.id) is True
    assert repos.subjects.list_all() == []
    assert len(repos.subjects.list_all(include_archived=True)) == 1

    assert repos.subjects.archive(subject.id, archived=False) is True
    assert len(repos.subjects.list_all()) == 1


def test_subject_status_transition_is_persisted(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("主题")

    repos.subjects.set_status(subject.id, SubjectState.TEACHING)

    assert repos.subjects.get(subject.id).status is SubjectState.TEACHING  # type: ignore[union-attr]


def test_subject_delete_cascades_everything(repos: DomainRepositories, context: AppContext) -> None:
    """FR-104：删除主题必须清掉路径、节点、检查点、测评、画像、记忆、作答、薄弱点、笔记、用量。"""
    from app.contracts.schemas import ConversationKind
    from app.infra.storage.models import (
        AnswerRecordRow,
        AssessmentLevelRow,
        ChatRoundRow,
        CheckpointRow,
        ConversationRow,
        KnowledgePointRow,
        LearningPathRow,
        MessageRow,
        StudyNoteRow,
        SubjectRow,
        UsageRecordRow,
        WeakPointRow,
    )
    from app.services.llm.memory import Memory, assistant_message, user_message

    subject = repos.subjects.create("待删除")
    sid = subject.id

    path_id = seed_path(repos, sid, node_count=2)
    node = repos.paths.get(path_id).ordered_nodes()[0]  # type: ignore[union-attr]
    checkpoint = node.checkpoints[0]

    repos.sessions.create(sid)
    assessment = repos.assessments.create(sid, max_turns=6, topics=["基础"])
    repos.assessments.save_levels(
        assessment.id, [ProfileLevel(topic="基础", level=MasteryLevel.BASIC, evidence="答对一半")]
    )
    repos.answers.add(
        subject_id=sid,
        node_id=node.id,
        checkpoint_id=checkpoint.id,
        question=checkpoint.question,
        user_answer="我的回答",
        verdict="partial",
        confidence=0.4,
        feedback="还差一点",
    )
    repos.weak_points.record(sid, node.id, WeakPointReason.PARTIAL)
    repos.notes.upsert(sid, node.id, "笔记内容")

    conversation_id = context.memories.ensure_conversation(ConversationKind.TEACH, subject_id=sid)
    memory = Memory()
    memory.add_round().add(user_message("问题"))
    memory.add_message_to_last_round(assistant_message("回答"))
    context.memories.save(conversation_id, memory)

    assert repos.subjects.delete(sid) is True

    with context.db.session() as session:
        assert session.get(SubjectRow, sid) is None
        for row_type in (
            LearningPathRow,
            KnowledgePointRow,
            CheckpointRow,
            AssessmentLevelRow,
            ConversationRow,
            ChatRoundRow,
            MessageRow,
            AnswerRecordRow,
            WeakPointRow,
            StudyNoteRow,
            UsageRecordRow,
        ):
            assert session.query(row_type).count() == 0, row_type.__name__

    assert repos.subjects.delete(sid) is False


def test_progress_and_summary(repos: DomainRepositories, context: AppContext) -> None:
    from app.contracts.schemas import LlmPurpose
    from app.infra.deepseek.protocol import TokenUsage
    from app.services.llm.usage import UsageRecorder

    subject = repos.subjects.create("有进度的主题")
    path_id = seed_path(repos, subject.id, node_count=4)
    nodes = repos.paths.get(path_id).ordered_nodes()  # type: ignore[union-attr]

    repos.paths.update_node(nodes[0].id, state=NodeState.MASTERED)
    repos.paths.update_node(nodes[1].id, state=NodeState.PARTIAL)
    repos.paths.update_node(nodes[2].id, state=NodeState.SKIPPED)

    repos.weak_points.record(subject.id, nodes[1].id, WeakPointReason.ADVANCED)
    UsageRecorder(context.db, subject_id=subject.id).record(
        LlmPurpose.TEACH, "deepseek-flash", TokenUsage(total_tokens=1234)
    )

    progress = repos.subjects.progress(subject.id)
    assert progress.total_nodes == 4
    assert progress.mastered_nodes == 1
    assert progress.partial_nodes == 1
    assert progress.skipped_nodes == 1
    assert progress.weak_points == 1
    assert progress.total_tokens == 1234
    assert progress.mastery_ratio == pytest.approx(0.25)

    summary = repos.subjects.summaries()[0]
    assert summary.subject.id == subject.id
    assert summary.last_studied_at is not None


def test_progress_of_empty_subject(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("还没开始")

    progress = repos.subjects.progress(subject.id)
    assert progress.total_nodes == 0
    assert progress.mastery_ratio == 0.0


# --------------------------------------------------------------------------- #
# 状态机
# --------------------------------------------------------------------------- #


def test_session_state_ensure_is_idempotent(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")

    first = repos.sessions.ensure(subject.id)
    second = repos.sessions.ensure(subject.id)

    assert first.state is SubjectState.DRAFT
    assert first == second


def test_session_pending_question_semantics(repos: DomainRepositories) -> None:
    """FR-410 的核心不变量：只有问题与检查点都存在时才算"待回答"。"""
    subject = repos.subjects.create("s")
    repos.sessions.ensure(subject.id)

    snapshot = repos.sessions.update(
        subject.id,
        pending_checkpoint_id="cp-1",
        pending_question="请解释贝叶斯公式。",
    )
    assert snapshot.has_pending_question is True

    only_id = repos.sessions.update(subject.id, pending_checkpoint_id="cp-2", pending_question=None)
    assert only_id.has_pending_question is False

    cleared = repos.sessions.clear_pending(subject.id)
    assert cleared.has_pending_question is False
    assert cleared.pending_checkpoint_id is None


def test_session_taught_marker(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    repos.sessions.ensure(subject.id)

    repos.sessions.update(subject.id, current_node_id="n1")
    assert repos.sessions.get(subject.id).node_taught is False  # type: ignore[union-attr]

    snapshot = repos.sessions.update(subject.id, taught_node_id="n1")
    assert snapshot.node_taught is True

    # 切到别的节点后不再是"已讲解"
    assert repos.sessions.update(subject.id, current_node_id="n2").node_taught is False


def test_session_update_rejects_unknown_field(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    repos.sessions.ensure(subject.id)

    with pytest.raises(ValueError, match="不支持更新的字段"):
        repos.sessions.update(subject.id, nonexistent=1)


def test_session_update_missing_subject_raises(repos: DomainRepositories) -> None:
    with pytest.raises(KeyError):
        repos.sessions.update("不存在", node_attempts=1)


# --------------------------------------------------------------------------- #
# 测评与画像
# --------------------------------------------------------------------------- #


def test_assessment_lifecycle(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")

    assessment = repos.assessments.create(subject.id, max_turns=3, topics=["基础", "进阶"])
    assert assessment.turn_count == 0
    assert assessment.turns_left == 3
    assert repos.assessments.get_running(subject.id).id == assessment.id  # type: ignore[union-attr]

    assert repos.assessments.increment_turn(assessment.id) == 1
    assert repos.assessments.increment_turn(assessment.id, topics=["基础", "进阶", "应用"]) == 2
    assert repos.assessments.get(assessment.id).turns_left == 1  # type: ignore[union-attr]

    finished = repos.assessments.finish(
        assessment.id, summary="基础尚可", suggested_start_node="从条件概率开始"
    )
    assert finished is not None
    assert finished.status is AssessmentStatus.COMPLETED
    assert finished.finished is True
    assert finished.completed_at is not None
    assert repos.assessments.get_running(subject.id) is None


def test_assessment_skip(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    assessment = repos.assessments.create(subject.id, max_turns=6)

    skipped = repos.assessments.finish(assessment.id, skipped=True)

    assert skipped is not None
    assert skipped.status is AssessmentStatus.SKIPPED
    assert skipped.skipped is True


def test_assessment_topics_are_persisted_as_json(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    assessment = repos.assessments.create(subject.id, max_turns=6, topics=["先验", "似然"])

    repos.assessments.set_topics(assessment.id, ["先验", "似然", "后验"])

    reloaded = repos.assessments.get(assessment.id)
    assert reloaded is not None
    assert reloaded.topics == ("先验", "似然", "后验")


def test_assessment_levels_and_override(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    assessment = repos.assessments.create(subject.id, max_turns=6)

    repos.assessments.save_levels(
        assessment.id,
        [
            ProfileLevel(topic="条件概率", level=MasteryLevel.PROFICIENT, evidence="解释准确"),
            ProfileLevel(topic="贝叶斯更新", level=MasteryLevel.NONE, evidence="未作答"),
        ],
    )

    levels = repos.assessments.levels(assessment.id)
    assert [level.topic for level in levels] == ["条件概率", "贝叶斯更新"]
    assert levels[0].level is MasteryLevel.PROFICIENT
    assert levels[0].user_overridden is False

    assert repos.assessments.override_level(assessment.id, "条件概率", MasteryLevel.BASIC) is True
    assert repos.assessments.override_level(assessment.id, "不存在的子领域", MasteryLevel.BASIC) is False

    overridden = repos.assessments.levels(assessment.id)[0]
    assert overridden.level is MasteryLevel.BASIC
    assert overridden.user_overridden is True


def test_save_levels_replaces_previous(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    assessment = repos.assessments.create(subject.id, max_turns=6)

    repos.assessments.save_levels(
        assessment.id, [ProfileLevel(topic="旧", level=MasteryLevel.NONE, evidence="e")]
    )
    repos.assessments.save_levels(
        assessment.id, [ProfileLevel(topic="新", level=MasteryLevel.BASIC, evidence="e2")]
    )

    levels = repos.assessments.levels(assessment.id)
    assert len(levels) == 1
    assert levels[0].topic == "新"


# --------------------------------------------------------------------------- #
# 路径与节点
# --------------------------------------------------------------------------- #


def test_path_creation_persists_nodes_and_checkpoints(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    path_id = seed_path(repos, subject.id, node_count=3)

    path = repos.paths.get(path_id)
    assert path is not None
    assert path.node_count == 3
    assert path.is_active is True
    assert path.rationale == "因为画像显示基础较弱"

    first = path.ordered_nodes()[0]
    assert first.order_index == 0
    assert first.key_points == ("要点一", "要点二", "要点三")
    assert first.state is NodeState.NOT_STARTED
    assert len(first.checkpoints) == 1

    checkpoint = first.checkpoints[0]
    assert checkpoint.question == "问题 0"
    assert checkpoint.expected_points == ("判定要点 A", "判定要点 B")
    assert checkpoint.reference_answer == "参考答案 0"


def test_new_path_deactivates_previous_and_bumps_version(repos: DomainRepositories) -> None:
    """FR-307：重新生成保留旧版本，但只有一条 active。"""
    subject = repos.subjects.create("s")
    first_id = seed_path(repos, subject.id, node_count=2)
    second_id = seed_path(repos, subject.id, node_count=4)

    versions = repos.paths.list_versions(subject.id)
    assert len(versions) == 2
    assert versions[0][1] == 2  # 版本号递增
    assert versions[0][2] is True

    assert repos.paths.get(first_id).is_active is False  # type: ignore[union-attr]
    assert repos.paths.get_active(subject.id).id == second_id  # type: ignore[union-attr]


def test_activate_switches_active_path(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    first_id = seed_path(repos, subject.id, node_count=2)
    seed_path(repos, subject.id, node_count=2)

    assert repos.paths.activate(first_id) is True
    assert repos.paths.get_active(subject.id).id == first_id  # type: ignore[union-attr]
    assert repos.paths.activate("不存在") is False


def test_next_node_walks_to_the_end(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    path_id = seed_path(repos, subject.id, node_count=3)

    first = repos.paths.first_node(path_id)
    assert first is not None
    assert first.order_index == 0

    second = repos.paths.next_node(path_id, after_order_index=first.order_index)
    assert second is not None
    assert second.order_index == 1

    third = repos.paths.next_node(path_id, after_order_index=second.order_index)
    assert third is not None
    assert third.order_index == 2

    # 已到末尾 → None，用于判定"学习完成"
    assert repos.paths.next_node(path_id, after_order_index=third.order_index) is None


def test_update_node_state_and_attempts(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    path_id = seed_path(repos, subject.id, node_count=1)
    node = repos.paths.first_node(path_id)
    assert node is not None

    assert repos.paths.increment_attempts(node.id) == 1
    assert repos.paths.increment_attempts(node.id) == 2

    updated = repos.paths.update_node(node.id, state=NodeState.MASTERED, summary_note="小结")
    assert updated is not None
    assert updated.state is NodeState.MASTERED
    assert updated.attempts == 2
    assert updated.summary_note == "小结"

    with pytest.raises(ValueError, match="不支持更新的字段"):
        repos.paths.update_node(node.id, title="不允许改标题")


def test_update_node_increments_missing_node_raises(repos: DomainRepositories) -> None:
    with pytest.raises(KeyError):
        repos.paths.increment_attempts("不存在")


def test_checkpoint_navigation(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    nodes = [
        {
            "title": "多检查点节点",
            "objective": "目标",
            "key_points": ["a", "b", "c"],
            "checkpoints": [
                {"question": f"问题 {i}", "expected_points": ["p"], "reference_answer": f"答案 {i}"}
                for i in range(3)
            ],
        }
    ]
    path = repos.paths.create(subject.id, rationale="r", model=None, nodes=nodes)
    node = path.ordered_nodes()[0]
    assert len(node.checkpoints) == 3

    first = repos.paths.first_checkpoint(node.id)
    assert first is not None
    assert first.question == "问题 0"

    second = repos.paths.next_checkpoint(node.id, first.order_index)
    assert second is not None
    assert second.question == "问题 1"

    third = repos.paths.next_checkpoint(node.id, second.order_index)
    assert third is not None
    assert repos.paths.next_checkpoint(node.id, third.order_index) is None

    assert repos.paths.checkpoint(first.id).question == "问题 0"  # type: ignore[union-attr]
    assert repos.paths.checkpoint("不存在") is None


def test_path_tolerates_malformed_node_payload(repos: DomainRepositories) -> None:
    """模型输出的字段类型未必守规，这里固化收敛规则：

    - 标量字符串按**单元素列表**处理（内容有效，丢弃会静默损失要点）；
    - ``None`` 与非列表（如字符串当列表用）按空处理；
    - ``checkpoints`` 不是列表时按"无检查点"处理，绝不抛异常把整条路径生成打回。
    """
    subject = repos.subjects.create("s")
    path = repos.paths.create(
        subject.id,
        rationale="r",
        model=None,
        nodes=[
            {
                "title": "节点",
                "objective": "目标",
                "prerequisites": "条件概率",
                "key_points": None,
                "checkpoints": "不是列表",
            }
        ],
    )

    node = path.ordered_nodes()[0]
    assert node.prerequisites == ("条件概率",)
    assert node.key_points == ()
    assert node.checkpoints == ()


def test_path_tolerates_non_string_items(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    path = repos.paths.create(
        subject.id,
        rationale="r",
        model=None,
        nodes=[
            {
                "title": "节点",
                "objective": "目标",
                "key_points": ["要点一", None, "", 42],
                "checkpoints": [{"question": "q", "expected_points": ["p"], "reference_answer": "a"}, "坏检查点"],
            }
        ],
    )

    node = path.ordered_nodes()[0]
    # None 与空串被过滤，数字被转成字符串
    assert node.key_points == ("要点一", "42")
    # 非字典的检查点被丢弃，合法的保留
    assert len(node.checkpoints) == 1


# --------------------------------------------------------------------------- #
# 作答、薄弱点、笔记
# --------------------------------------------------------------------------- #


def test_answer_records(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")
    path_id = seed_path(repos, subject.id, node_count=1)
    node = repos.paths.first_node(path_id)
    assert node is not None
    checkpoint = node.checkpoints[0]

    repos.answers.add(
        subject_id=subject.id,
        node_id=node.id,
        checkpoint_id=checkpoint.id,
        question=checkpoint.question,
        user_answer="第一次回答",
        verdict="fail",
        confidence=0.2,
        feedback="没答到点上",
        missing_points=["判定要点 A", "判定要点 B"],
        attempt_index=1,
    )
    repos.answers.add(
        subject_id=subject.id,
        node_id=node.id,
        checkpoint_id=checkpoint.id,
        question=checkpoint.question,
        user_answer="第二次回答",
        verdict="pass",
        confidence=0.9,
        feedback="很好",
        hit_points=["判定要点 A"],
        attempt_index=2,
    )

    records = repos.answers.list_for_node(node.id)
    assert len(records) == 2
    assert records[0].verdict == "fail"
    assert records[0].missing_points == ("判定要点 A", "判定要点 B")
    assert records[1].hit_points == ("判定要点 A",)
    assert repos.answers.count_for_node(node.id) == 2


def test_weak_point_accumulates_and_resolves(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")

    first = repos.weak_points.record(subject.id, "n1", WeakPointReason.PARTIAL)
    assert first.fail_count == 1
    assert first.resolved is False

    second = repos.weak_points.record(subject.id, "n1", WeakPointReason.ADVANCED)
    assert second.id == first.id  # 同一节点复用同一条记录
    assert second.fail_count == 2
    assert second.reason is WeakPointReason.ADVANCED

    repos.weak_points.record(subject.id, "n2", WeakPointReason.SKIPPED)
    assert repos.weak_points.count_open(subject.id) == 2

    assert repos.weak_points.resolve(subject.id, "n1") is True
    assert repos.weak_points.resolve(subject.id, "n1") is False

    open_points = repos.weak_points.list_open(subject.id)
    assert [point.node_id for point in open_points] == ["n2"]
    assert repos.weak_points.count_open(subject.id) == 1


def test_note_upsert_overwrites(repos: DomainRepositories) -> None:
    subject = repos.subjects.create("s")

    created = repos.notes.upsert(subject.id, "n1", "第一版笔记")
    assert created.content == "第一版笔记"
    assert created.source == "auto"

    updated = repos.notes.upsert(subject.id, "n1", "第二版笔记", source="user")
    assert updated.id == created.id
    assert updated.content == "第二版笔记"
    assert updated.source == "user"

    assert repos.notes.get(subject.id, "n1").content == "第二版笔记"  # type: ignore[union-attr]
    assert repos.notes.get(subject.id, "不存在") is None
    assert len(repos.notes.list_for_subject(subject.id)) == 1


def test_note_without_node_is_supported(repos: DomainRepositories) -> None:
    """主题级笔记（node_id 为空）也要能存。"""
    subject = repos.subjects.create("s")

    repos.notes.upsert(subject.id, None, "整个主题的总结")

    note = repos.notes.get(subject.id, None)
    assert note is not None
    assert note.content == "整个主题的总结"
