"""会话状态机测试（T2.2，FR-410）。

重点验证三类守卫，它们各自对应一个真实的错误后果：

- **非法迁移**被拒绝 → 不会出现"从 draft 直接进 teaching"的脏状态；
- **没有待回答问题**时拒绝作答 → 用户随手一句话不会被当成检查点答案（污染作答记录）；
- **正在等三选一**时拒绝作答 → 用户不会被"重新判定"顶掉选项（FR-408 被绕过）。
"""

import pytest

from app.contracts.schemas import SubjectState
from app.main_context import AppContext
from app.services.learning.repository import DomainRepositories
from app.services.learning.state import (
    ALLOWED_TRANSITIONS,
    AnswerNotAllowedError,
    SessionStateService,
    TransitionError,
)


@pytest.fixture
def services(context: AppContext) -> tuple[SessionStateService, DomainRepositories]:
    repos = DomainRepositories(context.db)
    return SessionStateService(repos.sessions, repos.subjects), repos


def make_subject(repos: DomainRepositories, title: str = "贝叶斯定理") -> str:
    return repos.subjects.create(title).id


def make_node(repos: DomainRepositories, subject_id: str):
    path = repos.paths.create(
        subject_id,
        rationale="r",
        model=None,
        nodes=[
            {
                "title": "条件概率",
                "objective": "能解释条件概率",
                "key_points": ["a", "b", "c"],
                "checkpoints": [
                    {"question": "什么是条件概率？", "expected_points": ["p"], "reference_answer": "a"}
                ],
            }
        ],
    )
    node = path.ordered_nodes()[0]
    assert node is not None
    return node


# --------------------------------------------------------------------------- #
# 迁移表
# --------------------------------------------------------------------------- #


def test_new_subject_starts_in_draft(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)

    snapshot = state.snapshot(subject_id)

    assert snapshot.state is SubjectState.DRAFT
    assert snapshot.has_pending_question is False
    assert snapshot.needs_user_decision is False


def test_happy_path_transitions(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)

    assert state.transition(subject_id, SubjectState.ASSESSING).state is SubjectState.ASSESSING
    assert state.transition(subject_id, SubjectState.PATH_REVIEW).state is SubjectState.PATH_REVIEW
    assert state.transition(subject_id, SubjectState.TEACHING).state is SubjectState.TEACHING
    assert state.transition(subject_id, SubjectState.PAUSED).state is SubjectState.PAUSED
    assert state.transition(subject_id, SubjectState.TEACHING).state is SubjectState.TEACHING
    assert state.transition(subject_id, SubjectState.COMPLETED).state is SubjectState.COMPLETED
    # 完成后可以重新开始学习（复习）
    assert state.transition(subject_id, SubjectState.TEACHING).state is SubjectState.TEACHING


def test_skipping_assessment_goes_draft_to_path_review(services) -> None:
    """FR-205：跳过测评时不应被迫先进 assessing。"""
    state, repos = services
    subject_id = make_subject(repos)

    assert state.transition(subject_id, SubjectState.PATH_REVIEW).state is SubjectState.PATH_REVIEW


def test_illegal_transitions_are_rejected(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)

    with pytest.raises(TransitionError) as exc:
        state.transition(subject_id, SubjectState.TEACHING)
    assert "draft" in str(exc.value)
    assert "teaching" in str(exc.value)

    # 状态未被改动
    assert state.snapshot(subject_id).state is SubjectState.DRAFT
    assert repos.subjects.get(subject_id).status is SubjectState.DRAFT  # type: ignore[union-attr]


def test_completed_cannot_go_back_to_assessing(services) -> None:
    """完成态只能继续学习，不能回到测评/路径确认——避免绕过已确认的路径。"""
    state, repos = services
    subject_id = make_subject(repos)
    state.transition(subject_id, SubjectState.PATH_REVIEW)
    state.transition(subject_id, SubjectState.TEACHING)
    state.transition(subject_id, SubjectState.COMPLETED)

    with pytest.raises(TransitionError):
        state.transition(subject_id, SubjectState.ASSESSING)
    with pytest.raises(TransitionError):
        state.transition(subject_id, SubjectState.PATH_REVIEW)


def test_same_state_transition_is_idempotent(services) -> None:
    """"重新生成路径"会重复进入 path_review，这不该报错。"""
    state, repos = services
    subject_id = make_subject(repos)
    state.transition(subject_id, SubjectState.PATH_REVIEW)

    again = state.transition(subject_id, SubjectState.PATH_REVIEW)

    assert again.state is SubjectState.PATH_REVIEW


def test_can_transition_matches_table(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)

    assert state.can_transition(subject_id, SubjectState.ASSESSING) is True
    assert state.can_transition(subject_id, SubjectState.TEACHING) is False
    assert state.can_transition(subject_id, SubjectState.DRAFT) is True  # 自身


def test_transition_syncs_subject_status(services) -> None:
    """列表页读的是 subject.status，必须与状态机同步。"""
    state, repos = services
    subject_id = make_subject(repos)

    state.transition(subject_id, SubjectState.ASSESSING)

    assert repos.subjects.get(subject_id).status is SubjectState.ASSESSING  # type: ignore[union-attr]
    assert repos.subjects.summaries()[0].subject.status is SubjectState.ASSESSING


def test_transition_table_covers_every_state() -> None:
    """每个状态都要在迁移表里有条目，否则新状态会成为死路。"""
    assert set(ALLOWED_TRANSITIONS) == set(SubjectState)


def test_unknown_subject_raises(services) -> None:
    state, _ = services

    with pytest.raises(KeyError):
        state.snapshot("不存在的主题")


# --------------------------------------------------------------------------- #
# 节点与待回答问题
# --------------------------------------------------------------------------- #


def test_enter_node_resets_attempts_and_pending(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)
    node = make_node(repos, subject_id)
    state.transition(subject_id, SubjectState.PATH_REVIEW)
    state.transition(subject_id, SubjectState.TEACHING)

    state.set_pending_question(subject_id, checkpoint_id="cp", question="q")
    state.increment_attempts(subject_id)
    state.mark_taught(subject_id)

    snapshot = state.enter_node(subject_id, node)

    assert snapshot.current_node_id == node.id
    assert snapshot.node_attempts == 0
    assert snapshot.has_pending_question is False
    assert snapshot.needs_user_decision is False
    assert snapshot.taught_node_id is None
    assert snapshot.node_taught is False


def test_mark_taught_tracks_current_node(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)
    node = make_node(repos, subject_id)
    state.transition(subject_id, SubjectState.PATH_REVIEW)
    state.transition(subject_id, SubjectState.TEACHING)
    state.enter_node(subject_id, node)

    assert state.snapshot(subject_id).node_taught is False

    snapshot = state.mark_taught(subject_id)

    assert snapshot.taught_node_id == node.id
    assert snapshot.node_taught is True


def test_pending_question_round_trip(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)
    state.transition(subject_id, SubjectState.PATH_REVIEW)
    state.transition(subject_id, SubjectState.TEACHING)

    state.set_pending_question(subject_id, checkpoint_id="cp-1", question="请解释贝叶斯公式。")
    snapshot, checkpoint_id, question = state.require_pending_question(subject_id)

    assert checkpoint_id == "cp-1"
    assert question == "请解释贝叶斯公式。"
    assert snapshot.has_pending_question is True

    state.clear_pending(subject_id)
    assert state.snapshot(subject_id).has_pending_question is False


def test_increment_attempts_returns_new_value(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)

    assert state.increment_attempts(subject_id) == 1
    assert state.increment_attempts(subject_id) == 2
    assert state.snapshot(subject_id).node_attempts == 2


# --------------------------------------------------------------------------- #
# 守卫
# --------------------------------------------------------------------------- #


def test_answer_rejected_without_pending_question(services) -> None:
    """核心不变量：没有正在等待的问题时，任何输入都不能被当成答案。"""
    state, repos = services
    subject_id = make_subject(repos)
    state.transition(subject_id, SubjectState.PATH_REVIEW)
    state.transition(subject_id, SubjectState.TEACHING)

    with pytest.raises(AnswerNotAllowedError) as exc:
        state.require_pending_question(subject_id)

    assert exc.value.reason == "no_pending_question"
    assert "没有等待回答的问题" in str(exc.value)


def test_answer_rejected_while_awaiting_decision(services) -> None:
    """FR-408：重试耗尽后必须先做选择，不能让用户用一次新作答把选项顶掉。"""
    state, repos = services
    subject_id = make_subject(repos)
    state.transition(subject_id, SubjectState.PATH_REVIEW)
    state.transition(subject_id, SubjectState.TEACHING)
    state.set_pending_question(subject_id, checkpoint_id="cp", question="q")

    state.request_decision(subject_id)
    snapshot = state.snapshot(subject_id)

    assert snapshot.needs_user_decision is True
    assert snapshot.has_pending_question is False  # 选项与作答互斥

    with pytest.raises(AnswerNotAllowedError) as exc:
        state.require_pending_question(subject_id)
    assert exc.value.reason == "awaiting_decision"
    assert "继续重讲" in str(exc.value)


def test_answer_rejected_when_paused(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)
    state.transition(subject_id, SubjectState.PATH_REVIEW)
    state.transition(subject_id, SubjectState.TEACHING)
    state.set_pending_question(subject_id, checkpoint_id="cp", question="q")
    state.transition(subject_id, SubjectState.PAUSED)

    with pytest.raises(AnswerNotAllowedError) as exc:
        state.require_pending_question(subject_id)
    assert exc.value.reason == "paused"
    assert "继续学习" in str(exc.value)


def test_answer_rejected_when_completed(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)
    state.transition(subject_id, SubjectState.PATH_REVIEW)
    state.transition(subject_id, SubjectState.TEACHING)
    state.transition(subject_id, SubjectState.COMPLETED)

    with pytest.raises(AnswerNotAllowedError) as exc:
        state.require_pending_question(subject_id)
    assert exc.value.reason == "completed"


def test_answer_rejected_in_early_stages(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)

    with pytest.raises(AnswerNotAllowedError) as exc:
        state.require_answering_state(subject_id)
    assert exc.value.reason == "wrong_stage"
    assert "创建" in str(exc.value)  # 阶段名用中文，提示可读


def test_decision_guard_requires_pending_decision(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)
    state.transition(subject_id, SubjectState.PATH_REVIEW)
    state.transition(subject_id, SubjectState.TEACHING)

    with pytest.raises(AnswerNotAllowedError) as exc:
        state.require_decision_pending(subject_id)
    assert exc.value.reason == "decision_not_expected"

    state.request_decision(subject_id)
    assert state.require_decision_pending(subject_id).needs_user_decision is True

    state.clear_decision(subject_id)
    assert state.snapshot(subject_id).needs_user_decision is False


# --------------------------------------------------------------------------- #
# 恢复视图
# --------------------------------------------------------------------------- #


def test_next_action_for_each_stage(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)
    loader = lambda node_id: repos.paths.node(node_id)  # noqa: E731

    assert state.resume(subject_id, node_loader=loader).next_action == "assess"

    state.transition(subject_id, SubjectState.ASSESSING)
    assert state.resume(subject_id, node_loader=loader).next_action == "assess"

    state.transition(subject_id, SubjectState.PATH_REVIEW)
    assert state.resume(subject_id, node_loader=loader).next_action == "confirm_path"

    state.transition(subject_id, SubjectState.TEACHING)
    # 已进入教学但还没讲解
    assert state.resume(subject_id, node_loader=loader).next_action == "teach"

    node = make_node(repos, subject_id)
    state.enter_node(subject_id, node)
    state.set_pending_question(subject_id, checkpoint_id=node.checkpoints[0].id, question="q")
    assert state.resume(subject_id, node_loader=loader).next_action == "answer"

    state.request_decision(subject_id)
    view = state.resume(subject_id, node_loader=loader)
    assert view.next_action == "decide"
    assert view.needs_decision is True
    assert view.can_answer is False

    state.transition(subject_id, SubjectState.PAUSED)
    assert state.resume(subject_id, node_loader=loader).next_action == "paused"

    state.transition(subject_id, SubjectState.TEACHING)
    state.transition(subject_id, SubjectState.COMPLETED)
    assert state.resume(subject_id, node_loader=loader).next_action == "completed"


def test_resume_view_exposes_current_node(services) -> None:
    """FR-410：恢复后要能拿到当前节点（含检查点），前端才能重建页面。"""
    state, repos = services
    subject_id = make_subject(repos)
    node = make_node(repos, subject_id)
    state.transition(subject_id, SubjectState.PATH_REVIEW)
    state.transition(subject_id, SubjectState.TEACHING)
    state.enter_node(subject_id, node)
    state.set_pending_question(
        subject_id, checkpoint_id=node.checkpoints[0].id, question=node.checkpoints[0].question
    )

    view = state.resume(subject_id, node_loader=repos.paths.node)

    assert view.node is not None
    assert view.node.title == "条件概率"
    assert view.node.checkpoints[0].question == "什么是条件概率？"
    assert view.can_answer is True
    assert view.state is SubjectState.TEACHING
    assert view.subject.id == subject_id


def test_resume_view_without_loader_returns_no_node(services) -> None:
    state, repos = services
    subject_id = make_subject(repos)

    view = state.resume(subject_id)

    assert view.node is None
    assert view.next_action == "assess"
