"""主题服务测试（p2-subjects，FR-101~FR-107）。"""

import pytest

from app.contracts.schemas import NodeState, SubjectState
from app.main_context import AppContext
from app.services.learning.repository import DomainRepositories
from app.services.learning.subjects import (
    TITLE_MAX_LENGTH,
    SubjectService,
    SubjectValidationError,
    normalize_title,
)


@pytest.fixture
def service(context: AppContext) -> tuple[SubjectService, DomainRepositories]:
    repos = DomainRepositories(context.db)
    return SubjectService(repos), repos


# --------------------------------------------------------------------------- #
# 名称校验
# --------------------------------------------------------------------------- #


def test_normalize_title_trims_and_keeps() -> None:
    assert normalize_title("  贝叶斯定理  ") == "贝叶斯定理"
    assert normalize_title("K8s") == "K8s"


@pytest.mark.parametrize(
    "raw",
    ["", "   ", None, "a", "  x  "],
)
def test_normalize_title_rejects_too_short(raw: str | None) -> None:
    with pytest.raises(SubjectValidationError) as exc:
        normalize_title(raw)
    assert "概念名称" in str(exc.value)


def test_normalize_title_rejects_too_long() -> None:
    with pytest.raises(SubjectValidationError) as exc:
        normalize_title("概" * (TITLE_MAX_LENGTH + 1))
    assert "最多" in str(exc.value)


@pytest.mark.parametrize("raw", ["!!!", "12345", "。。。", "...", "---"])
def test_normalize_title_rejects_meaningless_input(raw: str) -> None:
    """FR-102：疑似无意义输入给出明确提示（纯符号/纯数字不做联网校验也能识别）。"""
    with pytest.raises(SubjectValidationError) as exc:
        normalize_title(raw)
    assert "不像是可以学习的内容" in str(exc.value)


def test_normalize_title_accepts_mixed_content() -> None:
    assert normalize_title("TCP 三次握手") == "TCP 三次握手"
    assert normalize_title("2026 版 Kubernetes") == "2026 版 Kubernetes"


# --------------------------------------------------------------------------- #
# 创建
# --------------------------------------------------------------------------- #


def test_create_initialises_session_state(service) -> None:
    subjects, repos = service

    created = subjects.create("贝叶斯定理", goal="通过面试", background="学过概率论", depth="深入")

    assert created.title == "贝叶斯定理"
    assert created.goal == "通过面试"
    assert created.status is SubjectState.DRAFT

    snapshot = repos.sessions.get(created.id)
    assert snapshot is not None
    assert snapshot.state is SubjectState.DRAFT


def test_create_trims_and_normalises_optional_fields(service) -> None:
    subjects, _ = service

    created = subjects.create("  条件概率  ", goal="  ", background=None, depth="  基础  ")

    assert created.title == "条件概率"
    assert created.goal is None
    assert created.background is None
    assert created.depth == "基础"


def test_create_rejects_overlong_goal(service) -> None:
    subjects, _ = service
    with pytest.raises(SubjectValidationError) as exc:
        subjects.create("有效概念", goal="x" * 501)
    assert "学习目标" in str(exc.value)


def test_create_rejects_invalid_title(service) -> None:
    subjects, _ = service
    with pytest.raises(SubjectValidationError):
        subjects.create("!!!")


# --------------------------------------------------------------------------- #
# 修改
# --------------------------------------------------------------------------- #


def test_update_rename_and_goal(service) -> None:
    subjects, _ = service
    created = subjects.create("旧名", goal="旧目标")

    updated = subjects.update(created.id, title="新名", goal="新目标")

    assert updated.title == "新名"
    assert updated.goal == "新目标"
    assert updated.id == created.id


def test_update_partial_fields_only(service) -> None:
    subjects, _ = service
    created = subjects.create("概念", goal="目标", background="背景")

    updated = subjects.update(created.id, goal="改过的目标")

    assert updated.title == "概念"
    assert updated.background == "背景"
    assert updated.goal == "改过的目标"


def test_update_unknown_subject_raises(service) -> None:
    subjects, _ = service
    with pytest.raises(KeyError):
        subjects.update("不存在", title="新名")


def test_update_validates_title(service) -> None:
    subjects, _ = service
    created = subjects.create("概念")
    with pytest.raises(SubjectValidationError):
        subjects.update(created.id, title="!")


# --------------------------------------------------------------------------- #
# 归档与删除
# --------------------------------------------------------------------------- #


def test_archive_hides_from_default_list(service) -> None:
    subjects, _ = service
    created = subjects.create("要归档的")

    subjects.archive(created.id)

    assert subjects.list_summaries() == []
    assert len(subjects.list_summaries(include_archived=True)) == 1

    subjects.archive(created.id, archived=False)
    assert len(subjects.list_summaries()) == 1


def test_archive_unknown_subject_raises(service) -> None:
    subjects, _ = service
    with pytest.raises(KeyError):
        subjects.archive("不存在")


def test_delete_removes_subject_and_related_data(service) -> None:
    subjects, repos = service
    created = subjects.create("要删除的")
    node = None
    path = repos.paths.create(
        created.id,
        rationale="r",
        model=None,
        nodes=[
            {
                "title": "节点",
                "objective": "目标",
                "key_points": ["a", "b", "c"],
                "checkpoints": [{"question": "q", "expected_points": ["p"], "reference_answer": "a"}],
            }
        ],
    )
    node = path.ordered_nodes()[0]

    subjects.delete(created.id)

    assert repos.subjects.get(created.id) is None
    assert repos.paths.get(path.id) is None
    assert repos.paths.node(node.id) is None
    assert repos.sessions.get(created.id) is None


def test_delete_unknown_subject_raises(service) -> None:
    subjects, _ = service
    with pytest.raises(KeyError):
        subjects.delete("不存在")


# --------------------------------------------------------------------------- #
# 读取与概览
# --------------------------------------------------------------------------- #


def test_get_unknown_subject_raises(service) -> None:
    subjects, _ = service
    with pytest.raises(KeyError):
        subjects.get("不存在")


def test_list_summaries_includes_progress(service) -> None:
    subjects, repos = service
    created = subjects.create("有进度的")
    path = repos.paths.create(
        created.id,
        rationale="r",
        model=None,
        nodes=[
            {
                "title": f"节点 {index}",
                "objective": "目标",
                "key_points": ["a", "b", "c"],
                "checkpoints": [{"question": "q", "expected_points": ["p"], "reference_answer": "a"}],
            }
            for index in range(3)
        ],
    )
    repos.paths.update_node(path.ordered_nodes()[0].id, state=NodeState.MASTERED)

    summary = subjects.list_summaries()[0]

    assert summary.subject.id == created.id
    assert summary.progress.total_nodes == 3
    assert summary.progress.mastered_nodes == 1


def test_progress_of_unknown_subject_raises(service) -> None:
    subjects, _ = service
    with pytest.raises(KeyError):
        subjects.progress("不存在")


def test_has_same_title(service) -> None:
    """FR-107：允许同名，但要能提示。"""
    subjects, _ = service
    created = subjects.create("贝叶斯定理")

    assert subjects.has_same_title("贝叶斯定理") is True
    assert subjects.has_same_title("  贝叶斯定理  ") is True
    assert subjects.has_same_title("贝叶斯定理", exclude_id=created.id) is False
    assert subjects.has_same_title("另一个概念") is False
    # 非法名称不报错，只返回 False（用于"边输入边提示"的场景）
    assert subjects.has_same_title("!") is False


def test_status_reflects_state_machine(service) -> None:
    subjects, repos = service
    created = subjects.create("概念")
    repos.subjects.set_status(created.id, SubjectState.TEACHING)

    assert subjects.status(created.id) is SubjectState.TEACHING
