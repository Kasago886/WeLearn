"""主题、测评、路径与教学的 HTTP 接口（p2-api-sse）。

约定：

- **流式端点**（`/assessment/*`、`/path/generate`、`/teach/*`）返回 SSE，
  载荷格式见 docs/接口契约.md；错误以 `error` 事件承载，便于前端统一处理；
- **非流式端点**返回 JSON；请求级错误（主题不存在、状态非法）用 HTTP 状态码表达；
- 路由层只做参数校验与序列化，业务判断全部在服务层。
"""

from collections import Counter
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import get_stack, require_subject
from app.api.serializers import (
    answer_payload,
    assessment_payload,
    node_payload,
    path_payload,
    session_payload,
    subject_payload,
    subject_summary_payload,
    weak_point_payload,
)
from app.api.sse import TurnStream, sse_response
from app.contracts.events import (
    AssessmentEventStatus,
    AssessmentLevelPayload,
    AssessmentPayload,
    AssessmentProfilePayload,
    CheckpointPayload,
    HintPayload,
    NodeStatePayload,
    OptionsKind,
    OptionsPayload,
    PathNodePayload,
    PathPayload,
    SseEventType,
    VerdictPayload,
    encode_frame,
)
from app.contracts.schemas import AssessmentStatus, ConversationKind, MasteryLevel, NodeDecision
from app.services.learning.entities import KnowledgePoint
from app.services.learning.planning import PlanningError
from app.services.learning.stack import LearningStack
from app.services.learning.state import TransitionError
from app.services.learning.teaching import HINT_LEVEL_LABELS

router = APIRouter(prefix="/api/subjects", tags=["subjects"])


def _conflict(exc: Exception) -> HTTPException:
    """领域流程异常 → 409：请求本身没问题，是**当前状态**不允许（FR-410）。"""
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


# --------------------------------------------------------------------------- #
# 请求模型
# --------------------------------------------------------------------------- #


class CreateSubjectRequest(BaseModel):
    title: str
    goal: str | None = None
    background: str | None = None
    depth: str | None = None


class UpdateSubjectRequest(BaseModel):
    title: str | None = None
    goal: str | None = None
    background: str | None = None
    depth: str | None = None


class ArchiveRequest(BaseModel):
    archived: bool = True


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=4000)


class DecisionRequest(BaseModel):
    decision: NodeDecision
    direction: str | None = None


class OverrideLevelRequest(BaseModel):
    topic: str
    level: MasteryLevel


class UpdateNodeRequest(BaseModel):
    """改节点内容（FR-304）。全部字段可选，只改传了的那些。"""

    title: str | None = None
    objective: str | None = None
    key_points: list[str] | None = None
    prerequisites: list[str] | None = None


class AddNodeRequest(BaseModel):
    """在某个节点之后插入空白节点；不传则追加到末尾。"""

    after_node_id: str | None = None


class MoveNodeRequest(BaseModel):
    """-1 上移 / 1 下移。"""

    offset: int = Field(ge=-1, le=1)


class RegenerateRequest(BaseModel):
    """期望的调整方向（可留空）。整块请求体可以省略。"""

    requirement: str = Field(default="", max_length=500)


class HintRequest(BaseModel):
    """提示级别（FR-407）：不传则按连续失败次数自动选级。"""

    level: int | None = Field(default=None, ge=1, le=3)


class AskRequest(BaseModel):
    """穿插提问（FR-411）。"""

    question: str = Field(min_length=1, max_length=2000)


# --------------------------------------------------------------------------- #
# 主题
# --------------------------------------------------------------------------- #


@router.get("")
def list_subjects(
    include_archived: bool = Query(default=False),
    stack: LearningStack = Depends(get_stack),
) -> dict[str, Any]:
    """主题列表（FR-103、FR-106）。"""
    items = stack.subjects.list_summaries(include_archived=include_archived)
    payloads = [subject_summary_payload(item) for item in items]

    # FR-107：同名主题允许存在，但列表里要能看出来
    titles = Counter(payload["title"] for payload in payloads)
    for payload in payloads:
        payload["duplicate_title_hint"] = titles[payload["title"]] > 1

    return {"items": payloads}


@router.post("", status_code=status.HTTP_201_CREATED)
def create_subject(payload: CreateSubjectRequest, stack: LearningStack = Depends(get_stack)) -> dict[str, Any]:
    """创建学习主题（FR-101、FR-102）。"""
    from app.services.learning.subjects import SubjectValidationError

    try:
        subject = stack.subjects.create(
            payload.title,
            goal=payload.goal,
            background=payload.background,
            depth=payload.depth,
        )
    except SubjectValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    summary = next(
        (item for item in stack.subjects.list_summaries(include_archived=True) if item.subject.id == subject.id),
        None,
    )
    return {
        "subject": subject_payload(subject),
        "progress": subject_summary_payload(summary)["progress"] if summary else None,
        # FR-107：同名不算错误，只提示
        "duplicate_title_hint": stack.subjects.has_same_title(subject.title, exclude_id=subject.id),
    }


@router.get("/{subject_id}")
def get_subject(subject_id: str, stack: LearningStack = Depends(get_stack)) -> dict[str, Any]:
    require_subject(stack, subject_id)
    subject = stack.subjects.get(subject_id)
    progress = stack.subjects.progress(subject_id)
    return {
        "subject": subject_payload(subject),
        "progress": {
            "total_nodes": progress.total_nodes,
            "mastered_nodes": progress.mastered_nodes,
            "partial_nodes": progress.partial_nodes,
            "skipped_nodes": progress.skipped_nodes,
            "weak_points": progress.weak_points,
            "total_tokens": progress.total_tokens,
            "mastery_ratio": round(progress.mastery_ratio, 3),
        },
        "last_studied_at": None,
    }


@router.patch("/{subject_id}")
def update_subject(
    subject_id: str,
    payload: UpdateSubjectRequest,
    stack: LearningStack = Depends(get_stack),
) -> dict[str, Any]:
    from app.services.learning.subjects import SubjectValidationError

    require_subject(stack, subject_id)
    try:
        subject = stack.subjects.update(
            subject_id,
            title=payload.title,
            goal=payload.goal,
            background=payload.background,
            depth=payload.depth,
        )
    except SubjectValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return {"subject": subject_payload(subject)}


@router.post("/{subject_id}/archive")
def archive_subject(
    subject_id: str,
    payload: ArchiveRequest,
    stack: LearningStack = Depends(get_stack),
) -> dict[str, Any]:
    require_subject(stack, subject_id)
    subject = stack.subjects.archive(subject_id, archived=payload.archived)
    return {"subject": subject_payload(subject)}


@router.delete("/{subject_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_subject(subject_id: str, stack: LearningStack = Depends(get_stack)) -> None:
    """删除主题及其全部下级数据（FR-104）。界面负责二次确认。"""
    require_subject(stack, subject_id)
    stack.subjects.delete(subject_id)


# --------------------------------------------------------------------------- #
# 状态机（刷新恢复）
# --------------------------------------------------------------------------- #


@router.get("/{subject_id}/state")
def get_state(subject_id: str, stack: LearningStack = Depends(get_stack)) -> dict[str, Any]:
    """状态机快照——前端刷新后唯一的恢复入口（FR-410）。"""
    require_subject(stack, subject_id)
    view = stack.state.resume(subject_id, node_loader=stack.repos.paths.node)
    path = stack.planning.current_path(subject_id)

    return {
        "session": session_payload(view.snapshot, next_action=view.next_action),
        "subject": subject_payload(view.subject),
        "node": node_payload(view.node) if view.node is not None else None,
        "path": path_payload(path) if path is not None else None,
        "can_answer": view.can_answer,
        "needs_decision": view.needs_decision,
    }


@router.get("/{subject_id}/weak-points")
def list_weak_points(subject_id: str, stack: LearningStack = Depends(get_stack)) -> dict[str, Any]:
    require_subject(stack, subject_id)
    points = stack.repos.weak_points.list_open(subject_id)
    return {"items": [weak_point_payload(point) for point in points]}


# --------------------------------------------------------------------------- #
# 测评
# --------------------------------------------------------------------------- #


def _assessment_frames(turn: Any) -> list[str]:
    """把测评产出转成 ``assessment`` 事件（契约 v1.1）。"""
    status_value = (
        AssessmentEventStatus.SKIPPED
        if turn.skipped
        else AssessmentEventStatus.PROFILE
        if turn.profile is not None
        else AssessmentEventStatus.QUESTION
    )
    profile = None
    if turn.profile is not None:
        profile = AssessmentProfilePayload(
            levels=[
                AssessmentLevelPayload(
                    topic=item.topic,
                    level=item.level.value,
                    evidence=item.evidence,
                )
                for item in turn.profile.levels
            ],
            summary=turn.profile.summary,
            suggested_start_node=turn.profile.suggested_start_node,
        )

    payload = AssessmentPayload(
        status=status_value,
        topics=list(turn.topics),
        turn_number=turn.turn_number,
        total_turns=turn.total_turns,
        current_topic=turn.current_topic,
        profile=profile,
    )
    return [encode_frame(SseEventType.ASSESSMENT, payload)]


@router.post("/{subject_id}/assessment/start")
def assessment_start(subject_id: str, stack: LearningStack = Depends(get_stack)) -> Any:
    """开始测评：SSE 返回首个问题（FR-201）。"""
    require_subject(stack, subject_id)
    stream = TurnStream(lambda on_delta: stack.assessments.start(subject_id))
    return sse_response(stream, postprocess=_assessment_frames)


@router.post("/{subject_id}/assessment/answer")
def assessment_answer(
    subject_id: str,
    payload: AnswerRequest,
    stack: LearningStack = Depends(get_stack),
) -> Any:
    """提交测评作答：继续提问或产出画像（FR-202、FR-206）。"""
    require_subject(stack, subject_id)
    stream = TurnStream(
        lambda on_delta: stack.assessments.answer(subject_id, payload.answer, on_delta=on_delta)
    )
    return sse_response(stream, postprocess=_assessment_frames)


@router.post("/{subject_id}/assessment/skip")
def assessment_skip(subject_id: str, stack: LearningStack = Depends(get_stack)) -> dict[str, Any]:
    """跳过测评，按零基础处理（FR-205）。"""
    require_subject(stack, subject_id)
    turn = stack.assessments.skip(subject_id)
    return {"text": turn.text, "skipped": turn.skipped, "state": "path_review"}


@router.get("/{subject_id}/assessment")
def assessment_view(subject_id: str, stack: LearningStack = Depends(get_stack)) -> dict[str, Any]:
    """画像与测评状态（FR-206、FR-208）。

    测评进行中时额外返回 ``transcript`` 与 ``pending_question``：刷新后必须能重新看到
    "正在问的那个问题"，否则用户面对着输入框却不知道要答什么（FR-410）。
    """
    require_subject(stack, subject_id)
    view = stack.assessments.view(subject_id)
    if view is None:
        return {"assessment": None, "transcript": [], "pending_question": None}

    transcript: list[dict[str, str]] = []
    if view.assessment.status is AssessmentStatus.RUNNING:
        transcript = stack.transcript(subject_id, ConversationKind.ASSESS)
    pending = transcript[-1]["content"] if transcript and transcript[-1]["role"] == "assistant" else None

    return {
        "assessment": assessment_payload(view.assessment, view.levels),
        "transcript": transcript,
        "pending_question": pending,
    }


@router.post("/{subject_id}/assessment/level")
def override_level(
    subject_id: str,
    payload: OverrideLevelRequest,
    stack: LearningStack = Depends(get_stack),
) -> dict[str, Any]:
    """人工修正某个子领域的水平（FR-207）。修正后需要重新生成路径。"""
    require_subject(stack, subject_id)
    if not stack.assessments.override_level(subject_id, payload.topic, payload.level):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="没有找到该子领域的掌握情况")

    view = stack.assessments.view(subject_id)
    assert view is not None
    return {"assessment": assessment_payload(view.assessment, view.levels), "regenerate_path": True}


# --------------------------------------------------------------------------- #
# 学习路径
# --------------------------------------------------------------------------- #


def _path_frames(turn: Any) -> list[str]:
    if turn.path is None:
        return []
    path = turn.path
    payload = PathPayload(
        path_id=path.id,
        version=path.version,
        rationale=path.rationale,
        nodes=[
            PathNodePayload(
                id=node.id,
                index=node.order_index,
                title=node.title,
                objective=node.objective,
                key_points=list(node.key_points),
                state=node.state.value,
                is_quick_review=node.is_quick_review,
                checkpoints=[CheckpointPayload(id=item.id, question=item.question) for item in node.checkpoints],
            )
            for node in path.ordered_nodes()
        ],
    )
    return [encode_frame(SseEventType.PATH, payload)]


@router.post("/{subject_id}/path/generate")
def path_generate(subject_id: str, stack: LearningStack = Depends(get_stack)) -> Any:
    """生成学习路径（FR-301~FR-303、FR-308）。"""
    require_subject(stack, subject_id)
    stream = TurnStream(lambda on_delta: stack.planning.generate(subject_id))
    return sse_response(stream, postprocess=_path_frames)


@router.get("/{subject_id}/path")
def path_view(subject_id: str, stack: LearningStack = Depends(get_stack)) -> dict[str, Any]:
    require_subject(stack, subject_id)
    view = stack.planning.view(subject_id)
    if view is None:
        return {"path": None}

    return {
        "path": path_payload(view.path),
        "profile": [
            {"topic": level.topic, "level": level.level.value, "evidence": level.evidence}
            for level in view.profile_levels
        ],
        # 位置从 1 开始；全部完成时指向最后一个节点（是否完成看 completed）
        "current_index": min(view.current_index + 1, len(view.path.ordered_nodes())),
        "completed": view.completed,
        "versions": [
            {"path_id": path_id, "version": version, "is_active": is_active}
            for path_id, version, is_active, _created in stack.planning.versions(subject_id)
        ],
    }


@router.post("/{subject_id}/path/confirm")
def path_confirm(subject_id: str, stack: LearningStack = Depends(get_stack)) -> dict[str, Any]:
    """确认路径并进入教学（FR-306 的门禁）。"""
    require_subject(stack, subject_id)
    try:
        path = stack.planning.confirm(subject_id)
    except (PlanningError, TransitionError) as exc:
        raise _conflict(exc) from exc
    return {"path": path_payload(path), "state": "teaching"}


# --------------------------------------------------------------------------- #
# 路径编辑（FR-304、FR-305）
# --------------------------------------------------------------------------- #


def _edit_call(action: Callable[[], Any]) -> dict[str, Any]:
    """把编辑操作统一成 "返回最新路径 + 把领域错误转成 409"。

    编辑冲突（删掉已开始的节点、移动到头了）都是"请求合法但当前状态不允许"，
    用 409 而不是 400/422，前端可以据此提示而不是当成参数错误。
    """
    try:
        path = action()
    except PlanningError as exc:
        raise _conflict(exc) from exc
    return {"path": path_payload(path)}


@router.patch("/{subject_id}/path/nodes/{node_id}")
def path_update_node(
    subject_id: str,
    node_id: str,
    payload: UpdateNodeRequest,
    stack: LearningStack = Depends(get_stack),
) -> dict[str, Any]:
    """改节点的标题 / 目标 / 要点（FR-304）。"""
    require_subject(stack, subject_id)
    return _edit_call(
        lambda: stack.planning.update_node(
            subject_id,
            node_id,
            title=payload.title,
            objective=payload.objective,
            key_points=payload.key_points,
            prerequisites=payload.prerequisites,
        )
    )


@router.post("/{subject_id}/path/nodes")
def path_add_node(
    subject_id: str,
    payload: AddNodeRequest,
    stack: LearningStack = Depends(get_stack),
) -> dict[str, Any]:
    """插入一个空白节点（FR-304）。"""
    require_subject(stack, subject_id)
    return _edit_call(lambda: stack.planning.add_node(subject_id, after_node_id=payload.after_node_id))


@router.delete("/{subject_id}/path/nodes/{node_id}")
def path_delete_node(
    subject_id: str,
    node_id: str,
    stack: LearningStack = Depends(get_stack),
) -> dict[str, Any]:
    """删除节点（FR-304）；只允许删除尚未开始的节点。"""
    require_subject(stack, subject_id)
    return _edit_call(lambda: stack.planning.delete_node(subject_id, node_id))


@router.post("/{subject_id}/path/nodes/{node_id}/move")
def path_move_node(
    subject_id: str,
    node_id: str,
    payload: MoveNodeRequest,
    stack: LearningStack = Depends(get_stack),
) -> dict[str, Any]:
    """上移 / 下移节点（FR-304）。"""
    require_subject(stack, subject_id)
    return _edit_call(lambda: stack.planning.move_node(subject_id, node_id, offset=payload.offset))


@router.post("/{subject_id}/path/nodes/{node_id}/regenerate")
def path_regenerate_node(
    subject_id: str,
    node_id: str,
    payload: RegenerateRequest | None = None,
    stack: LearningStack = Depends(get_stack),
) -> Any:
    """重生成单个节点（FR-304），保留节点 id 与它的作答记录。"""
    require_subject(stack, subject_id)
    requirement = payload.requirement if payload is not None else ""
    stream = TurnStream(
        lambda on_delta: stack.planning.regenerate_node(subject_id, node_id, requirement=requirement)
    )
    return sse_response(stream, postprocess=_path_frames)


@router.post("/{subject_id}/path/regenerate")
def path_regenerate(subject_id: str, stack: LearningStack = Depends(get_stack)) -> Any:
    """整条重新生成（FR-305）：旧版本保留，确认后新版本才成为当前版本。"""
    require_subject(stack, subject_id)
    stream = TurnStream(lambda on_delta: stack.planning.regenerate(subject_id))
    return sse_response(stream, postprocess=_path_frames)


# --------------------------------------------------------------------------- #
# 教学
# --------------------------------------------------------------------------- #


def _options_frames(turn: Any) -> list[str]:
    if not getattr(turn, "decision_required", False):
        return []
    payload = OptionsPayload(
        kind=OptionsKind.EXHAUSTED,
        choices=[NodeDecision.RETEACH.value, NodeDecision.SHOW_ANSWER.value, NodeDecision.ADVANCE.value],
        message="这个节点已经尝试多次。你可以选择：继续换个角度重讲、查看参考答案，或直接进入下一个节点。",
        directions=["simpler", "example", "hint"],
    )
    return [encode_frame(SseEventType.OPTIONS, payload)]


def _hint_frames(turn: Any) -> list[str]:
    """提示事件：告诉前端"这是第几级"，避免它自己从文本里猜。"""
    level = int(getattr(turn, "hint_level", 0) or 0)
    if level <= 0:
        return []
    return [
        encode_frame(
            SseEventType.HINT,
            HintPayload(
                level=level,
                label=HINT_LEVEL_LABELS.get(level, ""),
                question=str(getattr(turn, "pending_question", "") or ""),
            ),
        )
    ]


def _teaching_frames(stack: LearningStack) -> Any:
    """教学阶段的收尾事件：判定结果 + 节点状态 + （必要时）三选一。"""

    def postprocess(turn: Any) -> list[str]:
        frames: list[str] = []

        judgement = getattr(turn, "verdict", None)
        if judgement is not None:
            frames.append(
                encode_frame(
                    SseEventType.VERDICT,
                    VerdictPayload(
                        verdict=judgement.verdict,
                        confidence=judgement.confidence,
                        feedback=judgement.feedback,
                        hit_points=list(judgement.hit_points),
                        missing_points=list(judgement.missing_points),
                    ),
                )
            )

        node: KnowledgePoint | None = getattr(turn, "node", None)
        if node is not None:
            latest = stack.repos.paths.node(node.id)
            if latest is not None:
                frames.append(
                    encode_frame(
                        SseEventType.NODE_STATE,
                        NodeStatePayload(
                            node_id=latest.id,
                            state=latest.state,
                            index=getattr(turn, "node_index", 0),
                            total=getattr(turn, "node_total", 0),
                        ),
                    )
                )

        frames.extend(_options_frames(turn))
        frames.extend(_hint_frames(turn))
        return frames

    return postprocess


@router.post("/{subject_id}/teach/start")
def teach_start(subject_id: str, stack: LearningStack = Depends(get_stack)) -> Any:
    """进入/继续当前节点：必要时讲解，然后抛出检查点问题（FR-401、FR-403）。"""
    require_subject(stack, subject_id)
    stream = TurnStream(lambda on_delta: stack.teaching.start(subject_id, on_delta=on_delta))
    return sse_response(stream, postprocess=_teaching_frames(stack))


@router.post("/{subject_id}/teach/answer")
def teach_answer(
    subject_id: str,
    payload: AnswerRequest,
    stack: LearningStack = Depends(get_stack),
) -> Any:
    """提交检查点作答：判定 → 通过推进 / 未通过重讲 / 重试耗尽交给用户选择（FR-404、FR-406、FR-408）。"""
    require_subject(stack, subject_id)
    stream = TurnStream(
        lambda on_delta: stack.teaching.answer(subject_id, payload.answer, on_delta=on_delta)
    )
    return sse_response(stream, postprocess=_teaching_frames(stack))


@router.post("/{subject_id}/teach/decision")
def teach_decision(
    subject_id: str,
    payload: DecisionRequest,
    stack: LearningStack = Depends(get_stack),
) -> Any:
    """三选一：继续重讲 / 查看参考答案 / 直接进入下一节点（FR-408）。"""
    require_subject(stack, subject_id)
    stream = TurnStream(
        lambda on_delta: stack.teaching.decide(
            subject_id, payload.decision, direction=payload.direction, on_delta=on_delta
        )
    )
    return sse_response(stream, postprocess=_teaching_frames(stack))


@router.post("/{subject_id}/teach/skip")
def teach_skip(subject_id: str, stack: LearningStack = Depends(get_stack)) -> Any:
    """跳过当前节点（FR-409）。"""
    require_subject(stack, subject_id)
    stream = TurnStream(lambda on_delta: _skip(stack, subject_id))
    return sse_response(stream, postprocess=_teaching_frames(stack))


@router.post("/{subject_id}/teach/hint")
def teach_hint(
    subject_id: str,
    payload: HintRequest,
    stack: LearningStack = Depends(get_stack),
) -> Any:
    """索取提示（FR-407）：不传级别时按连续失败次数自动加深。"""
    require_subject(stack, subject_id)
    stream = TurnStream(
        lambda on_delta: stack.teaching.hint(subject_id, level=payload.level, on_delta=on_delta)
    )
    return sse_response(stream, postprocess=_teaching_frames(stack))


@router.post("/{subject_id}/teach/ask")
def teach_ask(
    subject_id: str,
    payload: AskRequest,
    stack: LearningStack = Depends(get_stack),
) -> Any:
    """穿插提问（FR-411）：回答后回到主线，不改变节点状态。"""
    require_subject(stack, subject_id)
    stream = TurnStream(
        lambda on_delta: stack.teaching.ask(subject_id, payload.question, on_delta=on_delta)
    )
    return sse_response(stream, postprocess=_teaching_frames(stack))


@router.post("/{subject_id}/teach/regenerate")
def teach_regenerate(subject_id: str, stack: LearningStack = Depends(get_stack)) -> Any:
    """重新生成上一条回复（FR-413）：按记录下来的动作重跑一次。"""
    require_subject(stack, subject_id)
    stream = TurnStream(lambda on_delta: stack.teaching.regenerate_last(subject_id, on_delta=on_delta))
    return sse_response(stream, postprocess=_teaching_frames(stack))


async def _skip(stack: LearningStack, subject_id: str) -> Any:
    """``skip_node`` 是同步方法，这里包一层以复用统一的 SSE 设施。"""
    return stack.teaching.skip_node(subject_id)


@router.post("/{subject_id}/teach/pause")
def teach_pause(subject_id: str, stack: LearningStack = Depends(get_stack)) -> dict[str, Any]:
    require_subject(stack, subject_id)
    try:
        snapshot = stack.teaching.pause(subject_id)
    except TransitionError as exc:
        raise _conflict(exc) from exc
    return {"state": snapshot.state.value}


@router.post("/{subject_id}/teach/resume")
def teach_resume(subject_id: str, stack: LearningStack = Depends(get_stack)) -> dict[str, Any]:
    require_subject(stack, subject_id)
    try:
        snapshot = stack.teaching.resume(subject_id)
    except TransitionError as exc:
        raise _conflict(exc) from exc
    return {"state": snapshot.state.value}


@router.get("/{subject_id}/teach/current")
def teach_current(subject_id: str, stack: LearningStack = Depends(get_stack)) -> dict[str, Any]:
    """教学页当前状态（不产生任何模型调用）。

    ``answers`` 是当前节点的作答记录：刷新后据此重建判定卡片与作答历史（FR-410）。
    讲解正文不重放——它可以直接重新生成，而作答记录是用户自己的输入，丢了就真的丢了。
    """
    require_subject(stack, subject_id)
    view = stack.teaching.view(subject_id)
    answers = (
        [answer_payload(record) for record in stack.repos.answers.list_for_node(view.node.id)]
        if view.node is not None
        else []
    )
    return {
        "session": session_payload(
            view.snapshot,
            next_action=_next_action(view),
        ),
        "node": node_payload(view.node) if view.node is not None else None,
        "checkpoint": (
            {
                "id": view.checkpoint.id,
                "question": view.checkpoint.question,
                "order_index": view.checkpoint.order_index,
            }
            if view.checkpoint is not None
            else None
        ),
        "node_index": view.node_index,
        "node_total": view.node_total,
        "can_answer": view.can_answer,
        "needs_decision": view.needs_decision,
        "answers": answers,
        # 最近一次教学动作；为空表示没有可"重新生成"的回复（FR-413）
        "last_action": view.snapshot.last_action,
    }


def _next_action(view: Any) -> str:
    if view.needs_decision:
        return "decide"
    if view.can_answer:
        return "answer"
    return "teach"


__all__ = ["router"]
