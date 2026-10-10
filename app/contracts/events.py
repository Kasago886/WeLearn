"""SSE 下行事件契约（T0.6）。

前端只依赖本文件的 `SseEventType` 与各 payload 的字段名；任何新增事件类型必须
同时更新 docs/接口契约.md，否则视为破坏性变更。

线上格式：

    event: token
    data: {"text": "北"}

    event: done
    data: {"usage": {...}}
"""

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.contracts.schemas import NodeState, Verdict


class SseEventType(StrEnum):
    """下行事件类型。"""

    TOKEN = "token"
    """正文增量（流式）。"""

    REASONING = "reasoning"
    """思维链增量（流式，前端可折叠）。"""

    VERDICT = "verdict"
    """检查点判定结果。"""

    NODE_STATE = "node_state"
    """节点状态变更。"""

    ASSESSMENT = "assessment"
    """测评阶段产出：子领域列表 / 掌握度画像 / 已跳过（v1.1 新增）。"""

    PATH = "path"
    """学习路径生成完成（v1.1 新增）。"""

    OPTIONS = "options"
    """需要用户做出选择（重试耗尽后的三选一，v1.1 新增）。"""

    ERROR = "error"
    """可操作的错误（已分类，不含敏感信息）。"""

    DONE = "done"
    """本轮结束，附带用量。"""


class AssessmentEventStatus(StrEnum):
    """``assessment`` 事件的状态。"""

    QUESTION = "question"
    PROFILE = "profile"
    SKIPPED = "skipped"


class OptionsKind(StrEnum):
    """需要用户选择的场景。"""

    EXHAUSTED = "exhausted"
    """单节点重试已耗尽（FR-408）。"""


class UsagePayload(BaseModel):
    """一次请求的 token 用量（FR-705）。"""

    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    cache_miss_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class TokenPayload(BaseModel):
    text: str


class ReasoningPayload(BaseModel):
    text: str


class VerdictPayload(BaseModel):
    verdict: Verdict
    confidence: float
    feedback: str
    hit_points: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)


class NodeStatePayload(BaseModel):
    node_id: str
    state: NodeState
    index: int = 0
    total: int = 0


class ErrorPayload(BaseModel):
    """错误必须携带"发生了什么 + 能做什么"（NFR-116）。"""

    category: str
    message: str
    retryable: bool


class DonePayload(BaseModel):
    conversation_id: str | None = None
    usage: UsagePayload | None = None


class AssessmentLevelPayload(BaseModel):
    topic: str
    level: str
    evidence: str
    user_overridden: bool = False


class AssessmentProfilePayload(BaseModel):
    levels: list[AssessmentLevelPayload] = Field(default_factory=list)
    summary: str = ""
    suggested_start_node: str = ""


class AssessmentPayload(BaseModel):
    """测评阶段产出（FR-201、FR-206、FR-205）。"""

    status: AssessmentEventStatus
    topics: list[str] = Field(default_factory=list)
    turn_number: int = 0
    total_turns: int = 0
    current_topic: str | None = None
    profile: AssessmentProfilePayload | None = None


class CheckpointPayload(BaseModel):
    id: str
    question: str


class PathNodePayload(BaseModel):
    id: str
    index: int
    title: str
    objective: str
    key_points: list[str] = Field(default_factory=list)
    state: str
    is_quick_review: bool = False
    checkpoints: list[CheckpointPayload] = Field(default_factory=list)


class PathPayload(BaseModel):
    """学习路径（FR-301、FR-308）。"""

    path_id: str
    version: int
    rationale: str
    nodes: list[PathNodePayload] = Field(default_factory=list)


class OptionsPayload(BaseModel):
    """需要用户选择（FR-408）。"""

    kind: OptionsKind
    choices: list[str] = Field(default_factory=list)
    message: str = ""
    directions: list[str] = Field(default_factory=list)


class SseFrame(BaseModel):
    """一帧 SSE 事件。"""

    type: SseEventType
    data: dict[str, Any] = Field(default_factory=dict)


def encode_frame(event_type: SseEventType, data: BaseModel | dict[str, Any] | None = None) -> str:
    """把事件编码为 SSE 帧文本（含结尾空行）。"""
    if data is None:
        payload: dict[str, Any] = {}
    elif isinstance(data, BaseModel):
        payload = data.model_dump(mode="json")
    else:
        payload = data

    body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_type.value}\ndata: {body}\n\n"
