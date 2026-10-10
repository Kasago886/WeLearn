"""领域与结构化输出的契约模型。

这些模型同时承担三个职责：
1. 作为冻结契约（docs/接口契约.md）的代码表达；
2. 校验模型返回的结构化结果，**把 SRS 里的硬约束变成构造即报错**，避免脏数据进入记忆或数据库；
3. 供存储层序列化使用。
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class MasteryLevel(StrEnum):
    """子领域掌握水平（FR-206：固定枚举，不允许自由文本）。"""

    NONE = "none"
    BASIC = "basic"
    PROFICIENT = "proficient"


class NodeState(StrEnum):
    """知识点状态（FR-309）。"""

    NOT_STARTED = "not_started"
    LEARNING = "learning"
    MASTERED = "mastered"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    REVIEW_NEEDED = "review_needed"


class Verdict(StrEnum):
    """检查点判定结论（FR-404）。"""

    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


class NodeDecision(StrEnum):
    """重试耗尽后用户的选择（FR-408）。"""

    RETEACH = "reteach"
    SHOW_ANSWER = "show_answer"
    ADVANCE = "advance"


class ConversationKind(StrEnum):
    """记忆容器用途，决定记忆隔离边界（FR-602）。"""

    ASSESS = "assess"
    TEACH = "teach"
    REVIEW = "review"


class LlmPurpose(StrEnum):
    """LLM 调用用途，用于按用途选择模型与思考档位（FR-703）。"""

    ASSESS = "assess"
    PLAN = "plan"
    TEACH = "teach"
    JUDGE = "judge"
    COMPRESS = "compress"


class ReasoningEffort(StrEnum):
    """思考强度（官方取值 low/high/max，默认 high）。"""

    LOW = "low"
    HIGH = "high"
    MAX = "max"


# --------------------------------------------------------------------------- #
# 测评（FR-206）
# --------------------------------------------------------------------------- #


class AssessmentLevel(BaseModel):
    """单个子领域的掌握判定及依据。"""

    topic: str = Field(min_length=1, max_length=100)
    level: MasteryLevel
    evidence: str = Field(min_length=1)


class AssessmentProfile(BaseModel):
    """掌握度画像：测评的最终产出，也是路径生成的唯一输入（FR-206、FR-209）。"""

    levels: list[AssessmentLevel] = Field(min_length=1, max_length=12)
    summary: str = Field(min_length=1)
    suggested_start_node: str = Field(min_length=1)


# --------------------------------------------------------------------------- #
# 学习路径（FR-302）
# --------------------------------------------------------------------------- #


class CheckpointDraft(BaseModel):
    """检查点：必须可判定，因此必须给出判定要点与参考答案。"""

    question: str = Field(min_length=1)
    expected_points: list[str] = Field(min_length=1)
    reference_answer: str = Field(min_length=1)


class PathNodeDraft(BaseModel):
    """路径节点。要点 3~6 条、检查点 ≥1 个（FR-302）。"""

    title: str = Field(min_length=1, max_length=120)
    objective: str = Field(min_length=1)
    prerequisites: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(min_length=3, max_length=6)
    checkpoints: list[CheckpointDraft] = Field(min_length=1)


class LearningPathDraft(BaseModel):
    """学习路径。节点数落在配置区间内（默认 3~15，FR-301）。"""

    rationale: str = Field(min_length=1)
    nodes: list[PathNodeDraft] = Field(min_length=3, max_length=15)


# --------------------------------------------------------------------------- #
# 判定（FR-404、FR-405）
# --------------------------------------------------------------------------- #


class JudgementResult(BaseModel):
    """判定结果。

    ``hit_points`` 是抑制"放水"的关键：要求模型在判定为 ``pass`` 时**必须列出学习者命中的要点**，
    写不出来就说明这次回答不该通过（FR-404、FR-405）。判定器会对缺失 ``hit_points`` 的 `pass`
    自动降级为 `partial`（保守方向），见 `app/services/llm/judge.py`。
    """

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    feedback: str = Field(min_length=1)
    hit_points: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)

    def is_pass(self) -> bool:
        return self.verdict is Verdict.PASS

    def to_payload(self) -> dict[str, object]:
        """转成 SSE 事件的载荷。"""
        return {
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "feedback": self.feedback,
            "hit_points": list(self.hit_points),
            "missing_points": list(self.missing_points),
        }


# --------------------------------------------------------------------------- #
# 上下文压缩（FR-605、FR-607）
# --------------------------------------------------------------------------- #


class CompressionSummary(BaseModel):
    """压缩摘要。空摘要表示无需保留（对应解析结果 Empty）。"""

    compressed_context: str = ""
