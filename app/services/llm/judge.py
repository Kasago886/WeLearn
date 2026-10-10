"""检查点判定器（T1.6，FR-404、FR-405）。

两点关键设计：

**判定必须是独立的一次调用**（FR-405）。若让模型"边讲边判"，它会倾向于给自己刚讲过的内容打高分，
门禁形同虚设。因此这里用一份**全新的消息数组**（不含教学对话），只喂检查点与学习者作答。

**保守降级而非盲目相信**（FR-404）。官方 JSON Output 有概率返回空内容或结构不全，
当模型判定为 ``pass`` 却没给出 ``hit_points``（或同时给了 ``missing_points``）时，
说明这个 ``pass`` 没有依据，一律降级为 ``partial``——**宁可不放行，也不能假放行**。
"""

from dataclasses import dataclass

from app.contracts.schemas import JudgementResult, LlmPurpose, Verdict
from app.infra.logging.queue_logger import CHANNEL_LLM, get_logger
from app.prompts import JUDGE, render
from app.services.llm.structured import StructuredChannel, StructuredOutputError

DEGRADE_NOTE = "（判定依据不足，已按未完全掌握处理。）"


@dataclass(frozen=True)
class JudgeInput:
    """一次判定的输入。"""

    node_title: str
    objective: str
    question: str
    expected_points: list[str]
    reference_answer: str
    user_answer: str


class Judge:
    """独立的检查点判定器。"""

    def __init__(self, channel: StructuredChannel, *, max_points_in_feedback: int = 3) -> None:
        self._channel = channel
        self._max_points = max_points_in_feedback
        self._log = get_logger(CHANNEL_LLM)

    async def judge(self, payload: JudgeInput) -> JudgementResult:
        """判定一次作答。

        抛 :class:`StructuredOutputError` 表示连"可解析的判定"都拿不到；
        调用方应把它当作可重试失败，而**不是**当作未通过——两者对学习者的含义完全不同。
        """
        system = render(
            JUDGE,
            node_title=payload.node_title,
            objective=payload.objective,
            question=payload.question,
            expected_points=payload.expected_points,
            reference_answer=payload.reference_answer,
            user_answer=payload.user_answer,
        )
        user = "请依据以上判定要点给出 json 判定结果。"

        result, _usage = await self._channel.request(
            LlmPurpose.JUDGE,
            system=system,
            user=user,
            schema=JudgementResult,
        )
        return self._normalize(result)

    def _normalize(self, result: JudgementResult) -> JudgementResult:
        """把"没有依据的 pass"降级为 partial（保守方向）。"""
        if result.verdict is not Verdict.PASS:
            return result

        if result.hit_points:
            if result.missing_points:
                self._log.warning("判定为 pass 但同时给出 missing_points，按 partial 处理")
                return self._downgrade(result)
            return result

        self._log.warning("判定为 pass 但未列出 hit_points，按 partial 处理（FR-404 要求 pass 必须给出命中要点）")
        return self._downgrade(result)

    def _downgrade(self, result: JudgementResult) -> JudgementResult:
        missing = list(result.missing_points)
        if not missing:
            missing = ["未能确认是否命中全部判定要点"]
        feedback = result.feedback.rstrip()
        if not feedback:
            feedback = "暂未确认掌握情况。"
        return JudgementResult(
            verdict=Verdict.PARTIAL,
            confidence=min(result.confidence, 0.5),
            feedback=f"{feedback}{DEGRADE_NOTE}",
            hit_points=list(result.hit_points),
            missing_points=missing[: self._max_points],
        )


async def judge_or_error(
    judge: Judge,
    payload: JudgeInput,
) -> tuple[JudgementResult | None, str]:
    """判定并把结构化错误转成可操作提示，避免调用方散落异常处理。"""
    try:
        return await judge.judge(payload), ""
    except StructuredOutputError as exc:
        return None, exc.user_message
