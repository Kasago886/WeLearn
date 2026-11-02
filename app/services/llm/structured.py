"""结构化输出通道（T1.5，FR-716）。

把官方 JSON Output 的四个前置条件变成代码约束，而不是靠每个调用点自觉：

1. 请求带 ``response_format={"type":"json_object"}``；
2. 提示词中**必须出现 `json` 字样**并给出输出样例（缺失时直接报错，属于编码缺陷）；
3. **必须显式设置 `max_tokens`**，否则 JSON 会被中途截断；
4. 处理官方已知的"返回空 `content`"，以及 JSON 被截断。

解析容错分三级（FR-607）：严格 JSON → 去掉 markdown 代码块 → 截取最外层 ``{}``。
重试策略：空内容或解析失败都重试一次，并**把 `max_tokens` 翻倍**——因为截断的根因通常就是它太小。
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.contracts.schemas import LlmPurpose
from app.infra.llm.errors import LlmError, LlmErrorCategory
from app.infra.llm.protocol import TokenUsage
from app.infra.logging.queue_logger import CHANNEL_LLM, get_logger
from app.services.llm.agent import RoundExecutor

TModel = TypeVar("TModel", bound=BaseModel)

JSON_KEYWORD = "json"
DEFAULT_MAX_ATTEMPTS = 2


class ParseOutcome(StrEnum):
    """解析结果。"""

    OK = "ok"
    EMPTY = "empty"
    INVALID = "invalid"


@dataclass(frozen=True)
class ParseResult:
    outcome: ParseOutcome
    payload: dict[str, object] | None = None


def parse_json_payload(text: str | None) -> ParseResult:
    """三级容错解析 JSON 对象文本。"""
    if text is None:
        return ParseResult(ParseOutcome.EMPTY)

    candidate = text.strip()
    if not candidate:
        return ParseResult(ParseOutcome.EMPTY)

    direct = _try_load(candidate)
    if direct is not None:
        return direct

    unfenced = _strip_code_fence(candidate)
    if unfenced != candidate:
        fenced = _try_load(unfenced)
        if fenced is not None:
            return fenced

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        sliced = _try_load(candidate[start : end + 1])
        if sliced is not None:
            return sliced

    return ParseResult(ParseOutcome.INVALID)


def _try_load(text: str) -> ParseResult | None:
    import json

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(loaded, dict):
        return None
    return ParseResult(ParseOutcome.OK, loaded)


def _strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` 或 ``` ... ``` 包裹。"""
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if len(lines) < 2:
        return text

    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


class StructuredOutputError(LlmError):
    """结构化输出在重试后仍不可用。携带重试建议，供上层决定是否再试。"""

    def __init__(self, message: str, *, detail: str = "") -> None:
        super().__init__(LlmErrorCategory.PROTOCOL, message, retryable=True, detail=detail)


class StructuredChannel:
    """需要 JSON 输出的一次调用。"""

    def __init__(
        self,
        executor: RoundExecutor,
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_tokens_ceiling: int | None = None,
    ) -> None:
        self._executor = executor
        self._max_attempts = max(1, max_attempts)
        self._ceiling = max_tokens_ceiling
        self._log = get_logger(CHANNEL_LLM)

    async def request(
        self,
        purpose: LlmPurpose,
        *,
        system: str,
        user: str,
        schema: type[TModel],
        max_tokens: int | None = None,
    ) -> tuple[TModel, TokenUsage | None]:
        """请求并解析结构化输出。

        抛出 :class:`StructuredOutputError` 表示"重试后仍拿不到可用的 JSON"，
        上层可以据此重试，或（如判定器）降级为更保守的结论。
        """
        self._assert_json_instruction(system, user)

        attempts = 0
        # 必须解析出**具体**的 token 预算：只有具体值才能翻倍，
        # 否则重试时用的还是策略默认值，"防截断重试"就形同虚设（FR-716）。
        base_tokens = max_tokens if max_tokens is not None else self._executor.policy.resolve(purpose).max_tokens
        last_detail = ""

        while attempts < self._max_attempts:
            attempts += 1
            content, usage = await self._executor.call_blocking(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                purpose=purpose,
                max_tokens=base_tokens,
                json_object=True,
            )

            parsed = parse_json_payload(content)
            if parsed.outcome is ParseOutcome.OK and parsed.payload is not None:
                try:
                    return schema.model_validate(parsed.payload), usage
                except ValidationError as exc:
                    last_detail = f"结构校验失败: {exc.error_count()} 处，首处: {exc.errors()[0].get('msg')}"
                    self._log.warning(f"结构化输出校验失败（第 {attempts} 次）: {last_detail}")
            elif parsed.outcome is ParseOutcome.EMPTY:
                last_detail = "模型返回空内容（官方已知问题）"
                self._log.warning(f"结构化输出为空（第 {attempts} 次）purpose={purpose.value}")
            else:
                last_detail = "JSON 解析失败（可能被 max_tokens 截断）"
                self._log.warning(
                    f"结构化输出无法解析（第 {attempts} 次）purpose={purpose.value} "
                    f"首 120 字符: {content[:120]!r}"
                )

            base_tokens = self._grow_tokens(base_tokens)

        raise StructuredOutputError(
            f"模型未能返回可用的结构化结果（已尝试 {attempts} 次）。{last_detail}",
            detail=last_detail,
        )

    def _grow_tokens(self, current: int) -> int:
        """截断的根因通常是 max_tokens 太小，因此重试时翻倍（FR-716）。"""
        grown = current * 2
        if self._ceiling is not None:
            return min(grown, self._ceiling)
        return grown

    @staticmethod
    def _assert_json_instruction(system: str, user: str) -> None:
        combined = f"{system}\n{user}"
        if JSON_KEYWORD not in combined.lower():
            raise ValueError(
                "使用 JSON Output 时提示词必须包含 json 字样并给出输出样例（FR-716），"
                "否则官方不保证返回合法 JSON。"
            )
