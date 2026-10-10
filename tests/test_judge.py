"""判定器测试（T1.6，FR-404、FR-405）。"""

import pytest

from app.contracts.schemas import JudgementResult, Verdict
from app.infra.settings import AppSettings
from app.services.llm.agent import RoundExecutor
from app.services.llm.judge import DEGRADE_NOTE, Judge, JudgeInput, judge_or_error
from app.services.llm.memory import Memory, user_message
from app.services.llm.policy import ModelPolicy
from app.services.llm.structured import StructuredChannel, StructuredOutputError
from tests.stubs import FakeLlmClient, json_response

PAYLOAD = JudgeInput(
    node_title="贝叶斯定理",
    objective="能写出并解释贝叶斯公式",
    question="请解释 P(A|B) 的含义。",
    expected_points=["说明后验概率", "提到先验概率", "提到似然"],
    reference_answer="P(A|B) 是在观察到 B 后 A 的概率。",
    user_answer="它是看到证据之后更新得到的概率。",
)

PASS_JSON = (
    '{"verdict":"pass","confidence":0.9,"feedback":"你抓住了后验概率的核心。",'
    '"hit_points":["说明后验概率","提到先验概率"],"missing_points":[]}'
)


def make_judge(client: FakeLlmClient, *, teaching_memory: Memory | None = None) -> Judge:
    executor = RoundExecutor(
        client,
        teaching_memory if teaching_memory is not None else Memory(),
        ModelPolicy(AppSettings()),
    )
    return Judge(StructuredChannel(executor))


# --------------------------------------------------------------------------- #
# 正常判定
# --------------------------------------------------------------------------- #


async def test_pass_with_hit_points_is_kept() -> None:
    client = FakeLlmClient(blocking_script=[json_response(PASS_JSON)])

    result = await make_judge(client).judge(PAYLOAD)

    assert result.verdict is Verdict.PASS
    assert result.is_pass() is True
    assert result.hit_points == ["说明后验概率", "提到先验概率"]
    assert result.confidence == 0.9


async def test_partial_and_fail_are_preserved() -> None:
    for verdict in ("partial", "fail"):
        payload = (
            f'{{"verdict":"{verdict}","confidence":0.4,"feedback":"还差一点。",'
            f'"hit_points":[],"missing_points":["提到似然"]}}'
        )
        client = FakeLlmClient(blocking_script=[json_response(payload)])

        result = await make_judge(client).judge(PAYLOAD)

        assert result.verdict.value == verdict
        assert result.missing_points == ["提到似然"]
        assert DEGRADE_NOTE not in result.feedback


async def test_verdict_payload_shape_for_sse() -> None:
    client = FakeLlmClient(blocking_script=[json_response(PASS_JSON)])

    payload = (await make_judge(client).judge(PAYLOAD)).to_payload()

    assert payload["verdict"] == "pass"
    assert payload["hit_points"] == ["说明后验概率", "提到先验概率"]
    assert payload["missing_points"] == []


# --------------------------------------------------------------------------- #
# 保守降级：不放行没有依据的 pass（FR-404）
# --------------------------------------------------------------------------- #


async def test_pass_without_hit_points_is_downgraded_to_partial() -> None:
    """这是防"放水"的关键断言：模型说通过但说不出通过了什么，就不能算通过。"""
    client = FakeLlmClient(
        blocking_script=[
            json_response('{"verdict":"pass","confidence":0.95,"feedback":"很好。","hit_points":[]}'),
        ]
    )

    result = await make_judge(client).judge(PAYLOAD)

    assert result.verdict is Verdict.PARTIAL
    assert result.is_pass() is False
    assert DEGRADE_NOTE in result.feedback
    assert result.confidence <= 0.5
    assert result.missing_points


async def test_pass_with_missing_points_is_downgraded() -> None:
    client = FakeLlmClient(
        blocking_script=[
            json_response(
                '{"verdict":"pass","confidence":0.8,"feedback":"通过。",'
                '"hit_points":["说明后验概率"],"missing_points":["提到似然"]}'
            )
        ]
    )

    result = await make_judge(client).judge(PAYLOAD)

    assert result.verdict is Verdict.PARTIAL
    assert result.missing_points == ["提到似然"]


async def test_downgrade_fills_placeholder_when_model_gives_nothing() -> None:
    client = FakeLlmClient(
        blocking_script=[json_response('{"verdict":"pass","confidence":1.0,"feedback":"通过。","hit_points":[]}')]
    )

    result = await make_judge(client).judge(PAYLOAD)

    assert result.missing_points == ["未能确认是否命中全部判定要点"]


# --------------------------------------------------------------------------- #
# 独立性（FR-405）：判定不得携带教学对话
# --------------------------------------------------------------------------- #


async def test_judgement_does_not_include_teaching_history() -> None:
    """判定必须是一次独立调用：只喂检查点与作答，避免模型给自己刚讲过的内容打高分。"""
    teaching = Memory.from_protocol(
        [
            [user_message("请讲讲贝叶斯定理")],
            [user_message("我好像懂了")],
        ]
    )
    client = FakeLlmClient(blocking_script=[json_response(PASS_JSON)])

    await make_judge(client, teaching_memory=teaching).judge(PAYLOAD)

    call = client.calls[0]
    sent = "\n".join(str(message.get("content", "")) for message in call.messages)
    assert "请讲讲贝叶斯定理" not in sent
    assert "我好像懂了" not in sent

    # 只有 system + user 两条
    assert [message["role"] for message in call.messages] == ["system", "user"]


async def test_judgement_prompt_carries_checkpoint_and_answer() -> None:
    client = FakeLlmClient(blocking_script=[json_response(PASS_JSON)])

    await make_judge(client).judge(PAYLOAD)

    prompt = str(client.calls[0].messages[0]["content"])
    assert PAYLOAD.question in prompt
    assert "说明后验概率" in prompt
    assert PAYLOAD.user_answer in prompt
    assert PAYLOAD.reference_answer in prompt
    # 提示词必须含 json 字样与样例（官方 JSON Output 前置条件）
    assert "json" in prompt.lower()

    # 判定对象不进教学记忆
    assert client.calls[0].purpose.value == "judge"


# --------------------------------------------------------------------------- #
# 失败处理
# --------------------------------------------------------------------------- #


async def test_structured_failure_propagates() -> None:
    client = FakeLlmClient(blocking_script=[json_response("不是 JSON"), json_response("还是不是")])

    with pytest.raises(StructuredOutputError) as exc:
        await make_judge(client).judge(PAYLOAD)

    assert exc.value.retryable is True


async def test_judge_or_error_converts_failure_to_message() -> None:
    """拿不到判定 ≠ 未通过：两者的含义完全不同，调用方必须能区分。"""
    client = FakeLlmClient(blocking_script=[json_response("坏数据"), json_response("坏数据")])

    result, message = await judge_or_error(make_judge(client), PAYLOAD)

    assert result is None
    assert message
    assert "重试" in message or "结构化" in message


async def test_judge_or_error_returns_result_on_success() -> None:
    client = FakeLlmClient(blocking_script=[json_response(PASS_JSON)])

    result, message = await judge_or_error(make_judge(client), PAYLOAD)

    assert isinstance(result, JudgementResult)
    assert message == ""
