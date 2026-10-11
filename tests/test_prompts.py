"""提示词模板测试（NFR-112、FR-716）。"""

import pytest

from app.prompts import JUDGE, available_templates, load_template, render
from app.services.llm.judge import JudgeInput


def test_required_templates_exist() -> None:
    """P1 需要的模板必须存在；缺模板会让整条链路在运行时才崩。"""
    templates = available_templates()

    assert "judge" in templates
    assert "compressor" in templates


def test_render_substitutes_values() -> None:
    text = render(
        JUDGE,
        node_title="贝叶斯定理",
        objective="能解释公式",
        question="请解释 P(A|B)。",
        expected_points=["说明后验概率", "提到先验"],
        reference_answer="参考",
        user_answer="我的回答",
    )

    assert "贝叶斯定理" in text
    assert "请解释 P(A|B)。" in text
    assert "- 说明后验概率" in text
    assert "我的回答" in text
    # 模板里不能残留占位符
    assert "$" not in text


def test_render_handles_missing_values_without_crashing() -> None:
    """缺参数时保留占位符即可，不能让 KeyError 把整轮对话打崩。"""
    text = render(JUDGE, node_title="只有标题")

    assert "只有标题" in text
    assert "$question" in text


def test_render_formats_list_and_none() -> None:
    text = render(JUDGE, expected_points=["a", "b"], reference_answer=None)

    assert "- a" in text
    assert "- b" in text
    assert "（无）" in text


def test_render_formats_empty_list_as_placeholder() -> None:
    text = render(JUDGE, expected_points=[])

    assert "（无）" in text


def test_missing_template_raises_with_path() -> None:
    with pytest.raises(FileNotFoundError) as exc:
        load_template("不存在的模板")

    assert "不存在的模板" in str(exc.value)


def test_judge_template_requires_json_output_and_hit_points() -> None:
    """判定模板必须同时满足 FR-404 与官方 JSON Output 前置条件。"""
    text = render(
        JUDGE,
        node_title="n",
        objective="o",
        question="q",
        expected_points=["p"],
        reference_answer="r",
        user_answer="a",
    )

    lower = text.lower()
    assert "json" in lower  # 官方要求提示词含 json 字样
    assert "hit_points" in text  # FR-404：pass 必须列出命中要点
    assert "missing_points" in text
    # 不得出现打击性措辞约束
    assert "不得出现人身评判" in text


def test_judge_template_tells_model_the_pass_rule_explicitly() -> None:
    """Spike B 实测（docs/Spike实测结果.md）：模型会把"还能说得更好"也写进 missing_points，
    而 `pass` + 非空 `missing_points` 会触发放水守卫降级为 `partial`——
    结果是答对的人被反复重讲。规则必须写在提示词里，而不能只靠守卫兜底。"""
    text = render(JUDGE, user_answer="a")

    assert "`missing_points` 必须为空数组" in text
    assert "不是缺失要点" in text


def test_compressor_template_requires_json_output() -> None:
    from app.prompts import COMPRESSOR
    from app.prompts import render as render_prompt

    text = render_prompt(COMPRESSOR, payload="{}")

    assert "json" in text.lower()
    assert "compressed_context" in text
    # 必须声明"不新增事实"与空摘要语义
    assert "不新增任何事实" in text
    assert '{"compressed_context": ""}' in text


def test_judge_input_is_renderable() -> None:
    """端到端小检查：JudgeInput 的字段名与模板占位符必须对得上。"""
    payload = JudgeInput(
        node_title="n",
        objective="o",
        question="q",
        expected_points=["p"],
        reference_answer="r",
        user_answer="a",
    )

    text = render(
        JUDGE,
        node_title=payload.node_title,
        objective=payload.objective,
        question=payload.question,
        expected_points=payload.expected_points,
        reference_answer=payload.reference_answer,
        user_answer=payload.user_answer,
    )

    assert "$" not in text
