"""提示词模板（NFR-112：模板外置，禁止硬编码在业务逻辑中）。

模板使用 :class:`string.Template` 的 ``$变量`` 语法而非 ``str.format``：
提示词里必然出现 JSON 样例（大量花括号），用 ``format`` 会需要满篇转义，极易出错。
"""

from pathlib import Path
from string import Template

TEMPLATE_DIR = Path(__file__).parent

ASSESSOR = "assessor"
ASSESSOR_TOPICS = "assessor_topics"
ASSESSOR_PROFILE = "assessor_profile"
PLANNER = "planner"
TUTOR = "tutor"
TUTOR_FEEDBACK = "tutor_feedback"
TUTOR_REFERENCE = "tutor_reference"
TUTOR_SUMMARY = "tutor_summary"
JUDGE = "judge"
COMPRESSOR = "compressor"
REVIEWER = "reviewer"


def load_template(name: str) -> Template:
    """按名称加载模板（文件名即模板名加 ``.md``）。"""
    path = TEMPLATE_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"提示词模板不存在: {path}")
    return Template(path.read_text(encoding="utf-8"))


def render(name: str, **values: object) -> str:
    """渲染模板。

    ``safe_substitute`` 保证模板里未被赋值的 ``$变量`` 原样保留——提示词缺参数时
    宁可让模型看到占位符，也不要因为 KeyError 让整轮对话崩掉。
    """
    normalized = {key: _stringify(value) for key, value in values.items()}
    return load_template(name).safe_substitute(normalized)


def _stringify(value: object) -> str:
    if value is None:
        return "（无）"
    if isinstance(value, list | tuple):
        return "\n".join(f"- {item}" for item in value) or "（无）"
    return str(value)


def available_templates() -> list[str]:
    """已提供的模板名（用于启动自检与测试）。"""
    return sorted(path.stem for path in TEMPLATE_DIR.glob("*.md"))
