"""本地联调用的假 LLM 服务端（p2-web-scaffold 的配套工具）。

**为什么需要它**：前端最需要验证的是"接口层 + 流式渲染"这一整条链路，而不是模型输出质量。
用真实 API 调试前端，每点一次按钮都要花钱、要等、还不稳定；这里把 LLM 换成按提示词识别的
桩实现，让整条链路（SSE 分帧、事件处理、刷新恢复、三选一）可以零成本反复跑。

用法::

    # 终端 1：假 LLM 服务端（前端开发时用这个，不要用真实后端）
    .\\.venv\\Scripts\\python.exe scripts\\dev_stub_server.py

    # 终端 2：前端
    cd web; npm run dev

判定逻辑刻意做成"看回答长度"：**≥40 字判通过，否则不通过**，
这样可以手动制造"未通过 → 重讲 → 重试耗尽 → 三选一"的完整路径。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts.schemas import LlmPurpose
from app.infra.llm.protocol import BlockingResult, StreamEvent, TokenUsage, ToolCall
from app.main import create_app
from app.services.learning.stack import build_learning_stack

TOPICS = ["定义与直觉", "核心公式", "典型应用"]
PASS_THRESHOLD = 40

TOPICS_JSON = json.dumps(
    {
        "topics": TOPICS,
        "first_question": "先用你自己的话说说：这个概念主要解决什么问题？",
    },
    ensure_ascii=False,
)

PROFILE_JSON = json.dumps(
    {
        "levels": [
            {"topic": TOPICS[0], "level": "basic", "evidence": "能说出大致用途，但用词不精确。"},
            {"topic": TOPICS[1], "level": "none", "evidence": "公式记不住，也没解释过每一项的含义。"},
            {"topic": TOPICS[2], "level": "none", "evidence": "举不出自己遇到的例子。"},
        ],
        "summary": "整体处在「听过但讲不清」的阶段，需要从定义与直觉重新搭一遍。",
        "suggested_start_node": "从定义与直觉开始，先把每个符号的含义说清楚。",
    },
    ensure_ascii=False,
)


def _path_json() -> str:
    nodes = []
    for index in range(1, 4):
        nodes.append(
            {
                "title": f"第 {index} 步：节点 {index}",
                "objective": f"能用自己的话讲清 节点 {index}，并说明它为什么必要。",
                "prerequisites": [] if index == 1 else [f"第 {index - 1} 步：节点 {index - 1}"],
                "key_points": ["要点一：它解决什么问题", "要点二：关键式子怎么来的", "要点三：什么时候会用到"],
                "checkpoints": [
                    {
                        "question": f"请用自己的话解释 节点 {index} 的关键点，并说明它在什么场景下有用。",
                        "expected_points": ["要点一", "要点二", "要点三"],
                        "reference_answer": "先把问题说清楚，再写出式子，最后给一个具体场景。",
                    }
                ],
                "is_quick_review": False,
            }
        )
    return json.dumps({"rationale": "按「先直觉、再公式、后应用」的顺序推进。", "nodes": nodes}, ensure_ascii=False)


EXPLANATION = """### 先用一句话说清

这个概念的意义是：**把新证据折算成对原有判断的修正**。

$$
P(A\\mid B)=\\frac{P(B\\mid A)\\,P(A)}{P(B)}
$$

- $P(A)$：先验，你原本的相信程度
- $P(B\\mid A)$：似然，若假设成立则看到证据的概率
- $P(B)$：证据本身出现的概率

```python
def posterior(prior, likelihood, evidence):
    return likelihood * prior / evidence
```

> 关键直觉：分母做归一化，让更新后的概率仍然是一个合法的概率。
"""

FEEDBACK = """这次回答里已经提到了「修正原有判断」这一层意思，但还差两点：

1. 没有说明**先验来自哪里**；
2. 没提**分母为什么必须归一化**。

换个角度再讲一遍：先想象你在判断"明天会不会下雨"，然后看到天气预报说有雨——这就是一次证据更新。
"""

REFERENCE = """**这是参考答案：**

把新证据折算成对原有判断的修正：$P(A\\mid B)=\\frac{P(B\\mid A)P(A)}{P(B)}$。
先验是你原本的相信程度，似然是"假设成立时看到该证据的概率"，分母是证据本身出现的总概率。
分母的作用是把结果压回 $[0,1]$，否则它就不再是概率。
"""

SUMMARY = "已掌握：能写式子并解释每一项的含义。注意：把「先验从哪来」讲清楚才算真正读懂。"

COMPRESSED = json.dumps({"compressed_context": "（压缩摘要）前若干轮已覆盖定义与公式。"}, ensure_ascii=False)


def _verdict(answer: str) -> str:
    passed = len(answer.strip()) >= PASS_THRESHOLD
    payload = {
        "verdict": "pass" if passed else "fail",
        "confidence": 0.9 if passed else 0.3,
        "feedback": "关键点都讲到了。" if passed else "还差关键的两点：先验来源与分母的作用。",
        "hit_points": ["把新证据折算成对原有判断的修正"] if passed else [],
        "missing_points": [] if passed else ["先验来自哪里", "分母为什么必须归一化"],
    }
    return json.dumps(payload, ensure_ascii=False)


ANSWER_MARKER = "学习者的回答："


def _extract_answer(system: str, fallback: str) -> str:
    """判定提示词把学习者作答写在 system 里（问答都在同一段文本中），单独取出来。"""
    index = system.find(ANSWER_MARKER)
    if index < 0:
        return fallback
    rest = system[index + len(ANSWER_MARKER) :]
    end = rest.find("\n\n## ")
    return (rest[:end] if end >= 0 else rest).strip() or fallback


# 只看提示词的**首行标题**来判定用途：正文里会提到别的模块名
# （例如路径规划提示词里就写着"掌握度画像"），整篇搜索必然误判。
HEADINGS: tuple[tuple[str, str], ...] = (
    ("推导子领域", "topics"),
    ("生成掌握度画像", "profile"),
    ("上下文压缩器", "compress"),
    ("判定器", "judge"),
    ("学习路径规划", "path"),
    ("未通过反馈", "reteach"),
    ("给出参考答案", "reference"),
    ("节点小结", "summary"),
    ("教学（讲解）", "explain"),
)


class StubClient:
    """按提示词内容返回固定内容；接口签名与 LlmClient 一致。"""

    def __init__(self) -> None:
        self.stream_calls = 0
        self.blocking_calls = 0
        self._last_answer = ""
        # 只演示一次工具调用：第二轮（带工具结果）必须回到正常正文，否则会无限循环
        self._tool_used = False

    async def aclose(self) -> None:
        return None

    def update_policy(self, *, max_retries: int | None = None, timeout_seconds: int | None = None) -> None:
        return None

    # ------------------------------------------------------------------ #

    @staticmethod
    def _system_text(messages: Sequence[Mapping[str, Any]]) -> str:
        for message in messages:
            if message.get("role") == "system":
                return str(message.get("content", ""))
        return ""

    @staticmethod
    def _last_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    def _route(self, messages: Sequence[Mapping[str, Any]], purpose: LlmPurpose) -> str:
        system = self._system_text(messages)
        heading = system.splitlines()[0] if system else ""
        for marker, route in HEADINGS:
            if marker in heading:
                return route
        if purpose is LlmPurpose.ASSESS:
            return "question"
        if purpose is LlmPurpose.JUDGE:
            return "judge"
        if purpose is LlmPurpose.PLAN:
            return "path"
        if purpose is LlmPurpose.COMPRESS:
            return "compress"
        return "explain"

    async def stream_chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        thinking: bool = True,
        reasoning_effort: str = "high",
        max_tokens: int = 8192,
        json_object: bool = False,
        purpose: LlmPurpose = LlmPurpose.TEACH,
        timeout_seconds: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del model, thinking, reasoning_effort, max_tokens, json_object, timeout_seconds
        self.stream_calls += 1
        route = self._route(messages, purpose)
        self._last_answer = self._last_user_text(messages)

        # 演示工具调用（FR-714）：只在"第一次带 tools 的教学请求"上触发，
        # 这样前端能看到"查档案 → 再讲解"的完整流程，又不会无限循环
        if tools and not self._tool_used:
            self._tool_used = True
            yield StreamEvent(tool_calls=(ToolCall(id="stub-call-1", name="read_archive", arguments="{}"),))
            yield StreamEvent(
                usage=TokenUsage(prompt_tokens=80, cached_prompt_tokens=0, cache_miss_tokens=80, completion_tokens=10)
            )
            return

        text = {
            "explain": EXPLANATION,
            "reteach": FEEDBACK,
            "reference": REFERENCE,
            "summary": SUMMARY,
            "question": "换个方面再问一个问题：你会在什么场景里用到它？",
        }.get(route, EXPLANATION)

        yield StreamEvent(reasoning="（桩服务的思维链：先看提示词属于哪一类，再给出固定内容。）")
        # 故意切成小片，让前端真正经历"流式"而不是一次性拿到整段
        for index in range(0, len(text), 24):
            await asyncio.sleep(0.02)
            yield StreamEvent(content=text[index : index + 24])
        yield StreamEvent(
            usage=TokenUsage(
                prompt_tokens=120, cached_prompt_tokens=40, cache_miss_tokens=80, completion_tokens=len(text) // 3
            )
        )

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        thinking: bool = True,
        reasoning_effort: str = "high",
        max_tokens: int = 8192,
        json_object: bool = False,
        purpose: LlmPurpose = LlmPurpose.TEACH,
        timeout_seconds: int | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> BlockingResult:
        del model, thinking, reasoning_effort, max_tokens, json_object, timeout_seconds, tools
        self.blocking_calls += 1
        route = self._route(messages, purpose)
        answer = _extract_answer(self._system_text(messages), self._last_user_text(messages))

        content = {
            "topics": TOPICS_JSON,
            "profile": PROFILE_JSON,
            "path": _path_json(),
            "judge": _verdict(answer),
            "compress": COMPRESSED,
        }.get(route)
        if content is None:
            content = TOPICS_JSON
        return BlockingResult(
            content=content,
            reasoning=None,
            usage=TokenUsage(prompt_tokens=200, cached_prompt_tokens=0, cache_miss_tokens=200, completion_tokens=80),
        )

    async def test_connection(self, *, model: str) -> Any:
        from app.infra.llm.client import ConnectionTestResult

        return ConnectionTestResult(ok=True, message="桩服务：连接正常", category=None, latency_ms=1)

    async def list_models(self) -> list[Any]:
        return []

    async def get_balance(self, *, timeout_seconds: int | None = None) -> Mapping[str, Any]:
        """桩服务的余额（FR-710）：字段形状与官方一致，金额保持字符串。"""
        del timeout_seconds
        return {
            "is_available": True,
            "balance_infos": [
                {
                    "currency": "CNY",
                    "total_balance": "88.80",
                    "granted_balance": "8.80",
                    "topped_up_balance": "80.00",
                }
            ],
        }


def main() -> None:
    import uvicorn

    data_dir = os.environ.get("WELEARN_STUB_DATA_DIR") or tempfile.mkdtemp(prefix="welearn-stub-")
    app = create_app({"WELEARN_DATA_DIR": data_dir})
    stub = StubClient()
    app.state.ctx.llm = stub
    app.state.stack = build_learning_stack(app.state.ctx)

    host = os.environ.get("WELEARN_HOST", "127.0.0.1")
    port = int(os.environ.get("WELEARN_STUB_PORT", "8000"))
    print(f"假 LLM 服务端：http://{host}:{port}  数据目录：{data_dir}")
    print("判定规则：回答 ≥40 字判通过，否则不通过（用来触发重讲与三选一）")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
