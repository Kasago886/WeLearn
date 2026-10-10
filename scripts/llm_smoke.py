"""真实 API 冒烟测试与响应样本录制（T0.5、T0.7）。

用法::

    python scripts/llm_smoke.py               # 冒烟：一次真实流式请求 + 用量输出
    python scripts/llm_smoke.py --record      # 额外把原始响应录到 tests/fixtures/llm/recorded/

退出码：0 成功 / 1 调用失败 / 2 未配置 API Key（便于 CI 区分"没配密钥"与"接口坏了"）。
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.contracts.schemas import LlmPurpose  # noqa: E402
from app.infra.deepseek.errors import LlmError  # noqa: E402
from app.infra.deepseek.protocol import (  # noqa: E402
    BALANCE_PATH,
    CHAT_COMPLETIONS_PATH,
    MODELS_PATH,
    build_chat_request,
    build_endpoint,
)
from app.main_context import AppContext, build_context  # noqa: E402

RECORD_DIR = PROJECT_ROOT / "tests" / "fixtures" / "llm" / "recorded"
PROMPT = "用两句话解释什么是贝叶斯定理。"
EXIT_OK = 0
EXIT_API_ERROR = 1
EXIT_NO_KEY = 2


def _headers(context: AppContext) -> dict[str, str]:
    api_key = context.secrets.get_api_key()
    if not api_key:
        raise SystemExit(EXIT_NO_KEY)
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


async def smoke(context: AppContext, *, model: str, show_reasoning: bool) -> int:
    """走一次真实流式请求，打印增量与用量。"""
    settings = context.settings
    print(f"base_url      : {settings.base_url}")
    print(f"model         : {model}")
    print(f"api_key       : {context.secrets.masked_hint()} (source={context.secrets.source()})")
    print(f"thinking      : {settings.thinking_enabled} effort={settings.reasoning_effort.value}")

    deltas = 0
    reasoning_chars = 0
    text_parts: list[str] = []
    usage = None

    try:
        async for event in context.llm.stream_chat(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            thinking=settings.thinking_enabled,
            reasoning_effort=settings.reasoning_effort.value,
            max_tokens=settings.max_tokens,
            purpose=LlmPurpose.TEACH,
        ):
            if event.is_usage_only:
                usage = event.usage
                continue
            deltas += 1
            if event.reasoning:
                reasoning_chars += len(event.reasoning)
                if show_reasoning and reasoning_chars <= 120:
                    print(f"  [thinking] {event.reasoning}", end="", flush=True)
            if event.content:
                text_parts.append(event.content)
                if deltas <= 40:
                    print(event.content, end="", flush=True)
    except LlmError as exc:
        print(f"\n调用失败: category={exc.category.value} message={exc.user_message}")
        if exc.detail:
            print(f"detail: {exc.detail}")
        return EXIT_API_ERROR

    print(f"\n\n正文: {''.join(text_parts)}")
    print(f"增量块数: {deltas}，思维链字符数: {reasoning_chars}")
    if usage is not None:
        print(
            "用量: "
            f"total={usage.total_tokens} prompt={usage.prompt_tokens} "
            f"cached={usage.cached_prompt_tokens} miss={usage.cache_miss_tokens} "
            f"completion={usage.completion_tokens} reasoning={usage.reasoning_tokens}"
        )
    else:
        print("警告: 本次响应未返回 usage")
    return EXIT_OK


async def record(context: AppContext, *, model: str) -> None:
    """录制原始响应样本，供离线回放（T0.7）。"""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    headers = _headers(context)
    settings = context.settings

    async with httpx.AsyncClient(timeout=120.0) as http:
        # 1) 流式（保留原始 SSE 行，含保活注释）
        body = build_chat_request(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            stream=True,
            thinking=settings.thinking_enabled,
            reasoning_effort=settings.reasoning_effort.value,
            max_tokens=settings.max_tokens,
        )
        lines: list[str] = []
        async with http.stream(
            "POST",
            build_endpoint(settings.base_url, CHAT_COMPLETIONS_PATH),
            json=body,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                lines.append(line)
        (RECORD_DIR / "stream_normal.sse").write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 2) 非流式
        blocking = await http.post(
            build_endpoint(settings.base_url, CHAT_COMPLETIONS_PATH),
            json=build_chat_request(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                stream=False,
                thinking=False,
                reasoning_effort=settings.reasoning_effort.value,
                max_tokens=512,
            ),
            headers=headers,
        )
        blocking.raise_for_status()
        (RECORD_DIR / "response_blocking.json").write_text(
            json.dumps(blocking.json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 3) JSON Output（结构化解耦，用于 FR-716 的空内容/截断验证）
        structured = await http.post(
            build_endpoint(settings.base_url, CHAT_COMPLETIONS_PATH),
            json=build_chat_request(
                model=model,
                messages=[
                    {"role": "system", "content": '只输出 JSON，格式为 {"answer": "..."} 的样例。'},
                    {"role": "user", "content": "把下面这句话压缩成 JSON：贝叶斯定理用于更新概率。"},
                ],
                stream=False,
                thinking=False,
                reasoning_effort=settings.reasoning_effort.value,
                max_tokens=512,
                json_object=True,
            ),
            headers=headers,
        )
        structured.raise_for_status()
        (RECORD_DIR / "response_json_object.json").write_text(
            json.dumps(structured.json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 4) 模型列表（FR-702）
        models = await http.get(build_endpoint(settings.base_url, MODELS_PATH), headers=headers)
        models.raise_for_status()
        (RECORD_DIR / "models.json").write_text(
            json.dumps(models.json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 5) 余额（FR-710）
        balance = await http.get(build_endpoint(settings.base_url, BALANCE_PATH), headers=headers)
        if balance.is_success:
            (RECORD_DIR / "balance.json").write_text(
                json.dumps(balance.json(), ensure_ascii=False, indent=2), encoding="utf-8"
            )

    (RECORD_DIR / "meta.json").write_text(
        json.dumps(
            {
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "base_url": settings.base_url,
                "thinking_enabled": settings.thinking_enabled,
                "reasoning_effort": settings.reasoning_effort.value,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"样本已录制到 {RECORD_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="WeLearn DeepSeek 冒烟测试")
    parser.add_argument("--record", action="store_true", help="额外录制原始响应样本")
    parser.add_argument("--model", default=None, help="覆盖模型名")
    parser.add_argument("--show-reasoning", action="store_true", help="打印思维链片段")
    args = parser.parse_args()

    load_dotenv()
    context = build_context()
    try:
        if not context.secrets.has_api_key():
            print("未配置 API Key：请在 .env 中设置 WELEARN_API_KEY，或在设置页录入。")
            return EXIT_NO_KEY

        model = args.model or context.settings.model_default
        code = asyncio.run(smoke(context, model=model, show_reasoning=args.show_reasoning))
        if code == EXIT_OK and args.record:
            asyncio.run(record(context, model=model))
        return code
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
