"""性能校验（T5.2、NFR-101~NFR-103）。

为什么不放进 pytest：这里量的是**时间**，跑在共享机器上必然抖动；把它们混进单测，
要么阈值宽到测不出问题，要么偶发失败让人习惯性重跑。这里做成脚本，
把实测数字写进验证记录，只在"明显越界"时退出非 0。

用假 LLM 起一个**真实** uvicorn（不是 TestClient）：测的首 token 延迟里要包含
HTTP/SSE 这一层，而 TestClient 会绕过网络栈，量出来的数字没有参考价值。
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import create_app  # noqa: E402
from app.services.learning.stack import build_learning_stack  # noqa: E402

TTFT_BUDGET_MS = 3000
LOCAL_BUDGET_MS = 300
LOCAL_SAMPLES = 30

LONG_ANSWER = (
    "我的理解是：它把新证据折算成对原有判断的修正，先验来自以往的经验统计，"
    "分母做归一化，让更新后的结果仍然是一个合法的概率。"
)


def load_stub() -> ModuleType:
    """复用联调用的假 LLM，避免这里再维护一套"什么提示词返回什么"的规则。"""
    path = ROOT / "scripts" / "dev_stub_server.py"
    spec = importlib.util.spec_from_file_location("welearn_dev_stub", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_server(port: int) -> tuple[uvicorn.Server, threading.Thread, Path]:
    data_dir = Path(tempfile.mkdtemp(prefix="welearn-perf-"))
    app = create_app({"WELEARN_DATA_DIR": str(data_dir)})
    stub = load_stub().StubClient()
    app.state.ctx.llm = stub
    app.state.stack = build_learning_stack(app.state.ctx)

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="perf-server", daemon=True)
    thread.start()

    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=0.5)
            break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("服务未能在 20 秒内就绪")
    return server, thread, data_dir


def read_sse(client: httpx.Client, url: str, payload: dict[str, Any]) -> tuple[float | None, str]:
    """POST 一个流式端点，返回（首个正文 token 的耗时毫秒, 完整正文）。"""
    started = time.perf_counter()
    first_token_ms: float | None = None
    text: list[str] = []
    with client.stream("POST", url, json=payload, timeout=60) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[len("data:") :].strip()
            if not raw:
                continue
            try:
                data = __import__("json").loads(raw)
            except ValueError:
                continue
            chunk = data.get("text")
            if isinstance(chunk, str) and chunk:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                text.append(chunk)
    return first_token_ms, "".join(text)


def measure(client: httpx.Client, url: str, samples: int) -> tuple[float, float]:
    """返回（p50, p95）毫秒。"""
    timings: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        response = client.get(url)
        response.raise_for_status()
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    p50 = statistics.median(timings)
    p95 = timings[min(len(timings) - 1, int(len(timings) * 0.95))]
    return p50, p95


def main() -> int:
    parser = argparse.ArgumentParser(description="WeLearn 性能校验")
    parser.add_argument("--port", type=int, default=8131)
    args = parser.parse_args()

    server, thread, data_dir = start_server(args.port)
    base = f"http://127.0.0.1:{args.port}"
    failures: list[str] = []
    report: list[str] = []

    try:
        with httpx.Client(base_url=base, timeout=60) as client:
            created = client.post("/api/subjects", json={"title": "性能校验主题"})
            created.raise_for_status()
            subject_id = created.json()["subject"]["id"]

            client.post(f"/api/subjects/{subject_id}/assessment/skip").raise_for_status()

            # 路径生成是流式端点，但它推送的是 `path` 事件而不是正文 token；
            # 因此这里只关心链路是否走通，结果用 /path 读回验证
            read_sse(client, f"/api/subjects/{subject_id}/path/generate", {})
            path_view = client.get(f"/api/subjects/{subject_id}/path")
            path_view.raise_for_status()
            if path_view.json().get("path") is None:
                failures.append("路径生成后 /path 仍为空")

            client.post(f"/api/subjects/{subject_id}/path/confirm").raise_for_status()

            ttfts: list[float] = []
            for _ in range(3):
                ttft, body = read_sse(client, f"/api/subjects/{subject_id}/teach/start", {})
                if ttft is None or not body.strip():
                    failures.append("教学回合没有收到正文 token")
                    break
                ttfts.append(ttft)

            if ttfts:
                worst = max(ttfts)
                report.append(
                    f"首 token 延迟：p50={statistics.median(ttfts):.0f}ms，max={worst:.0f}ms"
                    f"（预算 {TTFT_BUDGET_MS}ms，假 LLM，不含厂商排队）"
                )
                if worst > TTFT_BUDGET_MS:
                    failures.append(f"首 token 延迟 {worst:.0f}ms 超出预算 {TTFT_BUDGET_MS}ms")

            endpoints = {
                "主题列表": "/api/subjects",
                "状态快照": f"/api/subjects/{subject_id}/state",
                "教学页数据": f"/api/subjects/{subject_id}/teach/current",
                "档案总览": f"/api/subjects/{subject_id}/archive",
                "全局检索": "/api/search?q=性能",
            }
            for label, url in endpoints.items():
                p50, p95 = measure(client, url, LOCAL_SAMPLES)
                report.append(f"{label}：p50={p50:.1f}ms，p95={p95:.1f}ms（预算 {LOCAL_BUDGET_MS}ms）")
                if p95 > LOCAL_BUDGET_MS:
                    failures.append(f"{label} p95 {p95:.1f}ms 超出预算 {LOCAL_BUDGET_MS}ms")
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    print("== 实测 ==")
    for line in report:
        print("  " + line)
    print(f"  数据目录：{data_dir}")

    if failures:
        print("\n== 未达标 ==")
        for item in failures:
            print("  - " + item)
        return 1
    print("\n全部在预算内。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
