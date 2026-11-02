"""SSE 解析（FR-704）。

官方保活机制：等待期间**流式请求会持续下发 `: keep-alive` 注释**，非流式请求持续返回空行。
因此解析器必须忽略注释行、空行与其他字段行（`event:`/`id:`/`retry:`），
只把 `data:` 载荷交给上层，并以 `[DONE]` 作为结束标记。

按 SSE 规范，同一事件的多行 `data:` 以换行拼接；DeepSeek 实际只发单行，但这里按规范实现，
避免个别代理或网关的行为差异导致解析失败。

:class:`SseEventParser` 是唯一的解析实现：同步迭代器（:func:`iter_payloads`）与
异步流式读取（客户端）都通过它驱动，避免"测试过的解析器"与"线上跑的解析器"不一致。
"""

from collections.abc import Iterable, Iterator
from enum import StrEnum

DATA_PREFIX = "data:"
DONE_PAYLOAD = "[DONE]"


class SseLineKind(StrEnum):
    """单行的语义分类。"""

    IGNORE = "ignore"
    DATA = "data"
    DONE = "done"


def parse_line(raw_line: str | None) -> tuple[SseLineKind, str]:
    """解析一行，返回 (类型, 载荷)。

    - 空行 / 注释行（以 `:` 开头）/ 其他字段行 → :attr:`SseLineKind.IGNORE`
    - `data: [DONE]` → :attr:`SseLineKind.DONE`
    - 其余 `data:` 行 → :attr:`SseLineKind.DATA`
    """
    if raw_line is None:
        return SseLineKind.IGNORE, ""

    line = raw_line.strip()
    if not line:
        return SseLineKind.IGNORE, ""
    if not line.startswith(DATA_PREFIX):
        return SseLineKind.IGNORE, ""

    payload = line[len(DATA_PREFIX) :].strip()
    if not payload:
        return SseLineKind.IGNORE, ""
    if payload == DONE_PAYLOAD:
        return SseLineKind.DONE, ""
    return SseLineKind.DATA, payload


class SseEventParser:
    """增量式 SSE 解析器：逐行喂入，产出完整事件载荷。"""

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._finished = False

    @property
    def finished(self) -> bool:
        """是否已遇到 `[DONE]`。"""
        return self._finished

    def feed(self, raw_line: str | None) -> list[str]:
        """喂入一行，返回本次产生的事件载荷（通常为 0 或 1 个）。"""
        if self._finished:
            return []

        kind, payload = parse_line(raw_line)

        if kind is SseLineKind.DONE:
            self._finished = True
            return self._flush()

        if kind is SseLineKind.DATA:
            self._buffer.append(payload)
            return []

        # 空行代表一个事件的结束（注释行与字段行不会到达这里）
        if raw_line is not None and not raw_line.strip():
            return self._flush()

        return []

    def _flush(self) -> list[str]:
        if not self._buffer:
            return []
        joined = "\n".join(self._buffer)
        self._buffer = []
        return [joined]


def iter_payloads(lines: Iterable[str]) -> Iterator[str]:
    """把行序列转换为 data 载荷序列；遇到 `[DONE]` 立即结束。"""
    parser = SseEventParser()
    for raw_line in lines:
        yield from parser.feed(raw_line)
        if parser.finished:
            return
