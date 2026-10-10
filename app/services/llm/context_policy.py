"""上下文成本策略（T1.7，FR-719、FR-720）。

两条官方规则的落地：

**FR-719 思维链回传**：请求**携带 `tools`** 时，历史轮次的 `reasoning_content` 必须回传并会被
拼进上下文；不携带 `tools` 时官方会忽略它。因此这里按是否启用工具决定"保留还是剔除"——
剔除不改变模型看到的内容，但能显著减小请求体（思维链通常比正文长得多）。

**FR-720 缓存友好**：官方上下文硬盘缓存只有**完整匹配缓存前缀单元**才计入命中，
因此同一会话的历史消息前缀必须稳定。本模块提供 :func:`stable_prefix_hash`，
用测试守住"我们不会逐轮改写历史消息"这一前提。
"""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from app.services.llm.memory import REASONING_KEY, Memory

ProtocolMessage = dict[str, Any]


def strip_reasoning(messages: Sequence[ProtocolMessage]) -> list[ProtocolMessage]:
    """剔除历史消息中的 ``reasoning_content``（不携带 tools 时的策略）。"""
    stripped: list[ProtocolMessage] = []
    for message in messages:
        if REASONING_KEY in message:
            message = {key: value for key, value in message.items() if key != REASONING_KEY}
        stripped.append(message)
    return stripped


def build_outbound_messages(memory: Memory, *, tools_enabled: bool = False) -> list[ProtocolMessage]:
    """构造发给模型的消息数组。

    ``tools_enabled=True`` 时保留 ``reasoning_content``（官方要求回传），
    否则剔除（官方会忽略，剔除只为减小请求体）。
    """
    messages = memory.outbound_messages()
    return messages if tools_enabled else strip_reasoning(messages)


def stable_prefix_hash(messages: Sequence[ProtocolMessage]) -> str:
    """历史消息前缀的内容指纹。

    用途是**守住缓存友好性**：同一份记忆在相邻两轮之间，除新增部分外前缀必须逐字节一致，
    否则缓存必然失效（表现为 ``prompt_cache_hit_tokens`` 长期为 0）。
    """
    payload = json.dumps(list(messages), ensure_ascii=False, sort_keys=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reasoning_payload_size(messages: Sequence[ProtocolMessage]) -> int:
    """历史消息中思维链的字符总量，用于观察上下文膨胀（FR-719）。"""
    total = 0
    for message in messages:
        value = message.get(REASONING_KEY)
        if isinstance(value, str):
            total += len(value)
    return total


def estimate_context_pressure(messages: Sequence[ProtocolMessage], threshold_rounds: int) -> bool:
    """粗略判断上下文是否接近需要压缩的程度。

    只做"是否需要触发压缩检查"的提示，真正的判断依据仍是轮次阈值（FR-604）——
    不同模型上下文窗口差异很大，按字符数硬编码阈值反而会误判。
    """
    return len(messages) >= threshold_rounds
