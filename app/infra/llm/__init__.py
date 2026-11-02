"""LLM 接入层：OpenAI 兼容协议、供应商能力档案、错误分类与 SSE 解析。"""

from app.infra.llm.client import ConnectionTestResult, LlmClient, LlmClientProtocol
from app.infra.llm.errors import LlmError, LlmErrorCategory
from app.infra.llm.protocol import BlockingResult, ModelInfo, StreamEvent, TokenUsage
from app.infra.llm.providers import ProviderKind, ProviderProfile, profile_for, provider_options

__all__ = [
    "BlockingResult",
    "ConnectionTestResult",
    "LlmClient",
    "LlmClientProtocol",
    "LlmError",
    "LlmErrorCategory",
    "ModelInfo",
    "ProviderKind",
    "ProviderProfile",
    "StreamEvent",
    "TokenUsage",
    "profile_for",
    "provider_options",
]
