"""DeepSeek 接入层：错误分类、SSE 解析、协议构建与客户端。"""

from app.infra.deepseek.client import ConnectionTestResult, DeepSeekClient, LlmClientProtocol
from app.infra.deepseek.errors import LlmError, LlmErrorCategory
from app.infra.deepseek.protocol import BlockingResult, ModelInfo, StreamEvent, TokenUsage

__all__ = [
    "BlockingResult",
    "ConnectionTestResult",
    "DeepSeekClient",
    "LlmClientProtocol",
    "LlmError",
    "LlmErrorCategory",
    "ModelInfo",
    "StreamEvent",
    "TokenUsage",
]
