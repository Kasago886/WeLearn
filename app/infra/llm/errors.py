"""LLM 错误分类（FR-709、FR-712）。

目标：把 HTTP 状态码 / 网络异常翻译成"发生了什么 + 能做什么 + 能不能重试"三件事，
使上层不必再判断状态码。重试策略完全由 :attr:`LlmError.retryable` 决定。

状态码语义取自 OpenAI 兼容接口的通用约定（DeepSeek 官方的错误码表与之一致）：
400 格式错误 / 401 认证失败 / 402 余额不足 / 422 参数错误 / 429 速率上限 / 500 服务故障 / 503 服务繁忙。
文案不写死供应商名字——同一份代码可能连的是 DeepSeek、本地 vLLM 或某个网关。
"""

from enum import StrEnum


class LlmErrorCategory(StrEnum):
    """错误分类。"""

    CONFIG = "config"
    BAD_REQUEST = "bad_request"
    AUTH = "auth"
    BALANCE = "balance"
    INVALID_PARAMS = "invalid_params"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    UNAVAILABLE = "unavailable"
    NETWORK = "network"
    TIMEOUT = "timeout"
    PROTOCOL = "protocol"
    EMPTY_CONTENT = "empty_content"
    UNKNOWN = "unknown"


_STATUS_TABLE: dict[int, tuple[LlmErrorCategory, bool, str]] = {
    400: (LlmErrorCategory.BAD_REQUEST, False, "请求体格式错误，请检查模型名与消息结构后重试。"),
    401: (LlmErrorCategory.AUTH, False, "API Key 无效或未授权，请在设置中更新 Key。"),
    402: (LlmErrorCategory.BALANCE, False, "账户余额不足，请充值后重试。"),
    422: (LlmErrorCategory.INVALID_PARAMS, False, "请求参数错误，请检查模型名与参数取值。"),
    429: (LlmErrorCategory.RATE_LIMIT, True, "请求速率达到上限，稍后会自动重试。"),
    500: (LlmErrorCategory.SERVER_ERROR, True, "模型服务故障，稍后会自动重试。"),
    503: (LlmErrorCategory.UNAVAILABLE, True, "模型服务繁忙，稍后会自动重试。"),
}


class LlmError(Exception):
    """一次 LLM 调用失败。"""

    def __init__(
        self,
        category: LlmErrorCategory,
        user_message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        detail: str = "",
    ) -> None:
        super().__init__(user_message)
        self.category = category
        self.user_message = user_message
        self.retryable = retryable
        self.status_code = status_code
        self.detail = detail

    @classmethod
    def from_status(cls, status_code: int, detail: str = "") -> "LlmError":
        """按官方错误码表构造；未知状态码按 4xx/5xx 归类。"""
        mapped = _STATUS_TABLE.get(status_code)
        if mapped is None:
            if 500 <= status_code < 600:
                mapped = (LlmErrorCategory.SERVER_ERROR, True, "服务端返回错误，稍后会自动重试。")
            elif 400 <= status_code < 500:
                mapped = (LlmErrorCategory.UNKNOWN, False, "请求被服务端拒绝，请检查配置后重试。")
            else:
                mapped = (LlmErrorCategory.UNKNOWN, False, "服务端返回未知状态，请查看日志。")

        category, retryable, message = mapped
        return cls(category, message, retryable=retryable, status_code=status_code, detail=detail)

    @classmethod
    def config(cls, message: str) -> "LlmError":
        return cls(LlmErrorCategory.CONFIG, message, retryable=False)

    @classmethod
    def timeout(cls, detail: str = "") -> "LlmError":
        return cls(
            LlmErrorCategory.TIMEOUT,
            "请求超时，可点击重试或调大超时时间。",
            retryable=True,
            detail=detail,
        )

    @classmethod
    def network(cls, detail: str = "") -> "LlmError":
        return cls(
            LlmErrorCategory.NETWORK,
            "网络连接中断，请检查网络后重试。",
            retryable=True,
            detail=detail,
        )

    @classmethod
    def protocol(cls, message: str, detail: str = "") -> "LlmError":
        """响应结构不符合预期（官方 10 分钟未开始推理即断开也归入此类观测）。"""
        return cls(LlmErrorCategory.PROTOCOL, message, retryable=False, detail=detail)

    @classmethod
    def empty_content(cls, detail: str = "") -> "LlmError":
        """只输出了思维链、正文为空（可重试）。

        思考模式下模型有概率把输出预算全花在思维链上，正文一个字都不给。实测与
        ``max_tokens`` 无关（``completion`` 远未顶到上限时也会发生），因此重试是有效的。
        """
        return cls(
            LlmErrorCategory.EMPTY_CONTENT,
            "模型这次只给出了思考过程、没有给出正文，已自动重试。请再点一次。",
            retryable=True,
            detail=detail,
        )

    def to_payload(self) -> dict[str, object]:
        """转成 SSE 错误事件的载荷（FR-906：不含敏感信息）。"""
        return {"category": self.category.value, "message": self.user_message, "retryable": self.retryable}
