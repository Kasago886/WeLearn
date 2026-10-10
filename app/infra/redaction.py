"""敏感信息脱敏（NFR-108）。

两条防线：
1. **登记式**：把已知密钥交给 :class:`Redactor`，任何输出前的文本都会被替换为掩码；
2. **兜底式**：即使密钥未被登记，也按 `sk-` 前缀识别并掩码，避免"漏登记即泄露"。

日志、错误响应体、前端可访问的字段都必须经过本模块。
"""

import re

_MASK = "****"

# 兜底规则：DeepSeek 及常见 OpenAI 兼容平台的 Key 前缀
_FALLBACK_PATTERN = re.compile(r"\b(sk-[A-Za-z0-9_\-]{6,})")


def mask_api_key(api_key: str | None) -> str:
    """生成掩码串，用于界面回显（FR-708）。

    过短的 Key 一律整体掩码，避免"掩码即泄露"。
    """
    if not api_key:
        return ""
    key = api_key.strip()
    if len(key) <= 10:
        return _MASK
    return f"{key[:3]}{_MASK}{key[-4:]}"


class Redactor:
    """维护需脱敏的敏感串集合。"""

    def __init__(self) -> None:
        self._secrets: set[str] = set()

    def register(self, secret: str | None) -> None:
        """登记敏感串；空白串会被忽略。"""
        if secret and secret.strip():
            self._secrets.add(secret.strip())

    def register_all(self, secrets: list[str | None] | tuple[str | None, ...]) -> None:
        for secret in secrets:
            self.register(secret)

    def redact(self, text: str | None) -> str:
        """替换文本中的所有敏感串。"""
        if not text:
            return ""
        result = text
        for secret in self._secrets:
            if secret in result:
                result = result.replace(secret, _MASK)
        return _FALLBACK_PATTERN.sub(_MASK, result)

    def truncate(self, text: str | None, limit: int = 500) -> str:
        """脱敏并截断，用于日志与错误响应体。"""
        redacted = self.redact(text)
        return redacted if len(redacted) <= limit else redacted[:limit] + "…"
