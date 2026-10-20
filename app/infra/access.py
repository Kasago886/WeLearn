"""可选访问口令（FR-903）。

设计取舍

- **只存哈希**：口令是"能不能看学习数据"的凭据，与 API Key 同理，没有必要在库里留明文。
  用 SHA-256 而不是加密：我们只需要验证"是不是同一个口令"，不需要还原它。
- **比较用常量时间**：``hmac.compare_digest`` 避免按字节提前返回，虽然本机自托管下
  时序攻击的现实威胁很低，但这条成本几乎为零，没理由留个坏示范。
- **口令是可选能力**：默认关闭（空哈希）。开启后除了健康检查与口令校验本身，
  其余接口都需要带口令；否则"未通过则不提供任何学习数据"就是一句空话。
"""

import hashlib
import hmac


def hash_token(token: str) -> str:
    """返回口令的十六进制摘要；空串表示不设口令。"""
    cleaned = (token or "").strip()
    if not cleaned:
        return ""
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def token_matches(token: str | None, stored_hash: str | None) -> bool:
    """口令是否匹配。未设置口令时一律通过（功能默认关闭）。"""
    if not stored_hash:
        return True
    if not token:
        return False
    return hmac.compare_digest(hash_token(token), stored_hash)
