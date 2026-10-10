"""脱敏测试（NFR-108、FR-708）。"""

from app.infra.redaction import Redactor, mask_api_key


def test_mask_api_key_format() -> None:
    assert mask_api_key("sk-1234567890abcdef") == "sk-****cdef"
    assert mask_api_key(None) == ""
    assert mask_api_key("") == ""


def test_short_keys_are_fully_masked() -> None:
    """过短的 Key 不做部分保留，避免"掩码即泄露"。"""
    assert mask_api_key("sk-123456") == "****"
    assert mask_api_key("sk-1234567") == "****"


def test_redactor_replaces_registered_secret() -> None:
    redactor = Redactor()
    redactor.register("sk-real-secret-value")
    redactor.register("")  # 空白串被忽略

    assert redactor.redact("key=sk-real-secret-value") == "key=****"
    assert redactor.redact("无敏感内容") == "无敏感内容"
    assert redactor.redact(None) == ""


def test_redactor_catches_unregistered_sk_tokens() -> None:
    """兜底规则：即使忘记登记，也不能让 sk- 开头的串漏出去。"""
    redactor = Redactor()
    assert redactor.redact("Authorization: Bearer sk-abcdef123456") == "Authorization: Bearer ****"


def test_truncate_redacts_then_truncates() -> None:
    redactor = Redactor()
    redactor.register("sk-secret")
    text = "sk-secret " + "x" * 100

    result = redactor.truncate(text, limit=20)
    assert "sk-secret" not in result
    assert result.endswith("…")
    assert len(result) == 21
