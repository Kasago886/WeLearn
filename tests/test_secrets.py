"""API Key 存储测试（FR-707、FR-708、AC-19）。"""

import pytest
from cryptography.fernet import Fernet

from app.infra.redaction import Redactor
from app.infra.secrets import ENV_API_KEY, ENV_SECRET_KEY, SecretKeyError, SecretStore
from app.infra.storage.models import ApiKeySecretRow
from app.main_context import AppContext

KEY = "sk-abcdef1234567890"


def make_store(context: AppContext, env: dict[str, str] | None = None) -> tuple[SecretStore, Redactor]:
    redactor = Redactor()
    store = SecretStore(context.db, context.paths, redactor=redactor, env=env or {})
    return store, redactor


def test_set_get_and_mask(context: AppContext) -> None:
    store, _ = make_store(context)

    assert store.has_api_key() is False
    assert store.masked_hint() is None

    hint = store.set_api_key(KEY)

    assert hint == "sk-****7890"
    assert store.get_api_key() == KEY
    assert store.masked_hint() == hint
    assert store.source() == "db"


def test_ciphertext_is_not_plaintext(context: AppContext) -> None:
    store, _ = make_store(context)
    store.set_api_key(KEY)

    with context.db.session() as session:
        row = session.get(ApiKeySecretRow, 1)

    assert row is not None
    assert KEY not in row.cipher
    assert row.masked_hint == "sk-****7890"


def test_clearing_key(context: AppContext) -> None:
    store, _ = make_store(context)
    store.set_api_key(KEY)

    assert store.set_api_key("") is None
    assert store.get_api_key() is None
    assert store.source() == "none"


def test_key_file_is_created_and_reused(context: AppContext) -> None:
    first, _ = make_store(context)
    first.set_api_key(KEY)

    key_file = context.paths.root / "secret.key"
    assert key_file.exists()

    # 第二次启动（同一数据目录、无环境变量）必须能解开第一次写入的密文
    second, _ = make_store(context)
    assert second.get_api_key() == KEY


def test_env_key_takes_precedence(context: AppContext) -> None:
    store, _ = make_store(context, {ENV_API_KEY: "sk-env-key-000000"})
    store.set_api_key("sk-db-key-999999")

    assert store.get_api_key() == "sk-env-key-000000"
    assert store.source() == "env"
    assert store.masked_hint() == "sk-****0000"


def test_env_key_is_registered_for_redaction(context: AppContext) -> None:
    """环境变量里的明文也必须在脱敏名单里，否则会经日志外泄。"""
    _, redactor = make_store(context, {ENV_API_KEY: "sk-env-key-000000"})
    assert redactor.redact("Authorization: Bearer sk-env-key-000000") == "Authorization: Bearer ****"


def test_decrypted_key_is_registered_for_redaction(context: AppContext) -> None:
    store, redactor = make_store(context)
    store.set_api_key(KEY)
    assert store.get_api_key() == KEY
    assert KEY not in redactor.redact(f"key={KEY}")


def test_invalid_secret_key_raises_clear_error(context: AppContext) -> None:
    with pytest.raises(SecretKeyError):
        SecretStore(context.db, context.paths, env={ENV_SECRET_KEY: "not-a-fernet-key"})


def test_explicit_secret_key_from_env_is_used(context: AppContext) -> None:
    explicit = Fernet.generate_key().decode("utf-8")
    store = SecretStore(context.db, context.paths, env={ENV_SECRET_KEY: explicit})

    store.set_api_key(KEY)

    # 用同一密钥重新打开必须能解出原文
    reopened, _ = make_store(context, {ENV_SECRET_KEY: explicit})
    assert reopened.get_api_key() == KEY

    # 换一个密钥则解不开，说明确实用的是显式提供的密钥
    other, _ = make_store(context, {ENV_SECRET_KEY: Fernet.generate_key().decode("utf-8")})
    assert other.get_api_key() is None


def test_key_rotation_makes_old_cipher_unreadable(context: AppContext) -> None:
    """换密钥后旧密文无法解密：按"未配置"处理并给出可操作提示，而不是崩溃。"""
    store, _ = make_store(context)
    store.set_api_key(KEY)

    rotated, _ = make_store(context, {ENV_SECRET_KEY: Fernet.generate_key().decode("utf-8")})

    assert rotated.get_api_key() is None
    assert rotated.has_api_key() is False
    # 密文仍留在库里（未被静默清除），但已不可解读
    assert rotated.source() == "db"
