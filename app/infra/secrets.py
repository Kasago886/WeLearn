"""API Key 管理（FR-707、FR-708）。

- 优先级：环境变量 `WELEARN_API_KEY` > 数据库密文（FR-707）；
- 静态加密：Fernet（AES-CBC + HMAC），密钥来自环境变量或数据目录下的 secret.key（0600）；
- 明文永不落库、永不进日志、永不出现在响应里；界面只回显掩码（FR-708）；
- 解密密文失败（换过密钥）时不崩溃，按"未配置"处理并记录日志，由用户重新录入。
"""

import contextlib
import os
import stat
from collections.abc import Mapping
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import delete, select

from app.infra.logging.queue_logger import CHANNEL_APP, get_logger
from app.infra.paths import DataPaths
from app.infra.redaction import Redactor, mask_api_key
from app.infra.storage.db import Database
from app.infra.storage.models import ApiKeySecretRow

ENV_API_KEY = "WELEARN_API_KEY"
ENV_SECRET_KEY = "WELEARN_SECRET_KEY"

SECRET_KEY_FILENAME = "secret.key"


class SecretKeyError(RuntimeError):
    """密钥配置错误（例如 WELEARN_SECRET_KEY 不是合法 Fernet 密钥）。"""


class SecretStore:
    """API Key 的加密存储与读取。"""

    def __init__(
        self,
        db: Database,
        paths: DataPaths,
        *,
        redactor: Redactor | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._db = db
        self._paths = paths
        self._env = env if env is not None else os.environ
        self._redactor = redactor if redactor is not None else Redactor()
        self._log = get_logger(CHANNEL_APP)
        self._fernet = Fernet(self._load_or_create_key())

        env_key = self._env_api_key()
        if env_key:
            # 环境变量里的明文也必须登记，避免经日志外泄
            self._redactor.register(env_key)
            self._log.info("已从环境变量 WELEARN_API_KEY 加载 API Key（环境变量优先于界面设置）")

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #

    def get_api_key(self) -> str | None:
        """返回生效的 API Key 明文；未配置时返回 None。"""
        env_key = self._env_api_key()
        if env_key:
            return env_key

        row = self._read_row()
        if row is None:
            return None

        try:
            plain = self._fernet.decrypt(row.cipher.encode("utf-8")).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError):
            self._log.error("API Key 解密失败（密钥可能已更换），请重新录入")
            return None

        if not plain.strip():
            return None
        self._redactor.register(plain)
        return plain

    def has_api_key(self) -> bool:
        """是否已配置可用 Key。"""
        return bool(self.get_api_key())

    def set_api_key(self, api_key: str | None) -> str | None:
        """保存（或清除）API Key，返回掩码提示串。

        传入空串或 None 表示清除配置。
        """
        if self._env_api_key():
            self._log.warning("环境变量 WELEARN_API_KEY 已设置，界面保存的 Key 不会生效（环境变量优先）")

        value = (api_key or "").strip()
        if not value:
            with self._db.session() as session:
                session.execute(delete(ApiKeySecretRow).where(ApiKeySecretRow.id == 1))
            self._log.info("API Key 已清除")
            return None

        cipher = self._fernet.encrypt(value.encode("utf-8")).decode("utf-8")

        with self._db.session() as session:
            row = session.get(ApiKeySecretRow, 1) or ApiKeySecretRow(id=1)
            row.cipher = cipher
            row.masked_hint = mask_api_key(value)
            row.source = "db"
            session.add(row)

        # 只记录掩码，明文不落日志
        self._log.info(f"API Key 已更新（{mask_api_key(value)}）")
        return mask_api_key(value)

    def masked_hint(self) -> str | None:
        """界面回显用的掩码串。"""
        env_key = self._env_api_key()
        if env_key:
            return mask_api_key(env_key)
        row = self._read_row()
        return row.masked_hint if row is not None else None

    def source(self) -> str:
        """当前 Key 的来源：`env` 或 `db`，未配置时返回 `none`。"""
        if self._env_api_key():
            return "env"
        return "db" if self._read_row() is not None else "none"

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #

    def _env_api_key(self) -> str | None:
        raw = (self._env.get(ENV_API_KEY) or "").strip()
        return raw or None

    def _read_row(self) -> ApiKeySecretRow | None:
        with self._db.session() as session:
            return session.scalar(select(ApiKeySecretRow).where(ApiKeySecretRow.id == 1))

    def _load_or_create_key(self) -> bytes:
        env_key = (self._env.get(ENV_SECRET_KEY) or "").strip()
        if env_key:
            try:
                Fernet(env_key.encode("utf-8"))
            except (ValueError, TypeError) as exc:
                raise SecretKeyError(
                    f"{ENV_SECRET_KEY} 不是合法的 Fernet 密钥（应为 44 字符 urlsafe base64）"
                ) from exc
            return env_key.encode("utf-8")

        key_path = self._paths.resolve(SECRET_KEY_FILENAME)
        if key_path.exists():
            return key_path.read_bytes().strip()

        key = Fernet.generate_key()
        self._paths.ensure_root()
        key_path.write_bytes(key)
        self._restrict_permissions(key_path)
        self._log.info(f"已生成新的加密密钥文件: {key_path}")
        return key

    @staticmethod
    def _restrict_permissions(path: Path) -> None:
        """尽力收紧权限（Windows 下仅有只读位语义，失败不致命）。"""
        with contextlib.suppress(OSError):
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
