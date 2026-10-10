"""应用级依赖容器（T0.1）。

把基础设施对象的构造与生命周期集中在这里，使 :mod:`app.main` 保持可读，
测试也能用同一个 :func:`build_context` 在临时目录上装配一套完整的依赖。
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from app.infra.deepseek.client import DeepSeekClient
from app.infra.logging.queue_logger import CHANNEL_APP, QueueLogger, configure_logger, get_logger
from app.infra.paths import DataPaths
from app.infra.redaction import Redactor
from app.infra.secrets import SecretStore
from app.infra.settings import AppSettings, SettingsService, resolve_data_dir
from app.infra.storage.db import Database

DATABASE_FILENAME = "welearn.db"
LOG_DIR_NAME = "logs"

ENV_LOG_PAYLOAD = "WELEARN_LOG_LLM_PAYLOAD"


@dataclass
class AppContext:
    """一次进程内的全部基础设施依赖。

    设置快照放在可变的 ``settings_holder`` 中：LLM 客户端的 getter 读 holder，
    因此 :meth:`refresh_settings` 能让设置改动即时生效，无需重建客户端。
    """

    paths: DataPaths
    db: Database
    schema_version: int
    settings_service: SettingsService
    settings_holder: dict[str, AppSettings]
    secrets: SecretStore
    redactor: Redactor
    logger: QueueLogger
    llm: DeepSeekClient
    env: Mapping[str, str]

    @property
    def settings(self) -> AppSettings:
        """当前生效的设置快照。"""
        return self.settings_holder["current"]

    def refresh_settings(self, settings: AppSettings) -> None:
        """设置页保存后刷新内存快照，并同步客户端的重试与超时策略。"""
        self.settings_holder["current"] = settings
        self.llm.update_policy(
            max_retries=settings.max_retries,
            timeout_seconds=settings.timeout_seconds,
        )

    def close(self) -> None:
        """释放资源：先排空日志，再关连接池。"""
        self.logger.close()
        self.db.dispose()


def build_context(env: Mapping[str, str] | None = None) -> AppContext:
    """按环境装配上下文。

    启动顺序有意固定：数据目录 → 数据库 → 设置 → 日志 → 密钥 → LLM 客户端。
    """
    source = env if env is not None else os.environ

    paths = DataPaths(resolve_data_dir(source))
    paths.ensure_root()

    db = Database.from_path(paths.resolve(DATABASE_FILENAME))
    schema_version = db.init()

    redactor = Redactor()
    settings_service = SettingsService(db, source)
    settings = settings_service.ensure_defaults()

    logger = QueueLogger(paths.resolve(LOG_DIR_NAME), level=settings.log_level, redactor=redactor)
    configure_logger(logger)
    get_logger(CHANNEL_APP).info(
        f"配置加载完成 level={settings.log_level} data_dir={paths.root} schema_version={schema_version}"
    )

    secrets = SecretStore(db, paths, redactor=redactor, env=source)

    log_payload = (source.get(ENV_LOG_PAYLOAD) or "").strip().lower() in {"1", "true", "yes", "on"}

    # 客户端通过 holder 读取设置，保证运行期修改即时生效
    holder: dict[str, AppSettings] = {"current": settings}
    llm = DeepSeekClient(
        base_url_getter=lambda: holder["current"].base_url,
        api_key_getter=secrets.get_api_key,
        redactor=redactor,
        max_retries=settings.max_retries,
        timeout_seconds=settings.timeout_seconds,
        log_payload=log_payload,
    )

    return AppContext(
        paths=paths,
        db=db,
        schema_version=schema_version,
        settings_service=settings_service,
        settings_holder=holder,
        secrets=secrets,
        redactor=redactor,
        logger=logger,
        llm=llm,
        env=source,
    )
