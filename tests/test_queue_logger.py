"""日志设施测试（FR-904~FR-906、AC-24）。"""

import time
from pathlib import Path

from app.infra.logging.queue_logger import (
    CHANNEL_APP,
    CHANNEL_ERROR,
    BoundLogger,
    QueueLogger,
    get_logger,
)
from app.infra.redaction import Redactor


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_writes_to_channel_file(tmp_path: Path) -> None:
    logger = QueueLogger(tmp_path, level="DEBUG")
    try:
        logger.log(CHANNEL_APP, "INFO", "服务已启动")
        assert logger.flush()
    finally:
        logger.close()

    text = (tmp_path / f"{CHANNEL_APP}.log").read_text(encoding="utf-8")
    assert "[INFO] 服务已启动" in text
    # 统一时间戳格式，便于按时间排查
    assert text[:4].isdigit()


def test_level_filtering(tmp_path: Path) -> None:
    logger = QueueLogger(tmp_path, level="WARNING")
    try:
        logger.log(CHANNEL_APP, "DEBUG", "调试细节")
        logger.log(CHANNEL_APP, "INFO", "普通信息")
        logger.log(CHANNEL_APP, "ERROR", "严重错误")
        assert logger.flush()
    finally:
        logger.close()

    text = (tmp_path / f"{CHANNEL_APP}.log").read_text(encoding="utf-8")
    assert "调试细节" not in text
    assert "普通信息" not in text
    assert "严重错误" in text


def test_redaction_is_applied(tmp_path: Path) -> None:
    redactor = Redactor()
    redactor.register("sk-super-secret")
    logger = QueueLogger(tmp_path, level="DEBUG", redactor=redactor)
    try:
        logger.log(CHANNEL_APP, "INFO", "key=sk-super-secret")
        assert logger.flush()
    finally:
        logger.close()

    text = (tmp_path / f"{CHANNEL_APP}.log").read_text(encoding="utf-8")
    assert "sk-super-secret" not in text
    assert "key=****" in text


def test_queue_overflow_does_not_raise(tmp_path: Path) -> None:
    """FR-905：队列溢出不得使主流程崩溃（此处刻意偏离 ReferenceCode 的抛异常实现）。"""
    (tmp_path / f"{CHANNEL_APP}.log").mkdir()  # 让写入持续失败以拖慢消费者

    logger = QueueLogger(tmp_path, level="DEBUG", max_queue=2)
    try:
        for index in range(200):
            logger.log(CHANNEL_APP, "ERROR", f"消息 {index}")

        # 丢弃计数会在 close() 时被"消费"成一条统计日志，因此先取值再关闭
        dropped = logger.dropped_count
    finally:
        logger.close()

    assert dropped >= 1


def test_write_failure_lands_in_error_log(tmp_path: Path) -> None:
    (tmp_path / f"{CHANNEL_APP}.log").mkdir()  # 目标路径是目录 → 写入必然失败

    logger = QueueLogger(tmp_path, level="DEBUG")
    try:
        logger.log(CHANNEL_APP, "ERROR", "这条会因为写入失败而转存")
        error_log = tmp_path / f"{CHANNEL_ERROR}.log"
        assert _wait_for(lambda: error_log.exists() and "日志写入失败" in error_log.read_text(encoding="utf-8"))
    finally:
        logger.close()


def test_flush_is_a_real_barrier(tmp_path: Path) -> None:
    """flush() 返回 True 必须意味着已经落盘。

    后台线程是"先取条目、再落盘"的，所以"队列空"并不等于"写完"。这里刻意让写入持续失败
    （重试共耗 0.05+0.10+0.15s，远长于取队时间），好把两种判据区分开：拿队列空当判据会
    在 0.05s 后就虚报成功，而那时错误日志还没写出来。
    """
    (tmp_path / f"{CHANNEL_APP}.log").mkdir()

    logger = QueueLogger(tmp_path, level="DEBUG")
    try:
        logger.log(CHANNEL_APP, "ERROR", "这条会因为写入失败而转存")

        assert logger.flush()

        # flush 之后不再等待：内容必须已经在错误日志里
        assert "日志写入失败" in (tmp_path / f"{CHANNEL_ERROR}.log").read_text(encoding="utf-8")
    finally:
        logger.close()


def test_close_drains_queued_messages(tmp_path: Path) -> None:
    """close() 之前投递的日志一条都不能丢（FR-906）。"""
    logger = QueueLogger(tmp_path, level="DEBUG")
    for index in range(50):
        logger.log(CHANNEL_APP, "INFO", f"消息 {index}")
    logger.close()

    text = (tmp_path / f"{CHANNEL_APP}.log").read_text(encoding="utf-8")
    assert all(f"消息 {index}" in text for index in range(50))


def test_get_logger_without_configuration_is_noop() -> None:
    """未配置时不抛异常、不写入，避免污染调用方（NFR-107）。"""
    logger = get_logger("app")
    assert isinstance(logger, BoundLogger)
    logger.info("丢弃即可")
