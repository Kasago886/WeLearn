"""异步日志写入（FR-904~FR-906）。

取自 ReferenceCode/LogManager.cs 的成熟设计：后台线程消费队列、写入失败有限重试、
重试仍失败则转入错误日志。**一处刻意偏离原实现**：原代码在队列满时抛异常，
而 FR-905 明确要求"队列溢出时不得使主流程崩溃"，因此这里改为丢弃并计数，
在队列恢复时补记一条统计日志。

日志属于旁路设施：任何写入异常都不得冒泡到调用方。
"""

import queue
import threading
import time
from datetime import datetime
from pathlib import Path

from app.infra.redaction import Redactor

CHANNEL_APP = "app"
CHANNEL_LLM = "llm"
CHANNEL_ERROR = "error"

_LEVEL_VALUES = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_DEFAULT_LEVEL = "INFO"
_MAX_WRITE_RETRIES = 3


class BoundLogger:
    """绑定到某个日志通道的写入入口。"""

    def __init__(self, logger: "QueueLogger | None", channel: str) -> None:
        self._logger = logger
        self._channel = channel

    def _write(self, level: str, message: str) -> None:
        if self._logger is not None:
            self._logger.log(self._channel, level, message)

    def debug(self, message: str) -> None:
        self._write("DEBUG", message)

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARNING", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)


class QueueLogger:
    """把日志投递到后台线程写盘。"""

    def __init__(
        self,
        log_dir: Path,
        *,
        level: str = _DEFAULT_LEVEL,
        max_queue: int = 1000,
        redactor: Redactor | None = None,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._redactor = redactor or Redactor()
        self._level_value = _LEVEL_VALUES.get(level.upper(), _LEVEL_VALUES[_DEFAULT_LEVEL])
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=max_queue)
        self._stopping = threading.Event()
        self._dropped = 0
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._process_queue, name="welearn-log", daemon=True)
        self._worker.start()

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    @property
    def dropped_count(self) -> int:
        """因队列溢出被丢弃的日志条数。"""
        with self._lock:
            return self._dropped

    def log(self, channel: str, level: str, message: str) -> None:
        """投递一条日志；任何情况下都不抛异常。"""
        level = level.upper()
        if _LEVEL_VALUES.get(level, _LEVEL_VALUES["INFO"]) < self._level_value:
            return

        try:
            self._queue.put_nowait((channel, self._format(level, message)))
        except queue.Full:
            with self._lock:
                self._dropped += 1

    def _format(self, level: str, message: str) -> str:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return f"{stamp} [{level}] {self._redactor.redact(message)}\n"

    def _process_queue(self) -> None:
        while True:
            try:
                channel, text = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stopping.is_set() and self._queue.empty():
                    return
                continue
            try:
                self._write_record(channel, text)
            finally:
                self._queue.task_done()

    def _write_record(self, channel: str, text: str) -> None:
        path = self._log_dir / f"{channel}.log"
        last_error: Exception | None = None
        for attempt in range(_MAX_WRITE_RETRIES):
            try:
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(text)
                self._report_dropped_if_any()
                return
            except OSError as exc:  # 目录被删、磁盘只读、句柄耗尽等
                last_error = exc
                time.sleep(0.05 * (attempt + 1))

        self._write_error_record(channel, path, text, last_error)

    def _report_dropped_if_any(self) -> None:
        with self._lock:
            dropped, self._dropped = self._dropped, 0
        if dropped:
            self._write_error_record(
                CHANNEL_ERROR,
                self._log_dir / f"{CHANNEL_ERROR}.log",
                f"{dropped} 条日志因队列溢出被丢弃（FR-905：不阻塞主流程）\n",
                None,
            )

    def _write_error_record(self, channel: str, path: Path, text: str, error: Exception | None) -> None:
        """最后兜底：把失败上下文写入错误日志；再失败就彻底放弃。"""
        reason = error if error is not None else "queue overflow"
        record = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [ERROR] 日志写入失败\n"
            f"channel: {channel}\ntarget: {path}\nreason: {type(reason).__name__}: {reason}\n"
            f"dropped payload:\n{text}\n"
        )
        try:
            with open(self._log_dir / f"{CHANNEL_ERROR}.log", "a", encoding="utf-8") as handle:
                handle.write(self._redactor.redact(record))
        except OSError:
            pass

    def flush(self, timeout: float = 2.0) -> bool:
        """等待队列排空，返回是否在超时前完成（测试与关停使用）。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._queue.empty():
                time.sleep(0.05)
                if self._queue.empty():
                    return True
            time.sleep(0.02)
        return False

    def close(self, timeout: float = 2.0) -> None:
        """停止后台线程并尽力排空队列。"""
        self.flush(timeout)
        self._stopping.set()
        self._worker.join(timeout=timeout)
        self._report_dropped_if_any()


_LOGGER: QueueLogger | None = None


def configure_logger(logger: QueueLogger) -> None:
    """注册全局日志器（应用启动时调用一次）。"""
    global _LOGGER
    _LOGGER = logger


def get_logger(channel: str) -> BoundLogger:
    """获取绑定到指定通道的日志器；未配置时退化为无操作，避免污染调用方。"""
    return BoundLogger(_LOGGER, channel)
