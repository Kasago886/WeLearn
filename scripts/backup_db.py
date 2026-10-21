"""数据库在线备份、恢复与校验（T5.5、NFR-106、D3）。

用法::

    python scripts/backup_db.py backup  [--data-dir DIR]
    python scripts/backup_db.py restore --file BACKUP [--data-dir DIR]
    python scripts/backup_db.py verify  [--file BACKUP] [--data-dir DIR]

数据目录与主程序同源：``--data-dir`` > 环境变量 ``WELEARN_DATA_DIR`` > ``./data``（D3：
数据放在项目外可配置目录，因此备份也跟着数据目录走，整体搬迁/升级时不会漏掉备份）。

**为什么必须用 SQLite 在线备份 API，而不能 ``shutil.copy`` 主库文件**：
WAL 模式下，已提交但尚未 checkpoint 的事务只写在 ``welearn.db-wal`` 里，主库文件本身
还是"旧"的。只复制主库文件，轻则丢掉最近提交的数据——副本仍能通过 ``integrity_check``
（页面自洽），但内容不是最新的；重则恰好撞上 checkpoint，拿到主库与 WAL 撕裂的组合。
:meth:`sqlite3.Connection.backup` 在源连接上按页读取，得到的是某一时刻的**一致视图**，
并且会把 WAL 中已提交的内容一并带走，所以运行中的库也能安全备份。
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 允许直接 `python scripts/backup_db.py`：此时仓库根目录不在 sys.path 上
    sys.path.insert(0, str(ROOT))

# 复用主程序的常量而不是另抄一份文件名：备份工具与主程序一旦对"哪个文件才是库"有分歧，
# 就会出现"备份成功但备份的是空文件"这种最难被发现的问题
from app.main_context import DATABASE_FILENAME  # noqa: E402

DEFAULT_DATA_DIR = "./data"
ENV_DATA_DIR = "WELEARN_DATA_DIR"
BACKUP_DIR_NAME = "backups"
BACKUP_PREFIX = "welearn"
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
CONNECT_TIMEOUT_SECONDS = 5.0
# 判断"是否被占用"时不能等太久：这里是恢复前的探测，等 5 秒只会让命令看起来像卡死
LOCK_PROBE_TIMEOUT_SECONDS = 0.2


class BackupError(RuntimeError):
    """可预期的操作失败：只输出"发生了什么 + 可以做什么"，不打印堆栈（NFR-109）。"""


@dataclass(frozen=True)
class BackupInfo:
    """一次备份的结果。"""

    source: Path
    path: Path
    size_bytes: int
    integrity: str

    @property
    def ok(self) -> bool:
        return self.integrity == "ok"


@dataclass(frozen=True)
class RestoreResult:
    """一次恢复的结果；``retained`` 是恢复前原库改名后的位置。"""

    backup_file: Path
    target: Path
    retained: Path | None
    integrity: str


@dataclass(frozen=True)
class DatabaseReport:
    """一个数据库文件的检查报告。"""

    path: Path
    integrity: str
    user_version: int | None
    schema_version: int | None

    @property
    def ok(self) -> bool:
        # schema_version 表缺失说明它不是 WeLearn 的库，即便 integrity_check 通过也不该拿来恢复
        return self.integrity == "ok" and self.schema_version is not None


# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #


def resolve_data_dir(explicit: str | None = None, env: Mapping[str, str] | None = None) -> Path:
    """决定数据目录：命令行 > 环境变量 > 内置默认值。"""
    source = env if env is not None else os.environ
    raw = (explicit or source.get(ENV_DATA_DIR) or "").strip() or DEFAULT_DATA_DIR
    return Path(raw).expanduser().resolve()


def database_file(data_dir: Path) -> Path:
    return data_dir / DATABASE_FILENAME


def backup_directory(data_dir: Path) -> Path:
    return data_dir / BACKUP_DIR_NAME


def _unique_path(path: Path) -> Path:
    """同一秒内连做两次备份时避免互相覆盖。"""
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise BackupError(f"无法为 {path.name} 找到可用的文件名（同名文件过多），请先清理旧的备份。")


def _timestamp(moment: datetime | None = None) -> str:
    return (moment or datetime.now()).strftime(TIMESTAMP_FORMAT)


# --------------------------------------------------------------------------- #
# 读取检查
# --------------------------------------------------------------------------- #


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=CONNECT_TIMEOUT_SECONDS)
    connection.execute(f"PRAGMA busy_timeout = {int(CONNECT_TIMEOUT_SECONDS * 1000)}")
    return connection


def read_integrity(path: Path) -> str:
    """返回 ``PRAGMA integrity_check`` 的结果；正常为 ``"ok"``。

    损坏的文件（包括"根本不是 SQLite 文件"）不抛异常，而是把原因作为字符串返回：
    调用方需要先看一眼再决定是否继续，抛异常会让 restore 在改名原库之后才发现问题。
    """
    if not path.exists():
        return f"文件不存在：{path}"
    try:
        connection = _connect(path)
    except sqlite3.Error as exc:
        return f"无法打开数据库：{exc}"
    try:
        row = connection.execute("PRAGMA integrity_check").fetchone()
    except sqlite3.DatabaseError as exc:
        return f"无法读取数据库：{exc}"
    finally:
        connection.close()
    return str(row[0]) if row is not None else "未知"


def read_user_version(path: Path) -> int | None:
    """``PRAGMA user_version``；读不出来时返回 ``None``。"""
    if not path.exists():
        # 绝不能先 connect：sqlite3.connect 会顺手把不存在的路径建成一个空库，
        # 检查一个写错的 --file 就会在磁盘上留下垃圾文件
        return None
    try:
        connection = _connect(path)
    except sqlite3.Error:
        return None
    try:
        row = connection.execute("PRAGMA user_version").fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        connection.close()
    return int(row[0]) if row is not None else None


def read_schema_version(path: Path) -> int | None:
    """``schema_version`` 表里的最大版本号；表不存在时返回 ``None``。"""
    if not path.exists():
        return None
    try:
        connection = _connect(path)
    except sqlite3.Error:
        return None
    try:
        exists = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
        ).fetchone()
        if exists is None:
            return None
        row = connection.execute("SELECT MAX(version) FROM schema_version").fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        connection.close()
    return int(row[0]) if row and row[0] is not None else None


def inspect_database(path: Path) -> DatabaseReport:
    return DatabaseReport(
        path=path,
        integrity=read_integrity(path),
        user_version=read_user_version(path),
        schema_version=read_schema_version(path),
    )


# --------------------------------------------------------------------------- #
# 备份
# --------------------------------------------------------------------------- #


def _copy_with_backup_api(source: Path, destination: Path) -> None:
    """用在线备份 API 把 ``source`` 复制成 ``destination``。

    这是整个工具的核心：``Connection.backup`` 在**源连接**上加读锁并按页复制，
    因此复制过程中写入的事务要么整体可见、要么整体不可见，不会出现半个事务。
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_connection = _connect(source)
    except sqlite3.Error as exc:
        raise BackupError(
            f"无法打开数据库：{source}\n可以做什么：确认文件没有被其他程序独占，或改用 --data-dir 指向正确的数据目录。"
        ) from exc
    try:
        target_connection = sqlite3.connect(str(destination))
        try:
            source_connection.backup(target_connection)
        finally:
            target_connection.close()
    except (sqlite3.Error, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise BackupError(
            f"写备份文件失败：{destination}\n原因：{exc}\n可以做什么：确认磁盘空间充足、目标目录可写，然后重试。"
        ) from exc
    finally:
        source_connection.close()


def snapshot_database(source: Path, destination: Path) -> BackupInfo:
    """对（可能正在被使用的）``source`` 做一致性快照并校验。"""
    if not source.exists():
        raise BackupError(
            f"没找到数据库文件：{source}\n可以做什么：用 --data-dir 指定正确的数据目录"
            "（默认读环境变量 WELEARN_DATA_DIR，未设置时用 ./data）。"
        )
    _copy_with_backup_api(source, destination)
    integrity = read_integrity(destination)
    if integrity != "ok":
        # 校验不过的副本没有任何价值，留着只会让人误以为"备份是好的"
        destination.unlink(missing_ok=True)
        raise BackupError(
            f"备份文件未通过完整性检查：{integrity}\n可以做什么：确认磁盘没有故障后重试；"
            "若反复失败，请先对原库执行 `verify` 判断源文件是否已损坏。"
        )
    return BackupInfo(
        source=source,
        path=destination,
        size_bytes=destination.stat().st_size,
        integrity=integrity,
    )


def create_backup(data_dir: Path, *, now: datetime | None = None) -> BackupInfo:
    """在 ``<data_dir>/backups/`` 下生成 ``welearn-YYYYmmdd-HHMMSS.db``。"""
    source = database_file(data_dir)
    if not source.exists():
        raise BackupError(
            f"没找到数据库文件：{source}\n可以做什么：用 --data-dir 指定正确的数据目录"
            "（默认读环境变量 WELEARN_DATA_DIR，未设置时用 ./data）。"
        )
    destination = _unique_path(backup_directory(data_dir) / f"{BACKUP_PREFIX}-{_timestamp(now)}.db")
    return snapshot_database(source, destination)


def latest_backup(data_dir: Path) -> Path:
    files = sorted(backup_directory(data_dir).glob(f"{BACKUP_PREFIX}-*.db"))
    if not files:
        raise BackupError(
            f"没有找到备份文件：{backup_directory(data_dir)}\n"
            "可以做什么：先运行 `python scripts/backup_db.py backup --data-dir <数据目录>` 生成一个备份。"
        )
    return max(files, key=lambda item: item.stat().st_mtime)


# --------------------------------------------------------------------------- #
# 恢复
# --------------------------------------------------------------------------- #


def _probe_not_in_use(target: Path) -> None:
    """确认目标库没有别的进程在写。

    用 ``BEGIN IMMEDIATE`` 探测：它要拿写锁，被占用时立刻失败，而且不会改动数据。
    Windows 上"文件被打开"还会让随后的改名直接失败，所以调用方仍需处理 PermissionError。
    """
    connection = sqlite3.connect(str(target), timeout=LOCK_PROBE_TIMEOUT_SECONDS, isolation_level=None)
    try:
        connection.execute(f"PRAGMA busy_timeout = {int(LOCK_PROBE_TIMEOUT_SECONDS * 1000)}")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("ROLLBACK")
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if "locked" in message or "busy" in message:
            raise BackupError(
                f"目标数据库正在被其他进程使用：{target}\n"
                "可以做什么：先停止正在运行的 WeLearn 服务（关闭 uvicorn / 应用窗口），再重新执行 restore。"
            ) from exc
        raise BackupError(f"无法检查目标数据库：{exc}\n可以做什么：确认文件权限后重试。") from exc
    finally:
        connection.close()


def restore_database(backup_file: Path, data_dir: Path, *, now: datetime | None = None) -> RestoreResult:
    """把 ``backup_file`` 恢复成 ``<data_dir>/welearn.db``。

    顺序上先校验、再锁检查、最后才动原库：任何一步失败都不会留下"原库已没了、新库还没上"的状态。
    """
    if not backup_file.exists():
        raise BackupError(
            f"备份文件不存在：{backup_file}\n可以做什么：用 `python scripts/backup_db.py verify --data-dir <数据目录>` "
            "查看已有备份，再以 --file 指定其中之一。"
        )

    integrity = read_integrity(backup_file)
    if integrity != "ok":
        raise BackupError(
            f"备份文件未通过完整性检查：{integrity}\n"
            "可以做什么：换一个备份文件；若所有备份都不过，说明备份介质可能出了问题，"
            "请勿用它覆盖现有数据库。"
        )
    if read_schema_version(backup_file) is None:
        raise BackupError(
            f"备份文件里没有 schema_version 表，看起来不是 WeLearn 的数据库：{backup_file}\n"
            "可以做什么：确认 --file 指向的是本工具生成的备份（默认在 <数据目录>/backups/ 下）。"
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    target = database_file(data_dir)
    retained: Path | None = None

    if target.exists():
        _probe_not_in_use(target)
        retained = _unique_path(target.with_name(f"{target.name}.before-restore-{_timestamp(now)}"))
        try:
            target.replace(retained)
        except PermissionError as exc:
            raise BackupError(
                f"目标数据库被其他进程占用，无法改名为 {retained.name}：{target}\n"
                "可以做什么：先停止正在运行的 WeLearn 服务（关闭 uvicorn / 应用窗口），再重新执行 restore。"
            ) from exc
        # WAL/SHM 必须跟着原库一起挪走：留在原地会被新库当成自己的日志，属于最坏的一类混用
        for suffix in ("-wal", "-shm"):
            sidecar = target.with_name(target.name + suffix)
            if sidecar.exists():
                sidecar.replace(retained.with_name(retained.name + suffix))

    staging = _unique_path(data_dir / f".{DATABASE_FILENAME}.restore-tmp")
    try:
        _copy_with_backup_api(backup_file, staging)
        staging_integrity = read_integrity(staging)
        if staging_integrity != "ok":
            raise BackupError(
                f"写入的新库未通过完整性检查：{staging_integrity}\n"
                "可以做什么：确认磁盘空间充足后重试，不要继续使用这个文件。"
            )
        os.replace(staging, target)
    except (BackupError, sqlite3.Error, OSError) as exc:
        hint = (
            f"原库已保留为：{retained}（把它改回 {DATABASE_FILENAME} 即可回到恢复前的状态）。"
            if retained is not None
            else "目标数据库此前不存在，没有需要还原的原库。"
        )
        raise BackupError(f"恢复过程中出错：{exc}\n{hint}") from exc
    finally:
        staging.unlink(missing_ok=True)

    final_integrity = read_integrity(target)
    if final_integrity != "ok":
        advice = (
            f"把 {target.name} 删掉，并把 {retained} 改回 {target.name}。"
            if retained is not None
            else f"把 {target.name} 删掉，然后重新执行 restore。"
        )
        raise BackupError(f"恢复后的数据库未通过完整性检查：{final_integrity}\n可以做什么：{advice}")
    return RestoreResult(backup_file=backup_file, target=target, retained=retained, integrity=final_integrity)


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


def human_size(size: int) -> str:
    if size < 1024:
        return f"{size} 字节"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB（{size} 字节）"
    return f"{size / (1024 * 1024):.2f} MB（{size} 字节）"


def _locate_backup(raw: str, data_dir: Path) -> Path:
    """``--file`` 既接受绝对/相对当前目录的路径，也接受 backups 目录下的文件名。

    只写文件名时优先在 backups/ 里找：restore/verify 的日常用法是"用哪个备份"，
    让用户每次拼完整路径既啰嗦又容易写错。
    """
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.parent != Path("."):
        return candidate.resolve()
    named = backup_directory(data_dir) / candidate
    return named.resolve() if named.exists() else candidate.resolve()


def _print_report(report: DatabaseReport) -> None:
    print(f"文件：{report.path}")
    print(f"完整性检查（PRAGMA integrity_check）：{report.integrity}")
    user_version = "读取失败" if report.user_version is None else str(report.user_version)
    print(f"PRAGMA user_version：{user_version}")
    if report.schema_version is None:
        print("schema_version 表：缺失（这个文件可能不是 WeLearn 数据库）")
    else:
        print(f"schema_version 表：{report.schema_version}")
    print(f"结果：{'正常' if report.ok else '异常'}")


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    # default=SUPPRESS 让同一个选项可以写在子命令前或后，且不会被子命令的默认值覆盖
    common.add_argument(
        "--data-dir",
        metavar="DIR",
        default=argparse.SUPPRESS,
        help=f"数据目录（默认读环境变量 {ENV_DATA_DIR}，未设置时用 {DEFAULT_DATA_DIR}）",
    )

    parser = argparse.ArgumentParser(
        prog="backup_db.py",
        description="WeLearn 数据库备份 / 恢复 / 校验（在线备份，运行中也可安全执行）",
        epilog="示例：python scripts/backup_db.py backup --data-dir D:\\welearn-data",
        parents=[common],
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="{backup,restore,verify}")

    backup_parser = commands.add_parser(
        "backup",
        parents=[common],
        help="生成一致性快照到 <数据目录>/backups/",
        description="用 SQLite 在线备份 API 生成一致性快照（不是复制文件）。",
    )
    backup_parser.set_defaults(handler=_command_backup)

    restore_parser = commands.add_parser(
        "restore",
        parents=[common],
        help="用备份覆盖当前库（原库会先改名留存）",
        description="校验备份后覆盖 welearn.db；恢复前会把现有库改名为 welearn.db.before-restore-<时间戳>。",
    )
    restore_parser.add_argument("--file", metavar="BACKUP", default=None, help="要恢复的备份文件")
    restore_parser.set_defaults(handler=_command_restore)

    verify_parser = commands.add_parser(
        "verify",
        parents=[common],
        help="检查一个备份或当前库是否完好",
        description="对指定文件执行 integrity_check 与 schema_version 检查；损坏时退出码为 1。",
    )
    verify_parser.add_argument(
        "--file",
        metavar="BACKUP",
        default=None,
        help="要检查的文件；省略时检查 backups/ 下最新的备份",
    )
    verify_parser.set_defaults(handler=_command_verify)
    return parser


def _command_backup(args: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(getattr(args, "data_dir", None))
    source = database_file(data_dir)
    print(f"正在备份：{source}")
    print(f"当前库完整性检查：{read_integrity(source)}")
    info = create_backup(data_dir)
    print(f"备份完成：{info.path}")
    print(f"文件大小：{human_size(info.size_bytes)}")
    print(f"备份文件完整性检查：{info.integrity}")
    return 0


def _command_restore(args: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(getattr(args, "data_dir", None))
    raw = getattr(args, "file", None)
    if not raw:
        existing = sorted(backup_directory(data_dir).glob(f"{BACKUP_PREFIX}-*.db"))
        listing = "\n".join(f"  - {item}" for item in existing[-5:]) or "  （暂无备份）"
        raise BackupError(
            "restore 需要指定要恢复的备份文件：--file <备份路径>\n"
            f"可用的备份（最多列出 5 个）：\n{listing}\n"
            "可以做什么：先运行 `python scripts/backup_db.py backup` 生成备份，再用 --file 指定它。"
        )

    backup_file = _locate_backup(raw, data_dir)
    print(f"正在校验备份：{backup_file}")
    print(f"备份文件完整性检查：{read_integrity(backup_file)}")
    result = restore_database(backup_file, data_dir)
    if result.retained is not None:
        print(f"原库已留存：{result.retained}")
    else:
        print("原库不存在，无需留存。")
    print(f"恢复完成：{result.target}")
    report = inspect_database(result.target)
    print(f"恢复后完整性检查：{report.integrity}")
    print(f"schema_version 表：{report.schema_version if report.schema_version is not None else '缺失'}")
    return 0


def _command_verify(args: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(getattr(args, "data_dir", None))
    raw = getattr(args, "file", None)
    path = _locate_backup(raw, data_dir) if raw else latest_backup(data_dir)
    if raw:
        print(f"待检查文件：{path}")
    else:
        print(f"待检查文件（backups/ 下最新）：{path}")
    report = inspect_database(path)
    _print_report(report)
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    try:
        return int(handler(args))
    except BackupError as exc:
        print(f"操作失败：{exc}", file=sys.stderr)
        return 1
    except (OSError, sqlite3.Error) as exc:
        print(
            f"操作失败：{exc}\n可以做什么：确认数据目录存在、当前用户有读写权限，然后重试。",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
