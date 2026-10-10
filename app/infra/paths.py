"""数据目录与路径安全（T0.3）。

移植 ReferenceCode/DataManager.cs 的核心防护：**任何读写路径在解析后都必须位于数据根目录
之内，且不得是根目录本身**，以此抵御 `../` 越界与绝对路径注入（FR-908、NFR-109）。
"""

from pathlib import Path


class PathTraversalError(ValueError):
    """路径越出数据根目录。"""


class DataPaths:
    """数据根目录与安全路径解析。"""

    def __init__(self, root: Path) -> None:
        self._root = Path(root).expanduser().resolve()

    @property
    def root(self) -> Path:
        """数据根目录（绝对路径）。"""
        return self._root

    def ensure_root(self) -> Path:
        """确保根目录存在并返回。"""
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def resolve(self, *parts: str | Path) -> Path:
        """把相对片段解析为根目录内的绝对路径；越界则抛 :class:`PathTraversalError`。

        允许解析尚不存在的路径（用于写入前校验），并禁止指向根本身。
        """
        if not parts:
            raise PathTraversalError("路径不能为空")

        candidate = self._root
        for part in parts:
            candidate = candidate / str(part)

        full = candidate.resolve()
        # 必须位于根目录之内，且不能是根目录本身（Windows 下不能用字符串前缀判断，
        # 路径分隔符是 "\"；is_relative_to 是平台无关且能正确处理大小写的写法）
        if full == self._root or not full.is_relative_to(self._root):
            raise PathTraversalError(f"路径越出数据根目录: {full}")
        return full

    def is_safe(self, *parts: str | Path) -> bool:
        """安全判定；越界或为空时返回 False 而不抛异常。"""
        try:
            self.resolve(*parts)
        except (PathTraversalError, OSError, ValueError):
            return False
        return True

    def ensure_parent(self, path: Path) -> Path:
        """确保目标文件所在目录存在。"""
        full = path if path.is_absolute() else self.resolve(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        return full
