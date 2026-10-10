"""路径安全测试（FR-908、NFR-109）。"""

from pathlib import Path

import pytest

from app.infra.paths import DataPaths, PathTraversalError


@pytest.fixture
def paths(tmp_path: Path) -> DataPaths:
    return DataPaths(tmp_path / "data")


def test_resolves_inside_root(paths: DataPaths) -> None:
    resolved = paths.resolve("llm", "settings.json")
    assert resolved == paths.root / "llm" / "settings.json"


def test_creates_root_on_demand(paths: DataPaths) -> None:
    assert not paths.root.exists()
    paths.ensure_root()
    assert paths.root.is_dir()


@pytest.mark.parametrize(
    "candidate",
    [
        "../outside.txt",
        "sub/../../outside.txt",
        "..",
    ],
)
def test_rejects_traversal(paths: DataPaths, candidate: str) -> None:
    paths.ensure_root()
    with pytest.raises(PathTraversalError):
        paths.resolve(candidate)
    assert paths.is_safe(candidate) is False


def test_rejects_absolute_path_outside_root(paths: DataPaths, tmp_path: Path) -> None:
    paths.ensure_root()
    outside = tmp_path / "outside" / "file.txt"
    outside.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PathTraversalError):
        paths.resolve(outside)


def test_rejects_root_itself(paths: DataPaths) -> None:
    """根目录本身不是合法目标（对应 ReferenceCode 的防护语义）。"""
    paths.ensure_root()
    assert paths.is_safe() is False
    with pytest.raises(PathTraversalError):
        paths.resolve()


def test_rejects_empty_path(paths: DataPaths) -> None:
    with pytest.raises(PathTraversalError):
        paths.resolve("")


def test_ensure_parent_creates_directory(paths: DataPaths) -> None:
    target = paths.ensure_parent(paths.root / "logs" / "app.log")
    assert target.parent.is_dir()
