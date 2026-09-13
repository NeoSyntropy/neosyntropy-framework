from pathlib import Path

import pytest

from neosyntropy.exceptions import PathSecurityError
from neosyntropy.knowledge.filesystem import FileSystemKnowledge
from neosyntropy.utils.path_safety import safe_join_relative_path


def test_safe_join_allows_path_inside_base(tmp_path: Path) -> None:
    base = tmp_path / "proj"
    base.mkdir()
    (base / "readme.txt").write_text("ok", encoding="utf-8")

    resolved = safe_join_relative_path(base, "readme.txt")
    assert resolved == (base / "readme.txt").resolve()


def test_safe_join_rejects_parent_escape(tmp_path: Path) -> None:
    base = tmp_path / "proj"
    base.mkdir()
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")

    with pytest.raises(PathSecurityError):
        safe_join_relative_path(base, "../outside.txt")


def test_safe_join_rejects_prefix_sibling(tmp_path: Path) -> None:
    """'/data/proj_secret' must not pass a startswith('/data/proj') check."""
    base = tmp_path / "proj"
    sibling = tmp_path / "proj_secret"
    base.mkdir()
    sibling.mkdir()
    (sibling / "keys.txt").write_text("leaked", encoding="utf-8")

    with pytest.raises(PathSecurityError):
        safe_join_relative_path(base, "../proj_secret/keys.txt")


def test_filesystem_knowledge_get_file_blocks_prefix_sibling(tmp_path: Path) -> None:
    base = tmp_path / "proj"
    sibling = tmp_path / "proj_secret"
    base.mkdir()
    sibling.mkdir()
    (base / "public.txt").write_text("visible", encoding="utf-8")
    (sibling / "keys.txt").write_text("leaked", encoding="utf-8")

    knowledge = FileSystemKnowledge(base_dir=str(base))
    assert knowledge._get_file("public.txt")[0].content == "visible"
    assert knowledge._get_file("../proj_secret/keys.txt") == []
