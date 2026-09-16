from __future__ import annotations

from pathlib import Path

from neosyntropy.tools.coding.python import PythonTools
from neosyntropy.tools.core.local_file_system import LocalFileSystemTools
from neosyntropy.tools.core.toolkit import Toolkit


def test_check_path_allows_relative_file_inside_base(tmp_path: Path) -> None:
    toolkit = Toolkit()
    ok, resolved = toolkit._check_path("notes.txt", tmp_path, True)
    assert ok
    assert resolved == (tmp_path / "notes.txt").resolve()


def test_check_path_rejects_parent_escape(tmp_path: Path) -> None:
    toolkit = Toolkit()
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    base = tmp_path / "sandbox"
    base.mkdir()

    ok, resolved = toolkit._check_path("../secret.txt", base, True)
    assert not ok
    assert resolved == base.resolve()


def test_check_path_rejects_absolute_path_when_restricted(tmp_path: Path) -> None:
    toolkit = Toolkit()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")

    ok, resolved = toolkit._check_path(str(outside), tmp_path / "sandbox", True)
    assert not ok
    assert resolved == (tmp_path / "sandbox").resolve()


def test_check_path_rejects_symlink_escape(tmp_path: Path) -> None:
    toolkit = Toolkit()
    outside = tmp_path / "outside.txt"
    outside.write_text("leaked", encoding="utf-8")
    base = tmp_path / "sandbox"
    base.mkdir()
    (base / "link.txt").symlink_to(outside)

    ok, resolved = toolkit._check_path("link.txt", base, True)
    assert not ok
    assert resolved == base.resolve()


def test_check_path_allows_escape_when_unrestricted(tmp_path: Path) -> None:
    toolkit = Toolkit()
    outside = tmp_path / "outside.txt"
    outside.write_text("ok", encoding="utf-8")
    base = tmp_path / "sandbox"
    base.mkdir()

    ok, resolved = toolkit._check_path(str(outside), base, False)
    assert ok
    assert resolved == outside.resolve()


def test_local_file_system_write_and_read(tmp_path: Path) -> None:
    tools = LocalFileSystemTools(target_directory=str(tmp_path))
    written = tools.write_file("hello world", filename="note.txt")
    assert written.startswith("Successfully wrote file")
    assert (tmp_path / "note.txt").read_text() == "hello world"
    assert tools.read_file("note.txt") == "hello world"


def test_local_file_system_blocks_parent_escape(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("classified", encoding="utf-8")
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    tools = LocalFileSystemTools(target_directory=str(sandbox))

    written = tools.write_file("pwned", filename="../secret.txt")
    assert "outside the allowed base directory" in written
    assert (tmp_path / "secret.txt").read_text() == "classified"

    read = tools.read_file("../secret.txt")
    assert "outside the allowed base directory" in read


def test_python_tools_save_and_run_stays_in_base(tmp_path: Path) -> None:
    tools = PythonTools(base_dir=tmp_path)
    result = tools.save_to_file_and_run("demo.py", "value = 7\n", variable_to_return="value")
    assert result == "7"
    assert (tmp_path / "demo.py").exists()

    blocked = tools.save_to_file_and_run(
        "../escape.py", "value = 1\n", variable_to_return="value"
    )
    assert "outside the allowed base directory" in blocked
    assert not (tmp_path.parent / "escape.py").exists()
