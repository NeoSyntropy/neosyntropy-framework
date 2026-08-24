from __future__ import annotations

from pathlib import Path

import pytest

from neosyntropy.tools.coding.coding_tools import CodingTools, CodingToolsError
from neosyntropy.tools.core.registry import ToolRegistry


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "def hello():\n    return 'hi'\n\ndef world():\n    return 'yo'\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    return tmp_path


def test_registers_core_and_exploration_tools(repo: Path) -> None:
    registry = ToolRegistry()
    CodingTools(base_dir=repo, all=True).register(registry)
    names = set(registry.names())
    assert names == {
        "read_file",
        "edit_file",
        "write_file",
        "run_shell",
        "grep",
        "find",
        "ls",
    }


def test_opt_in_grep_find_ls_default_off(repo: Path) -> None:
    registry = ToolRegistry()
    CodingTools(base_dir=repo).register(registry)
    names = set(registry.names())
    assert "read_file" in names
    assert "grep" not in names
    assert "find" not in names
    assert "ls" not in names


def test_read_grep_find_ls(repo: Path) -> None:
    registry = ToolRegistry()
    CodingTools(base_dir=repo, enable_grep=True, enable_find=True, enable_ls=True).register(
        registry
    )

    read = registry.invoke("read_file", {"file_path": "src/app.py"})
    assert read.ok
    assert "hello" in read.result["content"]

    grep = registry.invoke(
        "grep",
        {"pattern": "def ", "include": "*.py", "path": "src"},
    )
    assert grep.ok
    assert grep.result["match_count"] == 2

    found = registry.invoke("find", {"pattern": "**/*.py"})
    assert found.ok
    assert "src/app.py" in found.result["paths"]

    listing = registry.invoke("ls", {"path": "src"})
    assert listing.ok
    assert "app.py" in listing.result["entries"]


def test_edit_file_unique_replace(repo: Path) -> None:
    registry = ToolRegistry()
    CodingTools(base_dir=repo, enable_grep=False, enable_run_shell=False).register(registry)

    edited = registry.invoke(
        "edit_file",
        {
            "file_path": "src/app.py",
            "old_text": "def hello():\n    return 'hi'\n",
            "new_text": "def hello():\n    return 'hello'\n",
        },
    )
    assert edited.ok
    assert "hello" in edited.result["diff"]
    assert "return 'hello'" in (repo / "src" / "app.py").read_text(encoding="utf-8")


def test_path_escape_rejected(repo: Path) -> None:
    tools = CodingTools(base_dir=repo, enable_grep=True)
    with pytest.raises(CodingToolsError, match="escapes"):
        tools.workspace.read_file("../outside.py")


def test_run_shell_allowlist(repo: Path) -> None:
    registry = ToolRegistry()
    CodingTools(base_dir=repo, enable_run_shell=True).register(registry)

    denied = registry.invoke("run_shell", {"command": "curl https://example.com"})
    assert denied.ok
    assert denied.result["ok"] is False
    assert "allowed" in denied.result["error"]
