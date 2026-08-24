from __future__ import annotations

from pathlib import Path

import pytest

from neosyntropy.tools.coding.ast_tools import AstTools, AstToolsError
from neosyntropy.tools.core.registry import ToolRegistry


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        """
import os
from typing import Any

def greet(name: str) -> str:
    if not name:
        return "hi"
    return f"hello {name}"

class Greeter:
    def shout(self, name: str) -> str:
        try:
            return greet(name).upper()
        except:
            return "ERR"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_analyze_file_summarizes_module(sample_repo: Path) -> None:
    registry = ToolRegistry()
    AstTools(base_dir=sample_repo).register(registry)

    invocation = registry.invoke("analyze_file", {"path": "pkg/mod.py"})
    assert invocation.ok
    result = invocation.result
    assert result["path"] == "pkg/mod.py"
    assert [fn["name"] for fn in result["functions"]] == ["greet"]
    assert [cls["name"] for cls in result["classes"]] == ["Greeter"]
    assert result["classes"][0]["methods"][0]["name"] == "shout"
    assert "os" in result["imports"]
    assert result["bare_except_count"] == 1
    assert result["branch_nodes"] >= 2


def test_find_bare_excepts_gate(sample_repo: Path) -> None:
    registry = ToolRegistry()
    AstTools(base_dir=sample_repo).register(registry)

    invocation = registry.invoke("find_bare_excepts", {"path": "pkg/mod.py"})
    assert invocation.ok
    assert invocation.result["ok"] is False
    assert invocation.result["bare_except_count"] == 1


def test_path_escape_is_rejected(sample_repo: Path) -> None:
    tools = AstTools(base_dir=sample_repo)
    with pytest.raises(AstToolsError, match="escapes"):
        tools.analyzer.summarize("../outside.py")


def test_syntax_error_is_rejected(sample_repo: Path) -> None:
    (sample_repo / "bad.py").write_text("def broken(\n", encoding="utf-8")
    tools = AstTools(base_dir=sample_repo)
    with pytest.raises(AstToolsError, match="syntax error"):
        tools.analyzer.summarize("bad.py")
