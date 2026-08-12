"""AST analysis toolkit for CI-style static gates.

Agno-style local toolkit: paths are resolved under a caller-provided
``base_dir`` (fail-closed — no escape). Uses the stdlib ``ast`` module only.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .registry import ToolRegistry, tool


class AstToolsError(RuntimeError):
    """Raised when a path is unsafe or the source cannot be analyzed."""


@dataclass(frozen=True, slots=True)
class AstAnalyzer:
    """Parse Python source under a scoped base directory."""

    base_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_dir", self.base_dir.resolve())

    def resolve(self, relative_path: str) -> Path:
        if not relative_path or relative_path.strip() != relative_path:
            raise AstToolsError("path must be a non-empty relative path")
        candidate = (self.base_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.base_dir)
        except ValueError as exc:
            raise AstToolsError(
                f"path escapes base_dir: {relative_path!r}"
            ) from exc
        if not candidate.is_file():
            raise AstToolsError(f"file not found: {relative_path}")
        if candidate.suffix != ".py":
            raise AstToolsError(f"expected a .py file: {relative_path}")
        return candidate

    def read_tree(self, relative_path: str) -> tuple[Path, ast.AST]:
        path = self.resolve(relative_path)
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise AstToolsError(
                f"syntax error in {relative_path}: {exc.msg} "
                f"(line {exc.lineno})"
            ) from exc
        return path, tree

    def summarize(self, relative_path: str) -> dict[str, Any]:
        path, tree = self.read_tree(relative_path)
        functions: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        imports: list[str] = []
        bare_excepts: list[dict[str, int]] = []
        branch_nodes = 0

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(_function_summary(node))
            elif isinstance(node, ast.ClassDef):
                classes.append(_class_summary(node))
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(_import_names(node))

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler)):
                branch_nodes += 1
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                bare_excepts.append({"lineno": node.lineno or 0})
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in tree.body:
                imports.extend(_import_names(node))

        return {
            "path": str(path.relative_to(self.base_dir)).replace("\\", "/"),
            "functions": functions,
            "classes": classes,
            "imports": sorted(set(imports)),
            "bare_except_count": len(bare_excepts),
            "bare_excepts": bare_excepts,
            "branch_nodes": branch_nodes,
        }

    def find_bare_excepts(self, relative_path: str) -> dict[str, Any]:
        summary = self.summarize(relative_path)
        return {
            "path": summary["path"],
            "bare_except_count": summary["bare_except_count"],
            "bare_excepts": summary["bare_excepts"],
            "ok": summary["bare_except_count"] == 0,
        }


def _import_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = node.module or ""
    if not node.names:
        return [module] if module else []
    return [f"{module}.{alias.name}" if module else alias.name for alias in node.names]


def _function_summary(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    args = [arg.arg for arg in node.args.args]
    return {
        "name": node.name,
        "lineno": node.lineno,
        "args": args,
        "async": isinstance(node, ast.AsyncFunctionDef),
        "decorators": [_decorator_name(d) for d in node.decorator_list],
    }


def _class_summary(node: ast.ClassDef) -> dict[str, Any]:
    methods = [
        _function_summary(child)
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return {
        "name": node.name,
        "lineno": node.lineno,
        "methods": methods,
        "bases": [_expr_name(base) for base in node.bases],
    }


def _decorator_name(node: ast.AST) -> str:
    return _expr_name(node)


def _expr_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expr_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _expr_name(node.func)
    return type(node).__name__


class AnalyzeFileArgs(BaseModel):
    path: str = Field(
        description="Python file path relative to the toolkit base directory",
    )


class FindBareExceptsArgs(BaseModel):
    path: str = Field(
        description="Python file path relative to the toolkit base directory",
    )


class AstTools:
    """Register AST CI tools on a ToolRegistry (Agno-style toolkit)."""

    def __init__(self, *, base_dir: str | Path) -> None:
        self.analyzer = AstAnalyzer(base_dir=Path(base_dir))

    def register(self, registry: ToolRegistry | None = None) -> ToolRegistry:
        target = registry or ToolRegistry()

        @tool(registry=target, name="analyze_file")
        def analyze_file(args: AnalyzeFileArgs) -> dict[str, Any]:
            """Summarize functions, classes, imports, and branch complexity via AST."""
            return self.analyzer.summarize(args.path)

        @tool(registry=target, name="find_bare_excepts")
        def find_bare_excepts(args: FindBareExceptsArgs) -> dict[str, Any]:
            """Report bare ``except:`` handlers (CI static gate)."""
            return self.analyzer.find_bare_excepts(args.path)

        return target
