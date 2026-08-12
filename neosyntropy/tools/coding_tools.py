"""Coding toolkit (Agno CodingTools-style) for scoped file/shell work.

Core tools: ``read_file``, ``edit_file``, ``write_file``, ``run_shell``.
Exploration tools (opt-in): ``grep``, ``find``, ``ls``.

Paths and shell commands are restricted to ``base_dir`` by default.
``grep`` is implemented in pure Python so it works on Windows without
system ``grep``.
"""

from __future__ import annotations

import difflib
import fnmatch
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field

from .registry import ToolRegistry, tool


class CodingToolsError(RuntimeError):
    """Raised for security / configuration failures in the coding toolkit."""


_DANGEROUS_SHELL_PATTERNS = ("&&", "||", ";", "|", "$(", "`", ">", ">>", "<")

DEFAULT_ALLOWED_COMMANDS: tuple[str, ...] = (
    "python",
    "python3",
    "pytest",
    "pip",
    "pip3",
    "cat",
    "head",
    "tail",
    "wc",
    "ls",
    "find",
    "grep",
    "mkdir",
    "rm",
    "mv",
    "cp",
    "touch",
    "echo",
    "printf",
    "git",
    "chmod",
    "diff",
    "sort",
    "uniq",
    "tr",
    "cut",
)

_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".next",
    "dist",
    "build",
}


@dataclass
class CodingWorkspace:
    """File + shell operations under a scoped base directory."""

    base_dir: Path
    restrict_to_base_dir: bool = True
    max_lines: int = 2000
    max_bytes: int = 50_000
    shell_timeout: int = 120
    allowed_commands: tuple[str, ...] | None = field(
        default_factory=lambda: DEFAULT_ALLOWED_COMMANDS
    )

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).resolve()

    def resolve(
        self,
        relative_path: str | None,
        *,
        must_exist: bool = False,
        expect: str | None = None,
    ) -> Path:
        """Resolve a path under ``base_dir`` (fail-closed when restricted)."""
        if relative_path in (None, "", "."):
            candidate = self.base_dir
        else:
            if relative_path.strip() != relative_path:
                raise CodingToolsError("path must not have leading/trailing whitespace")
            raw = Path(relative_path)
            candidate = (
                raw.resolve()
                if raw.is_absolute()
                else (self.base_dir / relative_path).resolve()
            )
        if self.restrict_to_base_dir:
            try:
                candidate.relative_to(self.base_dir)
            except ValueError as exc:
                raise CodingToolsError(
                    f"path escapes base_dir: {relative_path!r}"
                ) from exc
        if must_exist and not candidate.exists():
            raise CodingToolsError(f"path not found: {relative_path or '.'}")
        if expect == "file" and candidate.exists() and not candidate.is_file():
            raise CodingToolsError(f"not a file: {relative_path}")
        if expect == "dir" and candidate.exists() and not candidate.is_dir():
            raise CodingToolsError(f"not a directory: {relative_path}")
        return candidate

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.base_dir)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")

    def truncate(self, text: str) -> tuple[str, bool, int]:
        lines = text.split("\n")
        total = len(lines)
        truncated = False
        if total > self.max_lines:
            lines = lines[: self.max_lines]
            truncated = True
        result = "\n".join(lines)
        if len(result.encode("utf-8", errors="replace")) > self.max_bytes:
            kept: list[str] = []
            size = 0
            for line in lines:
                add = len((line + "\n").encode("utf-8", errors="replace"))
                if size + add > self.max_bytes:
                    break
                kept.append(line)
                size += add
            result = "\n".join(kept)
            truncated = True
        return result, truncated, total

    def read_file(
        self,
        file_path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        path = self.resolve(file_path, must_exist=True, expect="file")
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            return {"ok": False, "error": f"binary file detected: {file_path}"}
        contents = raw.decode("utf-8", errors="replace")
        lines = contents.split("\n")
        total = len(lines)
        effective_limit = self.max_lines if limit is None else limit
        selected = lines[offset : offset + effective_limit]
        width = max(len(str(offset + len(selected))), 4)
        formatted = [
            f"{offset + i + 1:>{width}} | {line}" for i, line in enumerate(selected)
        ]
        text = "\n".join(formatted)
        text, was_truncated, _ = self.truncate(text)
        return {
            "ok": True,
            "path": self.rel(path),
            "content": text,
            "offset": offset,
            "shown_start": offset + 1 if selected else 0,
            "shown_end": offset + len(selected),
            "total_lines": total,
            "truncated": was_truncated or (offset + len(selected) < total),
        }

    def write_file(self, file_path: str, contents: str) -> dict[str, Any]:
        path = self.resolve(file_path, must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return {
            "ok": True,
            "path": self.rel(path),
            "lines": len(contents.split("\n")),
        }

    def edit_file(
        self,
        file_path: str,
        old_text: str,
        new_text: str,
    ) -> dict[str, Any]:
        if not old_text:
            return {"ok": False, "error": "old_text cannot be empty"}
        if old_text == new_text:
            return {"ok": True, "path": file_path, "diff": "", "unchanged": True}
        path = self.resolve(file_path, must_exist=True, expect="file")
        contents = path.read_text(encoding="utf-8")
        count = contents.count(old_text)
        if count == 0:
            return {
                "ok": False,
                "error": (
                    f"old_text not found in {file_path}. "
                    "Match must be exact (whitespace/indentation)."
                ),
            }
        if count > 1:
            return {
                "ok": False,
                "error": (
                    f"old_text matches {count} locations in {file_path}. "
                    "Add more surrounding context for a unique match."
                ),
            }
        updated = contents.replace(old_text, new_text, 1)
        path.write_text(updated, encoding="utf-8")
        diff = "".join(
            difflib.unified_diff(
                contents.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{file_path}",
                tofile=f"b/{file_path}",
                n=3,
            )
        )
        diff, truncated, total = self.truncate(diff)
        return {
            "ok": True,
            "path": self.rel(path),
            "diff": diff,
            "truncated": truncated,
            "diff_lines": total,
        }

    def check_command(self, command: str) -> str | None:
        if not self.restrict_to_base_dir:
            return None
        for pattern in _DANGEROUS_SHELL_PATTERNS:
            if pattern in command:
                return f"shell operator '{pattern}' is not allowed in restricted mode"
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            return "could not parse shell command"
        if not tokens:
            return "empty command"
        if self.allowed_commands is not None:
            cmd_base = Path(tokens[0]).name
            if cmd_base not in self.allowed_commands:
                return f"command '{cmd_base}' is not in the allowed commands list"
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            if "/" not in token and "\\" not in token and token != "..":
                continue
            try:
                resolved = (
                    Path(token).resolve()
                    if Path(token).is_absolute()
                    else (self.base_dir / token).resolve()
                )
                resolved.relative_to(self.base_dir)
            except ValueError:
                return f"command references path outside base directory: {token}"
            except (OSError, RuntimeError):
                continue
        return None

    def run_shell(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        violation = self.check_command(command)
        if violation:
            return {"ok": False, "error": violation}
        effective = self.shell_timeout if timeout is None else timeout
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=effective,
                cwd=str(self.base_dir),
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"command timed out after {effective} seconds",
            }
        output = completed.stdout or ""
        if completed.stderr:
            output += completed.stderr
        truncated_output, was_truncated, total_lines = self.truncate(output)
        return {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            "output": truncated_output,
            "truncated": was_truncated,
            "total_lines": total_lines,
        }

    def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        ignore_case: bool = False,
        include: str | None = None,
        context: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        if not pattern:
            return {"ok": False, "error": "pattern cannot be empty"}
        root = self.resolve(path, must_exist=True)
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            return {"ok": False, "error": f"invalid regex: {exc}"}

        matches: list[dict[str, Any]] = []
        files = [root] if root.is_file() else list(_iter_files(root, include))
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "\x00" in text[:8192]:
                continue
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if not regex.search(line):
                    continue
                start = max(0, idx - context)
                end = min(len(lines), idx + context + 1)
                matches.append(
                    {
                        "path": self.rel(file_path),
                        "lineno": idx + 1,
                        "line": line,
                        "context": [
                            {"lineno": start + j + 1, "text": lines[start + j]}
                            for j in range(end - start)
                        ]
                        if context
                        else None,
                    }
                )
                if len(matches) >= limit:
                    return {
                        "ok": True,
                        "pattern": pattern,
                        "match_count": len(matches),
                        "matches": matches,
                        "limited": True,
                    }
        return {
            "ok": True,
            "pattern": pattern,
            "match_count": len(matches),
            "matches": matches,
            "limited": False,
        }

    def find(
        self,
        pattern: str,
        *,
        path: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        if not pattern:
            return {"ok": False, "error": "pattern cannot be empty"}
        root = self.resolve(path, must_exist=True, expect="dir")
        found: list[str] = []
        for match in root.glob(pattern):
            try:
                match.relative_to(self.base_dir)
            except ValueError:
                continue
            suffix = "/" if match.is_dir() else ""
            found.append(self.rel(match) + suffix)
            if len(found) >= limit:
                break
        found.sort()
        return {
            "ok": True,
            "pattern": pattern,
            "count": len(found),
            "paths": found,
            "limited": len(found) >= limit,
        }

    def ls(self, path: str | None = None, *, limit: int = 500) -> dict[str, Any]:
        root = self.resolve(path, must_exist=True, expect="dir")
        entries: list[str] = []
        for entry in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            suffix = "/" if entry.is_dir() else ""
            entries.append(entry.name + suffix)
            if len(entries) >= limit:
                break
        return {
            "ok": True,
            "path": self.rel(root) if root != self.base_dir else ".",
            "entries": entries,
            "count": len(entries),
            "limited": len(entries) >= limit,
        }


def _iter_files(root: Path, include: str | None) -> Sequence[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        if include and not fnmatch.fnmatch(path.name, include):
            continue
        files.append(path)
    return files


# ── pydantic args ─────────────────────────────────────────────────────────────


class ReadFileArgs(BaseModel):
    file_path: str = Field(description="File path relative to base_dir")
    offset: int = Field(default=0, ge=0, description="0-based start line")
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Max lines to return (defaults to toolkit max_lines)",
    )


class WriteFileArgs(BaseModel):
    file_path: str = Field(description="File path relative to base_dir")
    contents: str = Field(description="Full file contents to write")


class EditFileArgs(BaseModel):
    file_path: str = Field(description="File path relative to base_dir")
    old_text: str = Field(description="Exact text to replace (must match once)")
    new_text: str = Field(description="Replacement text")


class RunShellArgs(BaseModel):
    command: str = Field(description="Shell command to run in base_dir")
    timeout: int | None = Field(
        default=None,
        ge=1,
        description="Timeout seconds (defaults to toolkit shell_timeout)",
    )


class GrepArgs(BaseModel):
    pattern: str = Field(description="Regex search pattern")
    path: str | None = Field(
        default=None,
        description="File or directory relative to base_dir (default: base_dir)",
    )
    ignore_case: bool = Field(default=False, description="Case-insensitive search")
    include: str | None = Field(
        default=None,
        description="Filename glob filter, e.g. '*.py'",
    )
    context: int = Field(default=0, ge=0, description="Context lines around matches")
    limit: int = Field(default=100, ge=1, description="Max matches to return")


class FindArgs(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py'")
    path: str | None = Field(
        default=None,
        description="Directory relative to base_dir (default: base_dir)",
    )
    limit: int = Field(default=500, ge=1, description="Max paths to return")


class LsArgs(BaseModel):
    path: str | None = Field(
        default=None,
        description="Directory relative to base_dir (default: base_dir)",
    )
    limit: int = Field(default=500, ge=1, description="Max entries to return")


class CodingTools:
    """Register Agno-style coding tools on a ToolRegistry."""

    def __init__(
        self,
        *,
        base_dir: str | Path,
        restrict_to_base_dir: bool = True,
        max_lines: int = 2000,
        max_bytes: int = 50_000,
        shell_timeout: int = 120,
        allowed_commands: Sequence[str] | None = DEFAULT_ALLOWED_COMMANDS,
        enable_read_file: bool = True,
        enable_edit_file: bool = True,
        enable_write_file: bool = True,
        enable_run_shell: bool = True,
        enable_grep: bool = False,
        enable_find: bool = False,
        enable_ls: bool = False,
        all: bool = False,
    ) -> None:
        cmds: tuple[str, ...] | None
        if allowed_commands is None:
            cmds = None
        elif allowed_commands is DEFAULT_ALLOWED_COMMANDS:
            cmds = DEFAULT_ALLOWED_COMMANDS
        else:
            cmds = tuple(allowed_commands)
        self.workspace = CodingWorkspace(
            base_dir=Path(base_dir),
            restrict_to_base_dir=restrict_to_base_dir,
            max_lines=max_lines,
            max_bytes=max_bytes,
            shell_timeout=shell_timeout,
            allowed_commands=cmds,
        )
        self.enable_read_file = all or enable_read_file
        self.enable_edit_file = all or enable_edit_file
        self.enable_write_file = all or enable_write_file
        self.enable_run_shell = all or enable_run_shell
        self.enable_grep = all or enable_grep
        self.enable_find = all or enable_find
        self.enable_ls = all or enable_ls

    def register(self, registry: ToolRegistry | None = None) -> ToolRegistry:
        target = registry or ToolRegistry()
        ws = self.workspace

        if self.enable_read_file:

            @tool(registry=target, name="read_file")
            def read_file(args: ReadFileArgs) -> dict[str, Any]:
                """Read a file with line numbers (supports offset/limit pagination)."""
                return ws.read_file(
                    args.file_path, offset=args.offset, limit=args.limit
                )

        if self.enable_edit_file:

            @tool(registry=target, name="edit_file")
            def edit_file(args: EditFileArgs) -> dict[str, Any]:
                """Exact find-and-replace edit; old_text must match exactly once."""
                return ws.edit_file(args.file_path, args.old_text, args.new_text)

        if self.enable_write_file:

            @tool(registry=target, name="write_file")
            def write_file(args: WriteFileArgs) -> dict[str, Any]:
                """Create or overwrite a file (parents created automatically)."""
                return ws.write_file(args.file_path, args.contents)

        if self.enable_run_shell:

            @tool(registry=target, name="run_shell")
            def run_shell(args: RunShellArgs) -> dict[str, Any]:
                """Run an allow-listed shell command inside base_dir."""
                return ws.run_shell(args.command, timeout=args.timeout)

        if self.enable_grep:

            @tool(registry=target, name="grep")
            def grep(args: GrepArgs) -> dict[str, Any]:
                """Search file contents with a regex (pure Python, Windows-safe)."""
                return ws.grep(
                    args.pattern,
                    path=args.path,
                    ignore_case=args.ignore_case,
                    include=args.include,
                    context=args.context,
                    limit=args.limit,
                )

        if self.enable_find:

            @tool(registry=target, name="find")
            def find(args: FindArgs) -> dict[str, Any]:
                """Find files by glob pattern under base_dir."""
                return ws.find(args.pattern, path=args.path, limit=args.limit)

        if self.enable_ls:

            @tool(registry=target, name="ls")
            def ls(args: LsArgs) -> dict[str, Any]:
                """List directory entries (directories end with /)."""
                return ws.ls(args.path, limit=args.limit)

        return target
