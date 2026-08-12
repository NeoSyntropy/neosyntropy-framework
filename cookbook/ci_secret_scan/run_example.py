"""CI cookbook: loop ``fsm.run`` once per file path under a directory.

Workflow per file::

    FilePathInput → ReasoningNode (read / scan tools) → SchemaNode → ApiKeysInFile

The outer loop lists every source file in ``--repo``, then runs the FSM on
each path and aggregates the API-key lists.

Run against the bundled sample repo::

    python cookbook/ci_secret_scan/run_example.py

Or any directory::

    python cookbook/ci_secret_scan/run_example.py --repo /path/to/checkout

Credentials load from ``tests/.env`` (see ``tests/.env.example``).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from neosyntropy import (
    Client,
    FSM,
    OpenInput,
    ReasoningNode,
    SchemaNode,
    TextOutput,
    ToolRegistry,
    edge_deterministic,
    edge_fallback,
    tool,
)
from neosyntropy.tools.coding_tools import CodingTools

COOKBOOK_DIR = Path(__file__).resolve().parent
SAMPLE_REPO = COOKBOOK_DIR / "sample_repo"
TESTS_ENV_PATH = COOKBOOK_DIR.parents[1] / "tests" / ".env"
VERTEX_MODEL = "gemini-2.5-flash"

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "neosyntropy_api_key",
        re.compile(
            r"""(?:api_key|NEOSYNTROPY_API_KEY)\s*=\s*["'](nsk_[A-Za-z0-9_\-]{16,})["']"""
        ),
    ),
    (
        "generic_secret_literal",
        re.compile(
            r"""(?:api_key|secret|token|password)\s*=\s*["']([^"']{20,})["']"""
        ),
    ),
    (
        "aws_access_key_id",
        re.compile(
            r"""(?:AWS_ACCESS_KEY_ID|aws_access_key_id)\s*=\s*["'](AKIA[0-9A-Z]{16})["']"""
        ),
    ),
    (
        "private_key_pem",
        re.compile(r"(-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"),
    ),
)

SKIP_SUFFIXES = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".exe", ".dll"}
SKIP_DIR_NAMES = {
    ".git",
    ".hg",
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


def _load_tests_env() -> None:
    if not TESTS_ENV_PATH.is_file():
        raise SystemExit(
            f"Missing {TESTS_ENV_PATH}. Copy tests/.env.example to tests/.env and fill values."
        )
    for raw in TESTS_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required value {name} in {TESTS_ENV_PATH}.")
    return value


def _client_from_env() -> Client:
    return Client(
        api_key=_require_env("NEOSYNTROPY_API_KEY"),
        project_id=_require_env("NEOSYNTROPY_PROJECT_ID"),
        base_url=os.environ.get("NEOSYNTROPY_API_URL", "https://api.neosyntropy.com").strip()
        or "https://api.neosyntropy.com",
        telemetry_timeout=20.0,
    )


# ── schemas ──────────────────────────────────────────────────────────────────


class FilePathInput(BaseModel):
    """One FSM run = one file path under the selected directory."""

    model_config = ConfigDict(extra="forbid")
    intent: str
    file_path: str = Field(description="Path relative to the scanned directory root")


class ScanFileArgs(BaseModel):
    file_path: str = Field(description="Relative file path to scan for secrets")


class ApiKeysInFile(BaseModel):
    """SchemaNode output: every API key / secret literal found in this file."""

    model_config = ConfigDict(extra="forbid")
    file_path: str
    api_keys: list[str] = Field(
        default_factory=list,
        description="Secret values found in this file (redact in summaries if needed)",
    )
    clean: bool = Field(description="True when api_keys is empty")


# ── file listing + deterministic extract ─────────────────────────────────────


def list_source_files(root: Path) -> list[str]:
    """Return relative POSIX paths for every scannable file under ``root``."""
    root = root.resolve()
    paths: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.name in {".env.example", ".env.sample"}:
            continue
        paths.append(str(path.relative_to(root)).replace("\\", "/"))
    return sorted(paths)


def extract_api_keys_from_file(repo_root: Path, relative: str) -> list[dict[str, object]]:
    """Deterministic findings for one file (used by the scan tool)."""
    target = (repo_root / relative).resolve()
    try:
        target.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes repo root: {relative!r}") from exc
    if not target.is_file():
        raise ValueError(f"file not found: {relative}")

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValueError(f"cannot read {relative}: {exc}") from exc
    if "\x00" in text[:4096]:
        return []

    findings: list[dict[str, object]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in SECRET_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            value = match.group(1) if match.lastindex else match.group(0)
            findings.append(
                {
                    "path": relative.replace("\\", "/"),
                    "lineno": lineno,
                    "rule": rule,
                    "value": value,
                    "snippet": line.strip()[:200],
                }
            )
    return findings


def build_registry(repo_root: Path) -> ToolRegistry:
    registry = ToolRegistry()
    CodingTools(
        base_dir=repo_root,
        enable_grep=True,
        enable_find=False,
        enable_ls=False,
        enable_edit_file=False,
        enable_write_file=False,
        enable_run_shell=False,
    ).register(registry)

    @tool(registry=registry, name="scan_file_secrets")
    def scan_file_secrets(args: ScanFileArgs) -> dict:
        """Scan one file for hardcoded API keys / secrets."""
        findings = extract_api_keys_from_file(repo_root, args.file_path)
        keys = [str(item["value"]) for item in findings]
        return {
            "ok": len(findings) == 0,
            "file_path": args.file_path,
            "finding_count": len(findings),
            "api_keys": keys,
            "findings": findings,
        }

    return registry


def build_fsm() -> FSM:
    """Single-file workflow: investigate path → SchemaNode list of api_keys."""
    investigate = ReasoningNode(
        id="InvestigateFile",
        input_schema=FilePathInput,
        provider=VERTEX_MODEL,
        prompt=(
            "You are scanning ONE source file for hardcoded secrets. "
            "The file path is in the run input as file_path. "
            "1) Call scan_file_secrets with that exact file_path. "
            "2) Optionally read_file to confirm. "
            "3) List every secret value the tools returned. Do not invent keys."
        ),
        tools=("scan_file_secrets", "read_file", "grep"),
    )

    report = SchemaNode(
        id="ListApiKeys",
        input_schema=OpenInput,
        output_schema=ApiKeysInFile,
        provider=VERTEX_MODEL,
        prompt=(
            "Return JSON for this file only. "
            "file_path must match the scanned path. "
            "api_keys is the list of secret string values found (from tools). "
            "clean=true only when api_keys is empty."
        ),
    )

    out_of_scope = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        provider=VERTEX_MODEL,
        prompt="Refuse requests unrelated to scanning this file for secrets.",
    )

    return FSM(
        entry=investigate,
        nodes=[investigate, report, out_of_scope],
        edges=[
            edge_deterministic("InvestigateFile", "ListApiKeys"),
            edge_deterministic("ListApiKeys", "End"),
            edge_fallback("InvestigateFile", "OutOfScope"),
        ],
    )


def _report_from_result(result: object) -> ApiKeysInFile | None:
    for step in getattr(result, "steps", []) or []:
        for item in step.results:
            output = item.output
            if isinstance(output, ApiKeysInFile):
                return output
            if isinstance(output, dict) and "api_keys" in output and "file_path" in output:
                return ApiKeysInFile.model_validate(output)
    return None


def run_directory_scan(
    repo_root: Path,
    *,
    client: Client,
    file_paths: list[str] | None = None,
) -> tuple[list[ApiKeysInFile], int]:
    """List files under ``repo_root``, then ``fsm.run`` once per path."""
    repo_root = repo_root.resolve()
    paths = file_paths if file_paths is not None else list_source_files(repo_root)
    registry = build_registry(repo_root)
    fsm = build_fsm()

    reports: list[ApiKeysInFile] = []
    exit_code = 0

    print(f"Scanning {len(paths)} file(s) under {repo_root}")
    for index, relative in enumerate(paths, start=1):
        print(f"\n--- [{index}/{len(paths)}] {relative} ---")
        result = fsm.run(
            FilePathInput(
                intent=f"Scan file for hardcoded API keys: {relative}",
                file_path=relative,
            ),
            state={"file_path": relative, "repo_path": str(repo_root)},
            client=client,
            tools=registry,
        )
        print(f"Final State: {result.final_state} (Rejected: {result.rejected})")
        if result.rejection:
            print(f"REJECTED: {result.rejection}")
            exit_code = 1

        for step in result.steps:
            for item in step.results:
                for record in item.tool_calls:
                    verdict = "ok" if record.ok else ("denied" if record.denied else "failed")
                    print(
                        f"  [{verdict}] {record.tool} {record.arguments} "
                        f"{record.error or ''}"
                    )

        report = _report_from_result(result)
        if report is None:
            print("  (no ApiKeysInFile schema output)")
            if result.rejected:
                exit_code = 1
            continue

        reports.append(report)
        print(f"  clean={report.clean} api_keys={report.api_keys!r}")
        if not report.clean or report.api_keys:
            exit_code = 1

    return reports, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Loop fsm.run over every file path; SchemaNode lists API keys"
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=SAMPLE_REPO,
        help="Directory to scan (default: bundled sample_repo)",
    )
    args = parser.parse_args(argv)

    repo_root = args.repo.resolve()
    if not repo_root.is_dir():
        raise SystemExit(f"Repo path is not a directory: {repo_root}")

    _load_tests_env()
    client = _client_from_env()
    reports, exit_code = run_directory_scan(repo_root, client=client)

    print("\n======== AGGREGATE ========")
    dirty = [r for r in reports if r.api_keys]
    print(f"files_scanned={len(reports)} files_with_secrets={len(dirty)}")
    for report in dirty:
        print(f"  {report.file_path}: {report.api_keys}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
