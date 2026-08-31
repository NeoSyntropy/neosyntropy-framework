"""Unit tests for the per-file CI secret-scan cookbook (no API)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

COOKBOOK = Path(__file__).resolve().parents[1] / "cookbook" / "ci_secret_scan"
SAMPLE_REPO = COOKBOOK / "sample_repo"


def _load_run_example():
    path = COOKBOOK / "run_example.py"
    spec = importlib.util.spec_from_file_location("ci_secret_scan_run_example", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scan_mod():
    return _load_run_example()


def test_list_source_files_includes_leaky_skips_env_example(scan_mod) -> None:
    paths = scan_mod.list_source_files(SAMPLE_REPO)
    assert "leaky_client.py" in paths
    assert "config.py" in paths
    assert not any(p.endswith(".env.example") for p in paths)


def test_extract_api_keys_from_leaky_file(scan_mod) -> None:
    findings = scan_mod.extract_api_keys_from_file(SAMPLE_REPO, "leaky_client.py")
    assert findings
    values = [f["value"] for f in findings]
    assert any(str(v).startswith("nsk_") for v in values)


def test_clean_file_has_no_keys(scan_mod) -> None:
    findings = scan_mod.extract_api_keys_from_file(SAMPLE_REPO, "config.py")
    assert findings == []


def test_scan_file_tool_per_path(scan_mod) -> None:
    registry = scan_mod.build_registry(SAMPLE_REPO)
    assert "scan_file_secrets" in registry.names()

    leaky = registry.invoke("scan_file_secrets", {"file_path": "leaky_client.py"})
    assert leaky.ok
    assert leaky.result["ok"] is False
    assert leaky.result["api_keys"]

    clean = registry.invoke("scan_file_secrets", {"file_path": "config.py"})
    assert clean.ok
    assert clean.result["ok"] is True
    assert clean.result["api_keys"] == []
