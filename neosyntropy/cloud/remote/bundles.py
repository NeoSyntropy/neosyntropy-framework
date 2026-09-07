"""Deterministic, content-addressed gzip JSON code bundles."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import json
import platform
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

BUNDLE_MEDIA_TYPE = "application/vnd.neosyntropy.remote.v1+json+gzip"
RUNTIME_POLICY = "compatible-runtime"


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON identically for equal values."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def gzip_json_bundle(value: Mapping[str, Any]) -> bytes:
    """Return deterministic gzip bytes (fixed mtime, filename and header OS)."""
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, compresslevel=9, mtime=0) as stream:
        stream.write(canonical_json_bytes(dict(value)))
    return output.getvalue()


def bundle_sha256(bundle: bytes) -> str:
    """SHA256 hex digest over the compressed bytes."""
    return hashlib.sha256(bundle).hexdigest()


@lru_cache(maxsize=1)
def runtime_compatibility() -> dict[str, Any]:
    """Describe the compatible host runtime required to hydrate a bundle."""
    try:
        framework_version = importlib.metadata.version("neosyntropy")
    except importlib.metadata.PackageNotFoundError:
        framework_version = "0.1.0"
    major, minor, *_ = (framework_version.split(".") + ["0"])
    framework = {
        "version": framework_version,
        "requires": f">={major}.{minor},<{major}.{int(minor) + 1}",
    }
    return {
        "policy": RUNTIME_POLICY,
        "python": {
            "implementation": platform.python_implementation().lower(),
            "requires": ">=3.10,<4",
        },
        "framework": framework,
        "neosyntropy": framework,
        "bundle_schema_version": 1,
    }


@lru_cache(maxsize=1)
def _package_distributions() -> dict[str, list[str]]:
    return importlib.metadata.packages_distributions()


def dependency_lock(imports: list[str]) -> dict[str, Any]:
    """Resolve third-party top-level imports to deterministic distribution versions."""
    package_map = _package_distributions()
    packages: list[dict[str, str]] = []
    for imported in sorted(set(imports)):
        if imported == "neosyntropy":
            continue
        distributions = package_map.get(imported) or []
        if not distributions:
            continue
        distribution = sorted(distributions, key=str.casefold)[0]
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        packages.append({"import": imported, "name": distribution, "version": version})
    return {"policy": RUNTIME_POLICY, "packages": packages}


def structure_hash(manifest: Mapping[str, Any]) -> str:
    """Hash a source-free manifest, excluding its existing structure hash."""
    payload = dict(manifest)
    payload.pop("structure_hash", None)
    payload.pop("revision", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def recovery_revision(manifest: Mapping[str, Any]) -> str:
    """Hash a complete recovery manifest independently of graph structure."""
    payload = dict(manifest)
    payload.pop("revision", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


__all__ = [
    "BUNDLE_MEDIA_TYPE",
    "bundle_sha256",
    "canonical_json_bytes",
    "dependency_lock",
    "gzip_json_bundle",
    "recovery_revision",
    "runtime_compatibility",
    "structure_hash",
]
