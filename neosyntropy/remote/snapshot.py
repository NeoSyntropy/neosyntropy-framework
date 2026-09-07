"""Atomic local storage for server-backed graph snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bundles import canonical_json_bytes

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GITIGNORE = b"# NeoSyntropy local snapshots\n*\n!.gitignore\n"


def _path_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{field} must be 1-255 ASCII letters, digits, dots, hyphens, or underscores"
        )
    if value in {".", ".."}:
        raise ValueError(f"{field} must not be a relative path segment")
    return value


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA256 digest")
    return value


def _stage(path: Path, content: bytes) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _gitignore_bytes(path: Path) -> bytes:
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        return _GITIGNORE
    lines = existing.splitlines()
    additions = [
        line
        for line in (b"# NeoSyntropy local snapshots", b"*", b"!.gitignore")
        if line not in lines
    ]
    if not additions:
        return existing
    separator = b"" if not existing or existing.endswith((b"\n", b"\r")) else b"\n"
    return existing + separator + b"\n".join(additions) + b"\n"


def write_graph_snapshot(
    server_graph: Mapping[str, Any],
    associations: Sequence[Mapping[str, Any]],
    bundles: Mapping[str, bytes],
    *,
    project_root: Path | str | None = None,
) -> Path:
    """Persist one complete graph snapshot and return its graph directory.

    ``server_graph`` must contain ``id``, ``project_id``, ``manifest``, and
    ``recovery_manifest``. Bundle keys and association artifact hashes are raw,
    lowercase SHA256 digests over the exact compressed bytes.
    """
    if not isinstance(server_graph, Mapping):
        raise TypeError("server_graph must be a mapping")
    if isinstance(associations, (str, bytes)) or not isinstance(associations, Sequence):
        raise TypeError("associations must be a sequence of mappings")
    if not isinstance(bundles, Mapping):
        raise TypeError("bundles must be a mapping")

    project_id = _path_id(server_graph.get("project_id"), "server_graph.project_id")
    graph_id = _path_id(server_graph.get("id"), "server_graph.id")
    structure = server_graph.get("manifest")
    recovery = server_graph.get("recovery_manifest")
    if not isinstance(structure, Mapping):
        raise ValueError("server_graph.manifest must be a mapping")
    if not isinstance(recovery, Mapping):
        raise ValueError("server_graph.recovery_manifest must be a mapping")

    association_list: list[dict[str, Any]] = []
    associated_hashes: set[str] = set()
    for index, association in enumerate(associations):
        if not isinstance(association, Mapping):
            raise TypeError(f"associations[{index}] must be a mapping")
        artifact = association.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError(f"associations[{index}].artifact must be a mapping")
        associated_hashes.add(
            _digest(artifact.get("sha256"), f"associations[{index}].artifact.sha256")
        )
        association_list.append(dict(association))

    verified_bundles: dict[str, bytes] = {}
    for key, bundle in bundles.items():
        digest = _digest(key, "bundle key")
        if not isinstance(bundle, bytes):
            raise TypeError(f"bundle {digest} must contain bytes")
        if hashlib.sha256(bundle).hexdigest() != digest:
            raise ValueError(f"bundle {digest} does not match its SHA256 digest")
        verified_bundles[digest] = bundle
    if associated_hashes != set(verified_bundles):
        missing = sorted(associated_hashes - set(verified_bundles))
        extra = sorted(set(verified_bundles) - associated_hashes)
        raise ValueError(f"association and bundle hashes differ (missing={missing}, extra={extra})")

    serialized = {
        "structure.json": canonical_json_bytes(dict(structure)),
        "recovery.json": canonical_json_bytes(dict(recovery)),
        "artifacts.json": canonical_json_bytes(association_list),
    }

    root = Path.cwd() if project_root is None else Path(project_root)
    snapshot_root = root / ".neosyntropy"
    graph_dir = snapshot_root / project_id / "graphs" / graph_id
    artifact_dir = graph_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        (snapshot_root / ".gitignore", _gitignore_bytes(snapshot_root / ".gitignore")),
        *((graph_dir / name, content) for name, content in serialized.items()),
        *(
            (artifact_dir / f"{digest}.json.gz", bundle)
            for digest, bundle in verified_bundles.items()
        ),
    ]
    staged: list[tuple[Path, Path]] = []
    try:
        for target, content in targets:
            staged.append((target, _stage(target, content)))
        for target, temporary in staged:
            os.replace(temporary, target)
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)

    retained = {f"{digest}.json.gz" for digest in verified_bundles}
    for stale in artifact_dir.glob("*.json.gz"):
        if stale.name not in retained:
            stale.unlink()
    return graph_dir


def read_graph_snapshot(
    server_graph: Mapping[str, Any],
    *,
    project_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return a verified matching local snapshot, or ``None`` on cache miss."""
    project_id = _path_id(server_graph.get("project_id"), "server_graph.project_id")
    graph_id = _path_id(server_graph.get("id"), "server_graph.id")
    structure = server_graph.get("manifest")
    recovery = server_graph.get("recovery_manifest")
    if not isinstance(structure, Mapping) or not isinstance(recovery, Mapping):
        return None

    root = Path.cwd() if project_root is None else Path(project_root)
    graph_dir = root / ".neosyntropy" / project_id / "graphs" / graph_id
    try:
        cached_structure = json.loads((graph_dir / "structure.json").read_bytes())
        cached_recovery = json.loads((graph_dir / "recovery.json").read_bytes())
        associations = json.loads((graph_dir / "artifacts.json").read_bytes())
        if cached_structure != dict(structure) or cached_recovery != dict(recovery):
            return None
        if not isinstance(associations, list):
            return None

        bundles: dict[str, bytes] = {}
        for index, association in enumerate(associations):
            if not isinstance(association, dict):
                return None
            artifact = association.get("artifact")
            if not isinstance(artifact, dict):
                return None
            digest = _digest(
                artifact.get("sha256"),
                f"associations[{index}].artifact.sha256",
            )
            bundle = (graph_dir / "artifacts" / f"{digest}.json.gz").read_bytes()
            if hashlib.sha256(bundle).hexdigest() != digest:
                return None
            bundles[digest] = bundle
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return {
        "graph": dict(server_graph),
        "artifacts": associations,
        "bundles": bundles,
    }


__all__ = ["read_graph_snapshot", "write_graph_snapshot"]
