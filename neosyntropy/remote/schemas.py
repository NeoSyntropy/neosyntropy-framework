"""Public schemas for extracted code artifacts and local bundle delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


class CallableProvenance(TypedDict, total=False):
    module: str | None
    name: str
    qualname: str
    is_async: bool
    source_file: str | None


class CodeArtifactRef(TypedDict, total=False):
    """Source-free metadata for one callable bundle."""

    id: str
    sha256: str | None
    media_type: str
    size_bytes: int
    compressed_size: int
    uncompressed_size: int
    entry_file: str
    external_imports: list[str]
    extractable: bool
    truncated: bool
    required: bool
    publishable: bool
    reason: str
    owner_role: str
    provenance: CallableProvenance
    bindings: dict[str, Any]
    runtime_compat: dict[str, Any]
    dependency_lock: dict[str, Any]
    recoverability: dict[str, Any]


@dataclass(frozen=True)
class ManifestBundle:
    """A slim manifest plus content-addressed bundle bytes kept locally."""

    manifest: dict[str, Any]
    bundles: dict[str, bytes]

    def bundle(self, artifact_id: str) -> bytes:
        """Return compressed bytes for an artifact id or SHA256 digest."""
        digest = artifact_id.removeprefix("sha256:")
        return self.bundles[digest]


__all__ = ["CallableProvenance", "CodeArtifactRef", "ManifestBundle"]
