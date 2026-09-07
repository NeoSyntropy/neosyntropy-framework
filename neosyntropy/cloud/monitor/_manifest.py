"""Shared helpers for structure-only telemetry manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def structure_hash(manifest: Mapping[str, Any]) -> str:
    """Hash a structure manifest independently of remote bundle metadata."""
    payload = dict(manifest)
    payload.pop("structure_hash", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["structure_hash"]
