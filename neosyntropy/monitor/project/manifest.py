"""Project manifest generator for UI visualisation and telemetry."""
from __future__ import annotations

from typing import Any


def project_manifest(
    project_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the serialisable description of a project registration.

    This is emitted when a project is first encountered from the SDK side so
    the console can reflect it even before any API-level project has been
    explicitly created.
    """
    return {
        "schema_version": 1,
        "project_id": project_id,
        "name": name,
        "description": description,
        "metadata": metadata or {},
    }
