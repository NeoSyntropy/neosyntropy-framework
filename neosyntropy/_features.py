"""Environment-controlled framework feature gates."""

from __future__ import annotations

import os

_ENABLED = "TRUE"


def remote_execution_enabled() -> bool:
    """Return whether executable graph publication and hydration are enabled."""
    return os.getenv("NEO_REMOTE_EXECUTION") == _ENABLED


def monitor_enabled() -> bool:
    """Return whether graph governance monitoring is enabled.

    Remote execution always implies monitoring because executable artifacts
    must be associated with a registered graph structure.
    """
    return (
        os.getenv("NEOSYNTROPY_MONITOR") == _ENABLED
        or remote_execution_enabled()
    )


__all__ = ["monitor_enabled", "remote_execution_enabled"]
