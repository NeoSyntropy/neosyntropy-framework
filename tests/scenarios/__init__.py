"""Scenario map for NeoSyntropy production-style graphs.

Each scenario lives in ``tests/scenarios/<id>/scenario.py`` — the story and
the executable graph are the same file. Deliveries under
``tests/scenarios/deliveries/`` re-run that graph and assert the related
database (and vector store) actually changed.

Cookbook examples are copied as ``cookbook_*`` scenarios. Live backend
endpoints for those copies are listed in ``BACKEND.md``.

Run the delivery suite::

    pytest tests/scenarios/deliveries
"""

from __future__ import annotations

from pathlib import Path

SCENARIOS_ROOT = Path(__file__).resolve().parent
DELIVERIES_ROOT = SCENARIOS_ROOT / "deliveries"


def scenario_ids() -> list[str]:
    """Return scenario directory names that contain a ``scenario.py``."""
    skip = {"deliveries", "__pycache__"}
    ids = [
        path.name
        for path in SCENARIOS_ROOT.iterdir()
        if path.is_dir() and path.name not in skip and (path / "scenario.py").is_file()
    ]
    return sorted(ids)
