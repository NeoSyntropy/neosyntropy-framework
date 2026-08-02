"""Concurrency-safe runtime state with one commit per step.

Ported from ``neosyntropy_backend_cli/core/fsm/state.py`` and extended with
``preview`` so gates can judge the post-commit state *before* anything
commits (fail-closed gates run on previews, commits happen after).
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .context import RunContext
from .models import NodeResult


class StateConflictError(ValueError):
    pass


def merge_step_results(
    results: list[NodeResult],
) -> tuple[dict[str, Any], str | None]:
    """Merge one step's proposals; conflicts raise instead of silently racing."""
    next_states = {result.next_state for result in results if result.next_state}
    if len(next_states) > 1:
        raise StateConflictError(
            f"parallel nodes proposed conflicting next states: {sorted(next_states)}"
        )
    updates: dict[str, Any] = {}
    for result in results:
        if result.status == "failed":
            continue
        for key, value in result.state_updates.items():
            if key in updates and updates[key] != value:
                raise StateConflictError(
                    f"parallel nodes proposed conflicting values for {key!r}"
                )
            updates[key] = value
    return updates, (next(iter(next_states)) if next_states else None)


class StateManager:
    def __init__(self, context: RunContext):
        self._state = deepcopy(context.state)
        self._current = context.current_state
        self._lock = asyncio.Lock()

    @property
    def current_state(self) -> str:
        return self._current

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._state)

    def preview(self, results: list[NodeResult]) -> tuple[dict[str, Any], str | None]:
        """State and next-state the workflow *would* have after this step."""
        updates, next_state = merge_step_results(results)
        state = self.snapshot()
        _deep_merge(state, updates)
        return state, next_state

    async def apply_step(self, results: list[NodeResult]) -> None:
        """Atomically commit one sequential step (possibly parallel nodes)."""
        updates, next_state = merge_step_results(results)
        async with self._lock:
            _deep_merge(self._state, updates)
            if next_state:
                self._current = next_state


def _deep_merge(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)
