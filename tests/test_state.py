from __future__ import annotations

import asyncio

import pytest

from neosyntropy import RunContext, StateConflictError, StateManager
from neosyntropy.core.models import NodeResult


def make_manager(state: dict | None = None, current: str = "Start") -> StateManager:
    context = RunContext(
        request_id="req-1",
        intent="test",
        current_state=current,
        state=state or {},
    )
    return StateManager(context)


def test_apply_step_commits_updates_and_transition():
    manager = make_manager({"a": 1})
    asyncio.run(
        manager.apply_step(
            [NodeResult(node_id="N", state_updates={"b": 2}, next_state="N")]
        )
    )
    assert manager.current_state == "N"
    assert manager.snapshot() == {"a": 1, "b": 2}


def test_conflicting_next_states_raise():
    manager = make_manager()
    results = [
        NodeResult(node_id="A", next_state="X"),
        NodeResult(node_id="B", next_state="Y"),
    ]
    with pytest.raises(StateConflictError, match="conflicting next states"):
        asyncio.run(manager.apply_step(results))


def test_conflicting_values_for_same_key_raise():
    manager = make_manager()
    results = [
        NodeResult(node_id="A", state_updates={"k": 1}),
        NodeResult(node_id="B", state_updates={"k": 2}),
    ]
    with pytest.raises(StateConflictError, match="conflicting values"):
        asyncio.run(manager.apply_step(results))


def test_failed_results_do_not_contribute_updates():
    manager = make_manager()
    asyncio.run(
        manager.apply_step(
            [NodeResult(node_id="A", status="failed", state_updates={"k": 1})]
        )
    )
    assert manager.snapshot() == {}


def test_preview_does_not_commit():
    manager = make_manager({"a": 1})
    preview, next_state = manager.preview(
        [NodeResult(node_id="N", state_updates={"b": 2}, next_state="N")]
    )
    assert preview == {"a": 1, "b": 2}
    assert next_state == "N"
    # Nothing committed.
    assert manager.snapshot() == {"a": 1}
    assert manager.current_state == "Start"


def test_deep_merge_of_nested_dicts():
    manager = make_manager({"cart": {"items": 1}})
    asyncio.run(
        manager.apply_step(
            [NodeResult(node_id="N", state_updates={"cart": {"total": 9.5}})]
        )
    )
    assert manager.snapshot() == {"cart": {"items": 1, "total": 9.5}}
