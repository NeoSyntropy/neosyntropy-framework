"""Delivery: GRAPH.md lists every scenario directory and each has a test."""

from __future__ import annotations

from tests.scenarios import DELIVERIES_ROOT, SCENARIOS_ROOT, scenario_ids


def test_graph_md_names_every_scenario() -> None:
    graph = (SCENARIOS_ROOT / "GRAPH.md").read_text(encoding="utf-8")
    ids = scenario_ids()
    assert ids, "expected at least one scenario directory"
    missing = [scenario_id for scenario_id in ids if scenario_id not in graph]
    assert missing == [], f"GRAPH.md is missing scenarios: {missing}"


def test_every_scenario_has_a_delivery_test() -> None:
    ids = scenario_ids()
    missing = [
        scenario_id
        for scenario_id in ids
        if not (DELIVERIES_ROOT / f"test_{scenario_id}.py").is_file()
    ]
    assert missing == [], f"deliveries missing tests for: {missing}"


def test_every_scenario_file_declares_matching_id() -> None:
    for scenario_id in scenario_ids():
        text = (SCENARIOS_ROOT / scenario_id / "scenario.py").read_text(encoding="utf-8")
        assert f'SCENARIO_ID = "{scenario_id}"' in text
