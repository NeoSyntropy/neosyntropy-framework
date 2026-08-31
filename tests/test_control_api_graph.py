"""Control API graph projection keeps routers out of wire nodes."""

from __future__ import annotations

from neosyntropy.backend import _control_api_graph


def test_control_api_graph_moves_router_stubs_to_routers() -> None:
    wire = _control_api_graph(
        {
            "schema_version": 1,
            "entry": "PhaseRouter",
            "input_schema": {"type": "object"},
            "nodes": [
                {
                    "id": "DoWork",
                    "name": "DoWork",
                    "kind": "schema",
                    "is_fallback": False,
                    "output_schema": {"type": "object"},
                    "input_schema": {"type": "object"},
                },
                {
                    "id": "OutOfScope",
                    "name": "OutOfScope",
                    "kind": "schema",
                    "is_fallback": True,
                    "output_schema": {"type": "object"},
                },
                {
                    "id": "PhaseRouter",
                    "name": "PhaseRouter",
                    "kind": "router",
                    "is_fallback": False,
                    "input_schema": {"type": "object"},
                    "output_schema": None,
                },
                {
                    "id": "SkillRouter",
                    "name": "SkillRouter",
                    "kind": "router",
                    "output_schema": None,
                },
            ],
            "routers": ["PhaseRouter"],
            "edges": [],
            "groups": [],
        }
    )

    node_ids = {node["id"] for node in wire["nodes"]}
    assert node_ids == {"DoWork", "OutOfScope"}
    assert "PhaseRouter" not in node_ids
    assert "SkillRouter" not in node_ids
    assert wire["routers"] == ["PhaseRouter", "SkillRouter"]
    assert all(node.get("output_schema") for node in wire["nodes"])
