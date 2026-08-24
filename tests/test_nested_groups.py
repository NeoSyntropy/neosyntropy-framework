"""Nested groups (groups-in-groups) compile into the parent FSM."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    FSM,
    Group,
    OpenInput,
    SchemaNode,
    SemanticRouter,
    TextOutput,
    edge_deterministic,
)
from neosyntropy.monitor.graph.manifest import graph_manifest


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str


def test_nested_groups_expand_and_manifest_parent() -> None:
    leaf_a = SchemaNode(
        id="DoA",
        input_schema=OpenInput,
        output_schema=_Out,
        prompt="a",
    )
    leaf_b = SchemaNode(
        id="DoB",
        input_schema=OpenInput,
        output_schema=_Out,
        prompt="b",
    )
    skill_a = Group(
        name="skill_a",
        entry=leaf_a,
        nodes=[leaf_a],
        edges=[edge_deterministic("DoA", "End")],
    )
    skill_b = Group(
        name="skill_b",
        entry=leaf_b,
        nodes=[leaf_b],
        edges=[edge_deterministic("DoB", "End")],
    )
    fallback = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="fallback",
    )
    skill_router = SemanticRouter(
        id="PhaseSkillRouter",
        input_schema=OpenInput,
        routes={"a": skill_a, "b": skill_b},
        fallback_node=fallback,
    )
    phase = Group(
        name="phase_one",
        entry=skill_router,
        routers=[skill_router],
        groups=[skill_a, skill_b],
        nodes=[],
        namespace=False,
    )

    assert skill_a.parent == "phase_one"
    assert skill_b.parent == "phase_one"
    assert {child.name for child in phase.child_groups()} == {"skill_a", "skill_b"}

    fsm = FSM(
        entry=skill_router,
        nodes=[fallback],
        groups=[phase],
        routers=[skill_router],
        validate_reachability=False,
    )
    assert "phase_one" in fsm.groups
    assert "skill_a" in fsm.groups
    assert fsm.groups["skill_a"].parent == "phase_one"
    assert "skill_a__DoA" in fsm.nodes
    assert "skill_b__DoB" in fsm.nodes

    manifest = graph_manifest(fsm)
    by_name = {item["name"]: item for item in manifest["groups"]}
    assert by_name["skill_a"]["parent"] == "phase_one"
    assert "parent" not in by_name["phase_one"]
    router_nodes = [n for n in manifest["nodes"] if n["id"] == "PhaseSkillRouter"]
    assert router_nodes and router_nodes[0]["group"] == "phase_one"
