"""Unit tests for FSM-like Group(entry=, nodes=, edges=) authoring."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    FSM,
    Group,
    OpenInput,
    SchemaNode,
    TextOutput,
    edge_deterministic,
)


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str


def test_group_fsm_like_constructor_compiles_into_parent_fsm() -> None:
    a = SchemaNode(
        id="StepA",
        input_schema=OpenInput,
        output_schema=_Out,
        prompt="step a",
    )
    b = SchemaNode(
        id="StepB",
        input_schema=OpenInput,
        output_schema=_Out,
        prompt="step b",
    )
    skill = Group(
        name="demo_skill",
        entry=a,
        nodes=[a, b],
        edges=[
            edge_deterministic("StepA", "StepB"),
            edge_deterministic("StepB", "End"),
        ],
    )
    assert skill.entry_id() == "demo_skill__StepA"
    assert "demo_skill__StepA" in skill.nodes
    assert "demo_skill__StepB" in skill.nodes

    fallback = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="fallback",
    )
    fsm = FSM(
        entry="demo_skill__StepA",
        nodes=[fallback],
        groups=[skill],
        edges=[edge_deterministic("demo_skill__StepA", "demo_skill__StepB")],
        validate_reachability=False,
    )
    assert "demo_skill__StepA" in fsm.nodes
    assert "demo_skill" in fsm.groups
