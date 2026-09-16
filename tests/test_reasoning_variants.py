"""Public reasoning factories keep dynamic and fixed control flow distinct."""
import pytest

from neosyntropy import (
    DeterministicReasoningNode,
    ReasoningNode,
    ReasoningStep,
    StochasticReasoningNode,
)
from neosyntropy.core.graph import END, FSM
from neosyntropy.core.node import Node


INPUT = {"type": "object"}


def test_stochastic_preserves_reasoning_execution_contract():
    node = StochasticReasoningNode(
        "Investigate", input_schema=INPUT, prompt="Investigate the request",
        tools=["lookup"], metadata={"owner": "support"},
    )
    assert isinstance(node, Node)
    assert node.mode == "reasoning"
    assert node.tools == ("lookup",)
    assert node.metadata["owner"] == "support"
    assert node.model_dump() == ReasoningNode(
        "Investigate", input_schema=INPUT, prompt="Investigate the request",
        tools=["lookup"], metadata={"owner": "support"},
    ).model_dump()


def test_deterministic_wires_order_and_preserves_step_contracts():
    flow = DeterministicReasoningNode(
        "Investigate", input_schema=INPUT,
        steps=[ReasoningStep("Look up", tools=["lookup"]), ReasoningStep("Summarize")],
        output_schema={"type": "string"},
    )
    assert isinstance(flow, FSM)
    nodes = flow.nodes
    assert nodes["Investigate_step_0"].tools == ("lookup",)
    assert nodes["Investigate_step_1"].tools == ()
    assert nodes["Investigate_step_1"].output_schema == {"type": "string"}
    assert nodes["Investigate_step_0"].mode == "reasoning"
    links = {(edge.source, edge.target) for edge in flow.edges}
    assert ("Investigate_step_0", "Investigate_step_1") in links
    assert ("Investigate_step_1", END) in links


def test_deterministic_rejects_empty_steps():
    with pytest.raises(ValueError, match="non-empty steps"):
        DeterministicReasoningNode("Empty", input_schema=INPUT, steps=[])


def test_stochastic_rejects_fixed_steps():
    with pytest.raises(TypeError):
        StochasticReasoningNode(
            "Mixed", input_schema=INPUT, prompt="Think", steps=[ReasoningStep("Think")],
        )


def test_legacy_steps_still_build_a_workflow():
    assert isinstance(ReasoningNode(
        "Legacy", input_schema=INPUT, steps=[ReasoningStep("Think")],
    ), FSM)
