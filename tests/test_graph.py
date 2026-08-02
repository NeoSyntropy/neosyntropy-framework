from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from neosyntropy import OpenInput, EmptyOutput, Edge, Graph, GraphValidationError, Group, Node

from .conftest import build_graph


def make_nodes(*ids: str, fallback: str = "Fallback") -> list[Node]:
    nodes = [Node(id=identifier, input_schema=OpenInput, output_schema=EmptyOutput) for identifier in ids]
    nodes.append(Node(id=fallback, is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput))
    return nodes


def test_requires_exactly_one_fallback():
    with pytest.raises(GraphValidationError, match="exactly one dedicated fallback"):
        Graph(nodes=[Node(id="A", input_schema=OpenInput, output_schema=EmptyOutput)])
    with pytest.raises(GraphValidationError, match="exactly one dedicated fallback"):
        Graph(
            nodes=[
                Node(id="A", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput),
                Node(id="B", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput),
            ]
        )


def test_rejects_nodes_without_output_schema():
    with pytest.raises(Exception, match="requires output_schema"):
        Node(id="A", input_schema=OpenInput)


def test_rejects_nodes_without_input_schema():
    with pytest.raises(Exception, match="requires input_schema"):
        Node(id="A", output_schema=EmptyOutput)


def test_node_input_schema_is_required_and_enforced():
    from pydantic import BaseModel, ConfigDict

    class NeedsAmount(BaseModel):
        model_config = ConfigDict(extra="forbid")

        amount: float

    open_node = Node(id="A", input_schema=OpenInput, output_schema=EmptyOutput)
    assert open_node.input_schema is not None
    assert open_node.input_error({"anything": True}) is None

    guarded = Node(id="B", input_schema=NeedsAmount, output_schema=EmptyOutput)
    assert guarded.input_error({}) is not None
    assert guarded.input_error({"amount": 10.0}) is None
    assert guarded.input_error({"amount": 10.0, "extra": 1}) is not None


def test_infers_mode_from_tools():
    assert Node(id="A", input_schema=OpenInput, output_schema=EmptyOutput).mode == "schema_extraction"
    assert (
        Node(id="B", tools=("lookup",), input_schema=OpenInput, output_schema=EmptyOutput).mode == "reasoning"
    )


def test_schema_extraction_rejects_tools():
    with pytest.raises(Exception, match="schema_extraction.*cannot declare tools"):
        Node(
            id="A",
            mode="schema_extraction",
            tools=("lookup",),
            input_schema=OpenInput, output_schema=EmptyOutput,
        )


def test_rejects_duplicate_node_ids():
    with pytest.raises(GraphValidationError, match="duplicate node id"):
        Graph(
            nodes=[
                Node(id="A", input_schema=OpenInput, output_schema=EmptyOutput),
                Node(id="A", input_schema=OpenInput, output_schema=EmptyOutput),
                Node(id="F", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput),
            ]
        )


def test_rejects_unknown_edge_endpoints():
    with pytest.raises(GraphValidationError, match="not a known node"):
        Graph(
            nodes=make_nodes("A"),
            edges=[Edge(source="A", target="Ghost")],
        )


def test_start_end_reachability():
    # B is on an island: unreachable from Start.
    with pytest.raises(GraphValidationError, match="unreachable from Start"):
        Graph(
            nodes=make_nodes("A", "B", "C"),
            edges=[
                Edge(source="Start", target="A"),
                Edge(source="A", target="End"),
                Edge(source="B", target="C"),
            ],
        )
    # A reaches a dead end that cannot reach End.
    with pytest.raises(GraphValidationError, match="cannot reach End"):
        Graph(
            nodes=make_nodes("A", "B"),
            edges=[
                Edge(source="Start", target="A"),
                Edge(source="A", target="End"),
                Edge(source="A", target="B"),
            ],
        )


def test_allows_is_fail_closed_by_default(refund_graph):
    assert refund_graph.allows("Start", "VerifyIdentity")
    assert not refund_graph.allows("Start", "IssueRefund")
    permissive = build_graph(allow_unlisted_transitions=True)
    assert permissive.allows("Start", "IssueRefund")


def test_edge_accepts_from_to_aliases():
    edge = Edge.model_validate({"from": "A", "to": "B", "kind": "deterministic"})
    assert edge.source == "A"
    assert edge.target == "B"
    assert edge.kind == "deterministic"


def test_semantic_group_edge_permits_group_members():
    graph = Graph(
        nodes=[
            Node(id="A", group="ops", input_schema=OpenInput, output_schema=EmptyOutput),
            Node(id="B", group="ops", input_schema=OpenInput, output_schema=EmptyOutput),
            Node(id="C", group="other", input_schema=OpenInput, output_schema=EmptyOutput),
            Node(id="F", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput),
        ],
        edges=[
            Edge(source="Start", target="ops", kind="semantic", target_kind="group"),
            Edge(source="A", target="End", kind="deterministic"),
            Edge(source="B", target="End", kind="deterministic"),
            Edge(source="Start", target="F", kind="fallback"),
        ],
    )
    assert graph.allows("Start", "A")
    assert graph.allows("Start", "B")
    assert not graph.allows("Start", "C")
    assert graph.semantic_candidate_ids("Start") == {"A", "B"}


def test_guards_fail_closed():
    def broken_guard(state):
        raise RuntimeError("boom")

    graph = Graph(
        nodes=make_nodes("A", "B"),
        edges=[
            Edge(source="Start", target="A"),
            Edge(source="A", target="B", guard=lambda state: state.get("go", False)),
            Edge(source="A", target="End", guard=broken_guard),
            Edge(source="B", target="End"),
        ],
    )
    assert not graph.guard_allows("A", "B", {})
    assert graph.guard_allows("A", "B", {"go": True})
    # A raising guard denies, never fails open.
    assert not graph.guard_allows("A", "End", {})
    # No matching edge: guards are vacuously satisfied.
    assert graph.guard_allows("Ghost", "Elsewhere", {})


def test_groups_are_organizational_only(refund_graph):
    assert set(refund_graph.groups) == {"refunds"}
    members = {item.id for item in refund_graph.nodes_in_group("refunds")}
    assert members == {"VerifyIdentity", "CalculateRefund", "IssueRefund"}


def test_explicit_groups_merge_with_node_references():
    graph = Graph(
        nodes=[
            Node(id="A", group="ops", input_schema=OpenInput, output_schema=EmptyOutput),
            Node(id="F", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput),
        ],
        groups=[Group(name="ops", description="Operations")],
    )
    assert graph.groups["ops"].description == "Operations"


# --- entry contract -----------------------------------------------------------


class EntryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cart_id: str
    locale: str = "en"


def _graph_with_entry_schema(**kwargs) -> Graph:
    return Graph(nodes=make_nodes("A"), input_schema=EntryInput, **kwargs)


def test_entry_schema_keeps_optional_fields_optional():
    schema = _graph_with_entry_schema().input_schema
    assert schema["required"] == ["cart_id"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["locale"]["default"] == "en"


def test_entry_input_is_validated_fail_closed():
    graph = _graph_with_entry_schema()
    assert graph.entry_input_error({"cart_id": "cart-1"}) is None
    assert graph.entry_input_error({"cart_id": "cart-1", "locale": "he"}) is None
    assert "input schema" in (graph.entry_input_error({}) or "")
    assert "input schema" in (graph.entry_input_error({"cart_id": 7}) or "")
    # Unknown keys cannot smuggle state past the entry point.
    assert "input schema" in (
        graph.entry_input_error({"cart_id": "cart-1", "admin": True}) or ""
    )


def test_graph_without_entry_schema_accepts_any_input():
    assert Graph(nodes=make_nodes("A")).entry_input_error({"anything": 1}) is None


def test_entry_schema_accepts_a_raw_json_schema():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    graph = Graph(nodes=make_nodes("A"), input_schema=schema)
    assert graph.input_schema == schema
    assert graph.input_model is None


def test_rejects_an_unusable_entry_schema():
    with pytest.raises(GraphValidationError, match="input_schema must be"):
        Graph(nodes=make_nodes("A"), input_schema={})
