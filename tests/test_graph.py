from __future__ import annotations

import pytest

from neosyntropy import Edge, Graph, GraphValidationError, Group, Node

from .conftest import build_graph


def make_nodes(*ids: str, fallback: str = "Fallback") -> list[Node]:
    nodes = [Node(id=identifier) for identifier in ids]
    nodes.append(Node(id=fallback, is_fallback=True))
    return nodes


def test_requires_exactly_one_fallback():
    with pytest.raises(GraphValidationError, match="exactly one dedicated fallback"):
        Graph(nodes=[Node(id="A")])
    with pytest.raises(GraphValidationError, match="exactly one dedicated fallback"):
        Graph(
            nodes=[
                Node(id="A", is_fallback=True),
                Node(id="B", is_fallback=True),
            ]
        )


def test_rejects_duplicate_node_ids():
    with pytest.raises(GraphValidationError, match="duplicate node id"):
        Graph(nodes=[Node(id="A"), Node(id="A"), Node(id="F", is_fallback=True)])


def test_rejects_unknown_edge_endpoints():
    with pytest.raises(GraphValidationError, match="unknown nodes"):
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
    edge = Edge.model_validate({"from": "A", "to": "B", "label": "next"})
    assert edge.source == "A"
    assert edge.target == "B"


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
        nodes=[Node(id="A", group="ops"), Node(id="F", is_fallback=True)],
        groups=[Group(name="ops", description="Operations")],
    )
    assert graph.groups["ops"].description == "Operations"
