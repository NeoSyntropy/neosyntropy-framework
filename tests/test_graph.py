from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    COMBINE_SCHEMA_SUFFIX,
    REASONING_OUTPUT_SCHEMA,
    CombineNode,
    Edge,
    EmptyOutput,
    FSM,
    FSMValidationError,
    Group,
    Node,
    OpenInput,
    ReasoningNode,
    SchemaNode,
    TextOutput,
    Workflow,
)

from .conftest import build_graph


def make_nodes(*ids: str, fallback: str = "Fallback") -> list[Node]:
    nodes = [Node(id=identifier, input_schema=OpenInput, output_schema=EmptyOutput) for identifier in ids]
    nodes.append(Node(id=fallback, is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput))
    return nodes


def test_requires_exactly_one_fallback():
    with pytest.raises(FSMValidationError, match="exactly one dedicated fallback"):
        FSM(nodes=[Node(id="A", input_schema=OpenInput, output_schema=EmptyOutput)])
    with pytest.raises(FSMValidationError, match="exactly one dedicated fallback"):
        FSM(
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


def test_schema_reasoning_combine_constructors():
    schema = SchemaNode(
        id="Extract",
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="extract json",
    )
    assert schema.mode == "schema_extraction"
    assert schema.kind == "schema"
    assert schema.tools == ()

    reasoning = ReasoningNode(
        id="Scout",
        input_schema=OpenInput,
        tools=("lookup",),
        prompt="gather notes",
    )
    assert reasoning.mode == "reasoning"
    assert reasoning.kind == "reasoning"
    assert reasoning.output_schema == REASONING_OUTPUT_SCHEMA

    combine = CombineNode(
        id="Lane",
        input_schema=OpenInput,
        tools=("lookup",),
        output_schema=TextOutput,
        prompt="reason then extract",
    )
    nodes, edges = combine.expand()
    assert [item.id for item in nodes] == ["Lane", f"Lane{COMBINE_SCHEMA_SUFFIX}"]
    assert nodes[0].mode == "reasoning"
    assert nodes[1].mode == "schema_extraction"
    assert edges[0].source == "Lane"
    assert edges[0].target == f"Lane{COMBINE_SCHEMA_SUFFIX}"


def test_workflow_builds_linear_fsm():
    first = Node(id="A", input_schema=OpenInput, output_schema=EmptyOutput)
    second = Node(id="B", input_schema=OpenInput, output_schema=EmptyOutput)
    fallback = Node(
        id="Fallback",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=EmptyOutput,
    )
    fsm = Workflow([first, second], fallback=fallback)
    assert set(fsm.nodes) == {"A", "B", "Fallback"}
    assert fsm.allows("Start", "A")
    assert fsm.allows("A", "B")
    assert fsm.allows("B", "End")
    assert fsm.allows("Start", "Fallback")
    assert not fsm.allows("Start", "B")


def test_workflow_chains_through_combine_schema_exit():
    combine = CombineNode(
        id="Lane",
        input_schema=OpenInput,
        tools=("lookup",),
        output_schema=TextOutput,
        prompt="reason then extract",
    )
    after = Node(id="After", input_schema=OpenInput, output_schema=EmptyOutput)
    fallback = SchemaNode(
        id="Fallback",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=EmptyOutput,
        prompt="stop",
    )
    fsm = Workflow([combine, after], fallback=fallback)
    schema_id = f"Lane{COMBINE_SCHEMA_SUFFIX}"
    assert fsm.allows("Start", "Lane")
    assert fsm.allows("Lane", schema_id)
    assert fsm.allows(schema_id, "After")
    assert fsm.allows("After", "End")


def test_workflow_requires_fallback_and_nonempty_sequence():
    node = Node(id="A", input_schema=OpenInput, output_schema=EmptyOutput)
    with pytest.raises(FSMValidationError, match="at least one node"):
        Workflow([], fallback=Node(id="F", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput))
    with pytest.raises(FSMValidationError, match="requires a fallback"):
        Workflow([node])


def test_authored_routers_compile_to_edges_without_manual_edges():
    from neosyntropy import DeterministicRouter, SemanticRouter, node

    billing = Group(name="billing", description="Pay flows")

    @node(id="PayInvoice", group="billing", input_schema=OpenInput, output_schema=EmptyOutput)
    def pay(ctx):
        return ctx.result(output={}, next_state="End")

    @node(id="Login", input_schema=OpenInput, output_schema=EmptyOutput)
    def login(ctx):
        return ctx.result(
            output={},
            state_updates={"token_valid": True},
            next_state="End",
        )

    @node(id="GeneralChat", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput)
    def general(ctx):
        return ctx.result(output={})

    intent = SemanticRouter(
        id="CustomerIntent",
        routes={"wants_to_pay": billing},
        fallback_node=general,
    )
    auth = DeterministicRouter(
        id="CheckAuth",
        rules=[
            (lambda ctx: ctx.state.get("token_valid") is True, intent),
            (lambda ctx: ctx.state.get("token_valid") is False, login),
        ],
    )
    fsm = FSM(
        nodes=[pay, login, general],
        groups=[billing],
        routers=[auth],
        entry=auth,
        edges=[
            Edge(source="Login", target="End", kind="deterministic"),
            Edge(source="PayInvoice", target="End", kind="deterministic"),
        ],
    )
    assert fsm.is_router_state("CheckAuth")
    assert fsm.is_router_state("CustomerIntent")
    assert fsm.allows("Start", "CheckAuth")
    assert fsm.allows("CheckAuth", "CustomerIntent")
    assert fsm.allows("CheckAuth", "Login")
    assert fsm.allows("CustomerIntent", "PayInvoice")
    assert fsm.allows("CustomerIntent", "GeneralChat")

    # Unauthenticated: Start → CheckAuth → Login
    result = fsm.run(intent="pay please", state={"token_valid": False})
    assert not result.rejected
    assert any(
        item.node_id == "Login"
        for step in result.steps
        for item in step.results
    )


def test_group_authors_nodes_routers_entry_and_edges():
    from neosyntropy import DeterministicRouter, SemanticRouter, node

    billing = Group(name="billing", description="Pay flows")

    @billing.node(id="ValidateCard", input_schema=OpenInput, output_schema=EmptyOutput)
    def validate(ctx):
        return ctx.result(
            output={},
            state_updates={"card_valid": True},
        )

    @billing.node(id="ProcessPayment", input_schema=OpenInput, output_schema=EmptyOutput)
    def pay(ctx):
        return ctx.result(output={}, state_updates={"paid": True})

    @billing.node(id="SendReceipt", input_schema=OpenInput, output_schema=EmptyOutput)
    def receipt(ctx):
        return ctx.result(output={}, state_updates={"receipt": True}, next_state="End")

    @billing.node(id="RejectCard", input_schema=OpenInput, output_schema=EmptyOutput)
    def reject(ctx):
        return ctx.result(output={}, next_state="End")

    @node(id="GeneralChat", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput)
    def general(ctx):
        return ctx.result(output={})

    internal_logic = DeterministicRouter(
        id="BillingLogic",
        rules=[
            (lambda ctx: ctx.state.get("card_valid") is True, "ProcessPayment"),
            (lambda ctx: ctx.state.get("card_valid") is False, "RejectCard"),
        ],
    )
    post_pay = SemanticRouter(
        id="PostPayIntent",
        routes={"wants_receipt": "SendReceipt"},
        fallback_node=general,
    )
    billing.routers = [internal_logic, post_pay]
    billing.entry = "ValidateCard"
    billing.add_edge("ValidateCard", "BillingLogic")
    billing.add_edge("ProcessPayment", "PostPayIntent")

    intent = SemanticRouter(
        id="CustomerIntent",
        routes={"wants_to_pay": billing},
        fallback_node=general,
    )
    fsm = FSM(
        nodes=[general],
        groups=[billing],
        routers=[intent],
        entry=intent,
        edges=[
            Edge(source="SendReceipt", target="End", kind="deterministic"),
            Edge(source="RejectCard", target="End", kind="deterministic"),
        ],
    )

    assert set(billing.nodes) == {
        "ValidateCard",
        "ProcessPayment",
        "SendReceipt",
        "RejectCard",
    }
    assert fsm.nodes["ValidateCard"].group == "billing"
    assert fsm.is_router_state("BillingLogic")
    assert fsm.is_router_state("PostPayIntent")
    assert fsm.allows("Start", "CustomerIntent")
    # Group entry: semantic edge to billing lands only on ValidateCard.
    assert fsm.allows("CustomerIntent", "ValidateCard")
    assert not fsm.allows("CustomerIntent", "ProcessPayment")
    assert fsm.allows("ValidateCard", "BillingLogic")
    assert fsm.allows("BillingLogic", "ProcessPayment")
    assert fsm.allows("ProcessPayment", "PostPayIntent")
    assert fsm.allows("PostPayIntent", "SendReceipt")

    result = fsm.run(intent="pay please", state={})
    assert not result.rejected
    executed = {
        item.node_id
        for step in result.steps
        for item in step.results
    }
    assert "ValidateCard" in executed
    assert "ProcessPayment" in executed
    assert "SendReceipt" in executed
    assert result.state.get("paid") is True
    assert result.state.get("receipt") is True
    assert result.final_state == "End"


def test_control_graph_manifest_includes_routers_groups_and_input_schema():
    from neosyntropy import DeterministicRouter, control_graph_manifest, node

    billing = Group(name="billing")

    @billing.node(id="ValidateCard", input_schema=OpenInput, output_schema=EmptyOutput)
    def validate(ctx):
        return ctx.result(output={})

    @billing.node(id="ProcessPayment", input_schema=OpenInput, output_schema=EmptyOutput)
    def pay(ctx):
        return ctx.result(output={}, next_state="End")

    @node(id="Fallback", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput)
    def fallback(ctx):
        return ctx.result(output={})

    logic = DeterministicRouter(
        id="BillingLogic",
        rules=[(lambda ctx: True, "ProcessPayment")],
    )
    billing.routers = [logic]
    billing.entry = "ValidateCard"
    billing.add_edge("ValidateCard", "BillingLogic")

    fsm = FSM(
        nodes=[fallback],
        groups=[billing],
        routers=[logic],
        entry="ValidateCard",
        edges=[Edge(source="ProcessPayment", target="End", kind="deterministic")],
    )
    manifest = control_graph_manifest(fsm)
    assert "BillingLogic" in manifest["routers"]
    assert any(
        group.get("name") == "billing" and group.get("entry") == "ValidateCard"
        for group in manifest["groups"]
    )
    by_id = {node["id"]: node for node in manifest["nodes"]}
    assert by_id["ValidateCard"]["input_schema"] is not None
    assert by_id["ValidateCard"]["output_schema"] is not None


def test_fsm_run_wires_public_client():
    from neosyntropy import Client, node

    @node(id="A", input_schema=OpenInput, output_schema=EmptyOutput)
    def step_a(ctx):
        return ctx.result(output={}, next_state="End")

    @node(id="Fallback", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput)
    def fallback(ctx):
        return ctx.result(output={})

    fsm = Workflow([step_a], fallback=fallback)
    client = Client(api_key="test-key", project_id="proj_test")
    manager = fsm._control_manager(client=client)
    assert manager._backend is client._as_backend()
    assert manager._backend.api_key == "test-key"
    assert manager._backend.project_id == "proj_test"

    # Without a client, local handlers still run offline. Start is implicit.
    result = fsm.run(intent="go", state={})
    assert not result.rejected
    assert result.final_state == "End"
    assert result.audit.initial_state == "Start"


def test_graph_flattens_combine_node():
    graph = FSM(
        nodes=[
            CombineNode(
                id="Lane",
                input_schema=OpenInput,
                tools=("lookup",),
                output_schema=TextOutput,
                prompt="reason then extract",
                group="ops",
            ),
            SchemaNode(
                id="Fallback",
                is_fallback=True,
                input_schema=OpenInput,
                output_schema=EmptyOutput,
                prompt="stop",
            ),
        ],
        edges=[
            Edge(source="Start", target="Lane", kind="deterministic"),
            Edge(source=f"Lane{COMBINE_SCHEMA_SUFFIX}", target="End", kind="deterministic"),
            Edge(source="Start", target="Fallback", kind="fallback"),
        ],
    )
    assert set(graph.nodes) == {"Lane", f"Lane{COMBINE_SCHEMA_SUFFIX}", "Fallback"}
    assert any(
        edge.source == "Lane" and edge.target == f"Lane{COMBINE_SCHEMA_SUFFIX}"
        for edge in graph.edges
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
    with pytest.raises(FSMValidationError, match="duplicate node id"):
        FSM(
            nodes=[
                Node(id="A", input_schema=OpenInput, output_schema=EmptyOutput),
                Node(id="A", input_schema=OpenInput, output_schema=EmptyOutput),
                Node(id="F", is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput),
            ]
        )


def test_rejects_unknown_edge_endpoints():
    with pytest.raises(FSMValidationError, match="not a known node"):
        FSM(
            nodes=make_nodes("A"),
            edges=[Edge(source="A", target="Ghost")],
        )


def test_start_end_reachability():
    # B is on an island: unreachable from Start.
    with pytest.raises(FSMValidationError, match="unreachable from Start"):
        FSM(
            nodes=make_nodes("A", "B", "C"),
            edges=[
                Edge(source="Start", target="A"),
                Edge(source="A", target="End"),
                Edge(source="B", target="C"),
            ],
        )
    # A reaches a dead end that cannot reach End.
    with pytest.raises(FSMValidationError, match="cannot reach End"):
        FSM(
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
    graph = FSM(
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

    graph = FSM(
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
    graph = FSM(
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


def _graph_with_entry_schema(**kwargs) -> FSM:
    return FSM(nodes=make_nodes("A"), input_schema=EntryInput, **kwargs)


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
    assert FSM(nodes=make_nodes("A")).entry_input_error({"anything": 1}) is None


def test_entry_schema_accepts_a_raw_json_schema():
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    graph = FSM(nodes=make_nodes("A"), input_schema=schema)
    assert graph.input_schema == schema
    assert graph.input_model is None


def test_rejects_an_unusable_entry_schema():
    with pytest.raises(FSMValidationError, match="input_schema must be"):
        FSM(nodes=make_nodes("A"), input_schema={})
