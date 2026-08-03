"""Active routing modes + billable transitions across a multi-group FSM.

Covers the control concept without changing it:

- semantic edges to groups scope which nodes the semantic router may pick
- nodes outside those groups stay illegal from Start
- semantic edges to concrete nodes stay node-scoped
- deterministic edges own the in-lane path after selection
- each committed transition burns a credit unit (fail-closed billing signal)
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from neosyntropy import (
    OpenInput,
    ControlManager,
    EmptyOutput,
    FSM,
    Group,
    RunRequest,
    edge_deterministic,
    edge_fallback,
    edge_semantic,
    node,
)

TRANSITION_RATE = Decimal("0.005")
STARTING_CREDITS = Decimal("1.00")

INTAKE_A = "intake.Alpha"
INTAKE_B = "intake.Beta"
FULFILL = "ops.Fulfill"  # ops group — reachable only via deterministic after intake
SPECIALIST = "special.Review"  # semantic node target (not a group)
FALLBACK = "desk.OutOfScope"


def build_routing_graph() -> FSM:
    @node(id=INTAKE_A, group="intake", input_schema=OpenInput, output_schema=EmptyOutput)
    def intake_alpha(ctx):
        """Intake for alpha-class requests."""
        return ctx.result(output={}, state_updates={"lane": "alpha"})

    @node(id=INTAKE_B, group="intake", input_schema=OpenInput, output_schema=EmptyOutput)
    def intake_beta(ctx):
        """Intake for beta-class requests."""
        return ctx.result(output={}, state_updates={"lane": "beta"})

    @node(id=FULFILL, group="ops", input_schema=OpenInput, output_schema=EmptyOutput)
    def fulfill(ctx):
        """Fulfill after intake."""
        return ctx.result(output={}, state_updates={"fulfilled": True})

    @node(id=SPECIALIST, group="special", input_schema=OpenInput, output_schema=EmptyOutput)
    def specialist(ctx):
        """Specialist review for edge cases."""
        return ctx.result(output={}, state_updates={"reviewed": True})

    @node(id=FALLBACK, is_fallback=True, input_schema=OpenInput, output_schema=EmptyOutput)
    def out_of_scope(ctx):
        """Safe stop."""
        return ctx.result(output={}, status="fallback")

    return FSM(
        nodes=[
            intake_alpha,
            intake_beta,
            fulfill,
            specialist,
            out_of_scope,
        ],
        edges=[
            # Semantic → group: only intake members are in scope from Start.
            edge_semantic("Start", "intake", target_kind="group"),
            # Semantic → node: a concrete specialist path.
            edge_semantic("Start", SPECIALIST),
            # Deterministic in-lane hops after selection.
            edge_deterministic(INTAKE_A, FULFILL),
            edge_deterministic(INTAKE_B, FULFILL),
            edge_deterministic(FULFILL, "End"),
            edge_deterministic(SPECIALIST, "End"),
            edge_fallback("Start", FALLBACK),
        ],
        groups=[
            Group(name="intake", description="Entry triage"),
            Group(name="ops", description="Fulfillment (not a Start semantic target)"),
            Group(name="special", description="Specialist lane"),
        ],
    )


class ActiveRoutingBackend:
    """Opaque control double that routes the way the backend cycle does.

    Selection is cue-driven so 100 samples exercise every mode. Internals
    (candidates, topology, plans) never appear in responses. Each committed
    transition debits the credit bank.
    """

    def __init__(self, credits: Decimal = STARTING_CREDITS) -> None:
        self.credits = credits
        self.charges: list[Decimal] = []
        self.scoped_starts: list[set[str]] = []
        self.selected: str | None = None
        self.current_state = "Start"
        self.state: dict[str, Any] = {}
        self.run_id = "routing-run"

    def _pick(self, current: str, intent: str) -> str | None:
        text = intent.lower()
        if current in {INTAKE_A, INTAKE_B}:
            return FULFILL
        if current in {FULFILL, SPECIALIST, FALLBACK, "End"}:
            return None
        if "specialist" in text or "edge case" in text:
            return SPECIALIST
        if "alpha" in text:
            return INTAKE_A
        if "beta" in text:
            return INTAKE_B
        if "fulfill" in text or "ops" in text:
            # Ops/fulfill is out of Start's semantic scope — must fall back.
            return FALLBACK
        return FALLBACK

    async def start_control_run(self, graph_manifest, request, *, category="general"):
        self.current_state = request.get("current_state", "Start")
        self.state = dict(request.get("state") or {})
        if self.current_state == "Start":
            self.scoped_starts.append({INTAKE_A, INTAKE_B, SPECIALIST})
        self.selected = self._pick(self.current_state, request["intent"])
        if self.selected is None:
            return {
                "run_id": self.run_id,
                "status": "completed",
                "current_state": self.current_state,
                "state": self.state,
                "step": None,
                "committed_transitions": [],
                "rejection": None,
                "completed": True,
            }
        return {
            "run_id": self.run_id,
            "status": "awaiting_execution",
            "current_state": self.current_state,
            "state": self.state,
            "step": {"step": 0, "nodes": [self.selected]},
            "committed_transitions": [],
            "rejection": None,
            "completed": False,
        }

    async def submit_control_results(
        self, run_id, *, results=None, client_rejection=None
    ):
        if client_rejection is not None:
            return {
                "run_id": run_id,
                "status": "rejected",
                "current_state": self.current_state,
                "state": self.state,
                "step": None,
                "committed_transitions": [],
                "rejection": client_rejection,
                "completed": False,
            }
        assert results is not None and len(results) == 1
        node_id = results[0]["node_id"]
        # Fallback answers without moving off Start (no billable hop).
        if node_id == FALLBACK:
            return {
                "run_id": run_id,
                "status": "completed",
                "current_state": self.current_state,
                "state": self.state,
                "step": None,
                "committed_transitions": [],
                "rejection": None,
                "completed": True,
            }
        # Mirror control_engine default: single non-fallback node → itself.
        committed = results[0].get("next_state") or node_id
        if committed == self.current_state:
            return {
                "run_id": run_id,
                "status": "completed",
                "current_state": self.current_state,
                "state": self.state,
                "step": None,
                "committed_transitions": [],
                "rejection": None,
                "completed": True,
            }
        if self.credits < TRANSITION_RATE:
            return {
                "run_id": run_id,
                "status": "rejected",
                "current_state": self.current_state,
                "state": self.state,
                "step": None,
                "committed_transitions": [],
                "rejection": "insufficient credits",
                "completed": False,
            }
        previous = self.current_state
        self.credits -= TRANSITION_RATE
        self.charges.append(TRANSITION_RATE)
        self.current_state = committed
        updates = results[0].get("state_updates") or {}
        self.state.update(updates)
        return {
            "run_id": run_id,
            "status": "completed",
            "current_state": committed,
            "state": self.state,
            "step": None,
            "committed_transitions": [f"{previous}->{committed}"],
            "rejection": None,
            "completed": True,
        }

    async def generate(self, prompt, *, schema=None, purpose="node"):
        return "{}"


def _samples(n: int = 100) -> list[tuple[str, str]]:
    """Deterministic 100 intents: group semantic, node semantic, out-of-scope."""
    cases: list[tuple[str, str]] = []
    templates = [
        ("alpha request number {i}", INTAKE_A),
        ("beta request number {i}", INTAKE_B),
        ("specialist edge case {i}", SPECIALIST),
        ("please fulfill ops file {i}", FALLBACK),
        ("write a poem about {i}", FALLBACK),
    ]
    i = 0
    while len(cases) < n:
        template, expects = templates[i % len(templates)]
        cases.append((template.format(i=i), expects))
        i += 1
    return cases


def test_graph_scopes_semantic_group_in_and_out() -> None:
    graph = build_routing_graph()
    assert graph.allows("Start", INTAKE_A)
    assert graph.allows("Start", INTAKE_B)
    assert graph.allows("Start", SPECIALIST)
    # ops.Fulfill is in the ops group — out of Start's semantic scope.
    assert not graph.allows("Start", FULFILL)
    assert graph.semantic_candidate_ids("Start") == {INTAKE_A, INTAKE_B, SPECIALIST}
    assert graph.allows(INTAKE_A, FULFILL)
    assert graph.allows(INTAKE_B, FULFILL)


def test_one_hundred_samples_route_and_burn_credits() -> None:
    graph = build_routing_graph()
    samples = _samples(100)
    assert len(samples) == 100

    backend = ActiveRoutingBackend(credits=STARTING_CREDITS)
    manager = ControlManager(graph, backend=backend)  # type: ignore[arg-type]

    reached: dict[str, int] = {}
    billable = 0
    for intent, expects in samples:
        result = manager.run(
            RunRequest(intent=intent, current_state="Start", state={})
        )
        assert not result.rejected, (intent, result.rejection)
        if expects == FALLBACK:
            assert result.final_state == "Start"
            assert result.audit.committed_transitions == []
        else:
            assert result.final_state == expects
            assert result.audit.committed_transitions == [f"Start->{expects}"]
            billable += 1
        reached[expects] = reached.get(expects, 0) + 1

        if result.final_state in {INTAKE_A, INTAKE_B}:
            follow = manager.run(
                RunRequest(
                    intent=intent,
                    current_state=result.final_state,
                    state=result.state,
                    prior_executions=[
                        {
                            "node_id": result.final_state,
                            "status": "succeeded",
                            "output": {},
                            "state_updates": {},
                        }
                    ],
                )
            )
            assert not follow.rejected, follow.rejection
            assert follow.final_state == FULFILL
            assert follow.audit.committed_transitions == [
                f"{result.final_state}->{FULFILL}"
            ]
            billable += 1

    assert set(reached) == {INTAKE_A, INTAKE_B, SPECIALIST, FALLBACK}
    assert billable == len(backend.charges)
    expected = STARTING_CREDITS - (TRANSITION_RATE * billable)
    assert backend.credits == expected
    assert backend.credits < STARTING_CREDITS
    for scoped in backend.scoped_starts:
        assert FULFILL not in scoped
        assert scoped == {INTAKE_A, INTAKE_B, SPECIALIST}


def test_insufficient_credits_reject_without_commit() -> None:
    graph = build_routing_graph()
    backend = ActiveRoutingBackend(credits=Decimal("0"))
    manager = ControlManager(graph, backend=backend)  # type: ignore[arg-type]
    result = manager.run(
        RunRequest(intent="alpha request", current_state="Start", state={})
    )
    assert result.rejected
    assert "credits" in (result.rejection or "")
    assert result.audit.committed_transitions == []
    assert backend.credits == Decimal("0")


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("alpha intake please", INTAKE_A),
        ("beta intake please", INTAKE_B),
        ("specialist edge case", SPECIALIST),
        ("fulfill ops ledger", FALLBACK),
    ],
)
def test_active_routing_modes_parametrized(intent: str, expected: str) -> None:
    graph = build_routing_graph()
    backend = ActiveRoutingBackend()
    before = backend.credits
    result = ControlManager(graph, backend=backend).run(  # type: ignore[arg-type]
        RunRequest(intent=intent, current_state="Start", state={})
    )
    assert not result.rejected, result.rejection
    if expected == FALLBACK:
        assert result.final_state == "Start"
        assert backend.credits == before
    else:
        assert result.final_state == expected
        assert backend.credits == before - TRANSITION_RATE
