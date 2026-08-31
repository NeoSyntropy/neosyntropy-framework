"""Tier 1 BMAD FSM: PhaseRouter plus the analyst 2-step skill.

Mirrors neo-code ``PhaseRouter`` labels and ``bmad_agent_analyst`` nodes.
Non-analysis phases land on stub SchemaNodes so routing is measurable
without pulling the full neo-code graph.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from neosyntropy import (
    FSM,
    OpenInput,
    ReasoningNode,
    SchemaNode,
    TextOutput,
    ToolRegistry,
    edge_deterministic,
    tool,
)
from neosyntropy.core.routing.semantic import SemanticRouter

ROUTER_PROVIDER = "neosyntropy/base"
VERTEX_MODEL = "glm-5-maas"

registry = ToolRegistry()

PHASE_ROUTES: dict[str, str] = {
    "analysis": "AnalystPhase",
    "plan": "PlanStub",
    "solutioning": "SolutioningStub",
    "implementation": "ImplementationStub",
    "core": "CoreStub",
    "help": "HelpStub",
}


class NeoCodeActivation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_request: str = ""
    is_headless: bool = False
    project_workspace: str = "."


class FinalizeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    analysis_report: str


class PhaseStubOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phase: str


class MemlogArgs(BaseModel):
    workspace: str
    entry_type: str = Field(
        description="One of: decision, change, override, assumption, event"
    )
    text: str


@tool(registry=registry)
def append_memlog(args: MemlogArgs) -> dict:
    """Append a decision or event to the .memlog.md audit trail."""
    print(f"[TOOL] Appending {args.entry_type} to memlog: {args.text}")
    return {"status": "success", "appended": True}


class ReadFileArgs(BaseModel):
    filepath: str


@tool(registry=registry)
def read_workspace_file(args: ReadFileArgs) -> dict:
    """Read a workspace file (e.g. existing brief or notes)."""
    print(f"[TOOL] Reading file: {args.filepath}")
    return {"status": "success", "content": "# Simulated workspace file"}


def _phase_stub(node_id: str, phase: str) -> SchemaNode:
    return SchemaNode(
        id=node_id,
        input_schema=OpenInput,
        output_schema=PhaseStubOutput,
        provider=VERTEX_MODEL,
        prompt=f"Return JSON with phase set to {phase!r}. Do not invent other fields.",
    )


def build_fsm() -> FSM:
    """Compile the Tier 1 graph: PhaseRouter → analyst 2-step or phase stubs."""
    analyst_node = ReasoningNode(
        id="AnalystPhase",
        input_schema=OpenInput,
        provider=VERTEX_MODEL,
        prompt=(
            "You are a systems analyst. Analyze the provided project context, "
            "identify constraints, and determine system requirements. "
            "Use tools to read workspace files and append decisions to the memlog."
        ),
        tools=("append_memlog", "read_workspace_file"),
    )
    finalize_node = SchemaNode(
        id="FinalizeNode",
        input_schema=OpenInput,
        output_schema=FinalizeOutput,
        provider=VERTEX_MODEL,
        prompt="Generate a structured systems analysis report.",
    )
    plan_stub = _phase_stub("PlanStub", "plan")
    solutioning_stub = _phase_stub("SolutioningStub", "solutioning")
    implementation_stub = _phase_stub("ImplementationStub", "implementation")
    core_stub = _phase_stub("CoreStub", "core")
    help_stub = _phase_stub("HelpStub", "help")
    out_of_scope = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        provider=VERTEX_MODEL,
        prompt="Politely refuse out-of-scope requests and suggest a BMAD skill.",
    )

    phase_router = SemanticRouter(
        id="PhaseRouter",
        input_schema=NeoCodeActivation,
        routes={
            "analysis": analyst_node,
            "plan": plan_stub,
            "solutioning": solutioning_stub,
            "implementation": implementation_stub,
            "core": core_stub,
            "help": help_stub,
        },
        fallback_node=out_of_scope,
        description="Pick BMAD phase (or help) from the user request",
        provider=ROUTER_PROVIDER,
    )

    return FSM(
        entry=phase_router,
        nodes=[
            analyst_node,
            finalize_node,
            plan_stub,
            solutioning_stub,
            implementation_stub,
            core_stub,
            help_stub,
            out_of_scope,
        ],
        routers=[phase_router],
        edges=[
            edge_deterministic("AnalystPhase", "FinalizeNode"),
            edge_deterministic("FinalizeNode", "End"),
            edge_deterministic("PlanStub", "End"),
            edge_deterministic("SolutioningStub", "End"),
            edge_deterministic("ImplementationStub", "End"),
            edge_deterministic("CoreStub", "End"),
            edge_deterministic("HelpStub", "End"),
            edge_deterministic("OutOfScope", "End"),
        ],
    )


fsm = build_fsm()

__all__ = [
    "FinalizeOutput",
    "NeoCodeActivation",
    "PHASE_ROUTES",
    "ROUTER_PROVIDER",
    "VERTEX_MODEL",
    "build_fsm",
    "fsm",
    "registry",
]
