"""Harbor Signal Desk — CombineNode lanes (reasoning → schema).

Each operational group is authored as one :class:`CombineNode` that expands
into two FSM states:

1. **Reasoning** (``{id}``) — may call tools; output is plain working notes.
2. **Schema extraction** (``{id}.Schema``) — no tools; reads reasoning text
   plus tool evidence from state and returns a typed JSON reply.

Groups: ``berth``, ``cargo``, ``pilot``. Fallback is a :func:`SchemaNode`.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from neosyntropy import (
    COMBINE_SCHEMA_SUFFIX,
    CombineNode,
    FSM,
    Group,
    Node,
    NodeContext,
    NodeResult,
    OpenInput,
    SchemaNode,
    ToolRegistry,
    edge_deterministic,
    edge_fallback,
    edge_semantic,
    graph_manifest,
    tool,
)

BERTH_SCOUT = "berth.Scout"
BERTH_CLEARANCE = f"{BERTH_SCOUT}{COMBINE_SCHEMA_SUFFIX}"
CARGO_INSPECT = "cargo.Inspect"
CARGO_MANIFEST = f"{CARGO_INSPECT}{COMBINE_SCHEMA_SUFFIX}"
PILOT_BRIEF = "pilot.Brief"
PILOT_ADVISORY = f"{PILOT_BRIEF}{COMBINE_SCHEMA_SUFFIX}"
FALLBACK = "signal.OutOfScope"


# The entry contract. Every lane needs a different handle on the caller's
# request, so the desk accepts any one of them — but never a bare call and
# never a key the desk does not understand.
HARBOR_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "vessel": {"type": "string", "description": "Vessel asking for a slip"},
        "crate_id": {"type": "string", "description": "Freight awaiting weigh-in"},
        "channel": {"type": "string", "description": "Channel needing an advisory"},
    },
    "anyOf": [
        {"required": ["vessel"]},
        {"required": ["crate_id"]},
        {"required": ["channel"]},
    ],
}


class SlipArgs(BaseModel):
    vessel: str


class CrateArgs(BaseModel):
    crate_id: str


class TideArgs(BaseModel):
    channel: str


class BerthClearance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guest_text: str = Field(min_length=1)
    slip: str
    status: Literal["cleared", "hold", "diverted"]
    evidence_tools: list[str] = Field(default_factory=list)


class CargoManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guest_text: str = Field(min_length=1)
    crate_id: str
    disposition: Literal["stow", "inspect", "quarantine"]
    kilograms: float = Field(ge=0)
    evidence_tools: list[str] = Field(default_factory=list)


class PilotAdvisory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guest_text: str = Field(min_length=1)
    channel: str
    window: str
    advice: Literal["proceed", "wait", "reroute"]
    evidence_tools: list[str] = Field(default_factory=list)


class OutOfScopeReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guest_text: str = Field(min_length=1)
    department: Literal["out_of_scope"] = "out_of_scope"


def build_harbor_tools() -> ToolRegistry:
    registry = ToolRegistry()

    @tool(registry=registry)
    def lookup_slip(args: SlipArgs) -> dict:
        """Find an open berth slip and tide window for a vessel."""
        key = args.vessel.strip().lower()
        slips = {
            "aurora": {"slip": "B-12", "window": "03:40–05:10", "depth_m": 8.2},
            "cinder": {"slip": "C-3", "window": "01:15–02:00", "depth_m": 5.5},
        }
        return slips.get(key, {"slip": "HOLD-1", "window": "unscheduled", "depth_m": 0.0})

    @tool(registry=registry)
    def weigh_crate(args: CrateArgs) -> dict:
        """Return approved weight and hazard flags for a cargo crate."""
        weights = {
            "CR-77": {"kg": 420.0, "hazard": False},
            "CR-9": {"kg": 1180.0, "hazard": True},
        }
        hit = weights.get(args.crate_id.upper(), {"kg": 0.0, "hazard": True})
        return {"crate_id": args.crate_id.upper(), **hit}

    @tool(registry=registry)
    def tide_chart(args: TideArgs) -> dict:
        """Read channel tide and current guidance."""
        charts = {
            "north": {"window": "flood +40min", "current_kn": 1.2},
            "south": {"window": "slack in 12min", "current_kn": 0.3},
        }
        hit = charts.get(args.channel.strip().lower(), {"window": "unknown", "current_kn": 9.9})
        return {"channel": args.channel.strip().lower(), **hit}

    return registry


def _record_reasoning(
    ctx: NodeContext,
    *,
    text: str,
    tool_name: str,
    tool_result: dict[str, Any],
) -> NodeResult:
    """Publish plain-text notes + tool evidence for the extraction sibling."""
    return ctx.result(
        output=text,
        state_updates={
            "reasoning_text": text,
            "tool_evidence": [{"tool": tool_name, "result": tool_result}],
            "lane": ctx.node.group,
        },
    )


def _notes_and_evidence(ctx: NodeContext) -> tuple[str, list[dict[str, Any]]]:
    text = str(ctx.state.get("reasoning_text") or "").strip()
    evidence = list(ctx.state.get("tool_evidence") or [])
    if not text:
        for record in reversed(ctx.prior_executions):
            if isinstance(record.output, str) and record.output.strip():
                text = record.output.strip()
                break
    return text, evidence


def _evidence_tools(evidence: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for item in evidence:
        name = item.get("tool")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    return names


def _with_handlers(
    combine: CombineNode,
    *,
    reasoning: Callable[[NodeContext], NodeResult],
    schema: Callable[[NodeContext], NodeResult],
) -> CombineNode:
    """Attach local handlers to a combine expansion for offline demos/tests."""

    class _HandledCombine(CombineNode):
        def expand(self) -> tuple[list[Node], list]:
            nodes, edges = CombineNode.expand(self)
            reason_node, schema_node = nodes
            return (
                [
                    reason_node.model_copy(update={"handler": reasoning, "kind": "handler"}),
                    schema_node.model_copy(update={"handler": schema, "kind": "handler"}),
                ],
                edges,
            )

    return _HandledCombine(
        id=combine.id,
        input_schema=combine.input_schema,
        tools=combine.tools,
        output_schema=combine.output_schema,
        prompt=combine.prompt,
        name=combine.name,
        description=combine.description,
        prerequisites=combine.prerequisites,
        group=combine.group,
        metadata=combine.metadata,
        provider=combine.provider,
        schema_prompt=combine.schema_prompt,
    )


def build_harbor_graph() -> FSM:
    def berth_scout(ctx: NodeContext) -> NodeResult:
        vessel = str(ctx.state.get("vessel") or ctx.intent)
        slip = ctx.tools.invoke("lookup_slip", {"vessel": vessel})
        text = (
            f"Vessel {vessel!r} maps to slip {slip['slip']} "
            f"in window {slip['window']} (depth {slip['depth_m']}m)."
        )
        return _record_reasoning(
            ctx, text=text, tool_name="lookup_slip", tool_result=slip
        )

    def berth_clearance(ctx: NodeContext) -> NodeResult:
        notes, evidence = _notes_and_evidence(ctx)
        slip = "HOLD-1"
        if evidence and isinstance(evidence[0].get("result"), dict):
            slip = str(evidence[0]["result"].get("slip") or slip)
        status: Literal["cleared", "hold", "diverted"] = (
            "hold" if slip.startswith("HOLD") else "cleared"
        )
        payload = BerthClearance(
            guest_text=(
                f"Clearance {status} for slip {slip}. Desk notes: {notes}"
                if notes
                else f"Clearance {status} for slip {slip}."
            ),
            slip=slip,
            status=status,
            evidence_tools=_evidence_tools(evidence),
        )
        return ctx.result(output=payload.model_dump(), next_state="End")

    def cargo_inspect(ctx: NodeContext) -> NodeResult:
        crate_id = str(ctx.state.get("crate_id") or "CR-77")
        weighed = ctx.tools.invoke("weigh_crate", {"crate_id": crate_id})
        flag = "HAZARD" if weighed["hazard"] else "clean"
        text = (
            f"Crate {weighed['crate_id']} weighs {weighed['kg']}kg "
            f"and is flagged {flag}."
        )
        return _record_reasoning(
            ctx, text=text, tool_name="weigh_crate", tool_result=weighed
        )

    def cargo_manifest(ctx: NodeContext) -> NodeResult:
        notes, evidence = _notes_and_evidence(ctx)
        result = (evidence[0].get("result") if evidence else {}) or {}
        crate_id = str(result.get("crate_id") or ctx.state.get("crate_id") or "UNKNOWN")
        kg = float(result.get("kg") or 0.0)
        hazard = bool(result.get("hazard"))
        disposition: Literal["stow", "inspect", "quarantine"] = (
            "quarantine" if hazard else ("inspect" if kg > 1000 else "stow")
        )
        payload = CargoManifest(
            guest_text=(
                f"Manifest disposition={disposition} for {crate_id}. Notes: {notes}"
                if notes
                else f"Manifest disposition={disposition} for {crate_id}."
            ),
            crate_id=crate_id,
            disposition=disposition,
            kilograms=kg,
            evidence_tools=_evidence_tools(evidence),
        )
        return ctx.result(output=payload.model_dump(), next_state="End")

    def pilot_brief(ctx: NodeContext) -> NodeResult:
        channel = str(ctx.state.get("channel") or "north")
        chart = ctx.tools.invoke("tide_chart", {"channel": channel})
        text = (
            f"Channel {chart['channel']}: window {chart['window']}, "
            f"current {chart['current_kn']} kn."
        )
        return _record_reasoning(
            ctx, text=text, tool_name="tide_chart", tool_result=chart
        )

    def pilot_advisory(ctx: NodeContext) -> NodeResult:
        notes, evidence = _notes_and_evidence(ctx)
        result = (evidence[0].get("result") if evidence else {}) or {}
        channel = str(result.get("channel") or ctx.state.get("channel") or "unknown")
        window = str(result.get("window") or "unknown")
        current = float(result.get("current_kn") or 0.0)
        advice: Literal["proceed", "wait", "reroute"] = (
            "reroute" if current >= 5 else ("wait" if current >= 1.0 else "proceed")
        )
        payload = PilotAdvisory(
            guest_text=(
                f"Pilot advice={advice} on {channel} ({window}). Notes: {notes}"
                if notes
                else f"Pilot advice={advice} on {channel} ({window})."
            ),
            channel=channel,
            window=window,
            advice=advice,
            evidence_tools=_evidence_tools(evidence),
        )
        return ctx.result(output=payload.model_dump(), next_state="End")

    berth = _with_handlers(
        CombineNode(
            id=BERTH_SCOUT,
            group="berth",
            tools=("lookup_slip",),
            input_schema=OpenInput,
            output_schema=BerthClearance,
            prompt=(
                "Scout a berth for the vessel. Call lookup_slip when you need a slip "
                "assignment. Reply with plain working notes only — not JSON."
            ),
            schema_prompt=(
                "Turn berth scout notes and tool evidence into a clearance JSON "
                "ticket. No tools — extract schema only."
            ),
        ),
        reasoning=berth_scout,
        schema=berth_clearance,
    )
    cargo = _with_handlers(
        CombineNode(
            id=CARGO_INSPECT,
            group="cargo",
            tools=("weigh_crate",),
            input_schema=OpenInput,
            output_schema=CargoManifest,
            prompt=(
                "Inspect cargo. Call weigh_crate for mass and hazard flags. "
                "Reply with plain working notes only — not JSON."
            ),
            schema_prompt=(
                "Turn cargo inspection notes and scale evidence into a manifest "
                "JSON record. No tools — extract schema only."
            ),
        ),
        reasoning=cargo_inspect,
        schema=cargo_manifest,
    )
    pilot = _with_handlers(
        CombineNode(
            id=PILOT_BRIEF,
            group="pilot",
            tools=("tide_chart",),
            input_schema=OpenInput,
            output_schema=PilotAdvisory,
            prompt=(
                "Brief the pilot. Call tide_chart for the channel. "
                "Reply with plain working notes only — not JSON."
            ),
            schema_prompt=(
                "Turn pilot brief notes and tide evidence into an advisory JSON. "
                "No tools — extract schema only."
            ),
        ),
        reasoning=pilot_brief,
        schema=pilot_advisory,
    )

    out_of_scope = SchemaNode(
        id=FALLBACK,
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=OutOfScopeReply,
        prompt="Politely refuse non-harbor requests as structured JSON.",
    )
    # Offline demo: attach a handler to the fallback schema node.
    out_of_scope = out_of_scope.model_copy(
        update={
            "handler": lambda ctx: ctx.result(
                output=OutOfScopeReply(
                    guest_text=(
                        "Signal desk only handles berth, cargo, and pilot traffic."
                    )
                ).model_dump()
            ),
            "kind": "handler",
        }
    )

    return FSM(
        nodes=[berth, cargo, pilot, out_of_scope],
        edges=[
            edge_semantic("Start", "berth", target_kind="group"),
            edge_semantic("Start", "cargo", target_kind="group"),
            edge_semantic("Start", "pilot", target_kind="group"),
            # CombineNode auto-adds reasoning → schema edges.
            edge_deterministic(BERTH_CLEARANCE, "End"),
            edge_deterministic(CARGO_MANIFEST, "End"),
            edge_deterministic(PILOT_ADVISORY, "End"),
            edge_fallback("Start", FALLBACK),
        ],
        groups=[
            Group(name="berth", description="Slip assignment and clearance"),
            Group(name="cargo", description="Freight weigh-in and manifest"),
            Group(name="pilot", description="Channel brief and advisory"),
        ],
        input_schema=HARBOR_INPUT_SCHEMA,
    )


def harbor_manifest() -> dict[str, Any]:
    """Console-safe topology for the Harbor Signal Desk."""
    return graph_manifest(build_harbor_graph(), build_harbor_tools())
