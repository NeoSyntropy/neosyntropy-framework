"""SLM router adapter preserving the trained wire contract exactly.

The instruction template, the 10-candidate list with reserved index 9
(``UNSUPPORTED_OR_OUT_OF_SCOPE_INTENT``), the ``### Instruction:`` /
``### Response:`` prompt wrap, and the constrained-decoding plan schema are
ported verbatim from ``train_slm_as_router``. Changing any of these breaks
trained router weights and datasets.

Hybrid plans stay ``topology: "sequential"`` with parallel inner steps on the
wire; this adapter maps that shape to ``Topology.HYBRID`` for the validator.
"""
from __future__ import annotations

import inspect
import json
from typing import Any

from ..core.context import RunContext
from ..core.models import Candidate, RoutingPlan, Topology
from ..providers.base import Provider
from .base import RouterError

PROMPT_STYLE = "### Instruction:\n{instruction}\n\n### Response:\n"
REJECTION_TEXT = "UNSUPPORTED_OR_OUT_OF_SCOPE_INTENT"
MAX_ACTIONABLE = 9
CANDIDATE_SLOTS = 10


def build_instruction(
    context: RunContext,
    candidate_names: list[str],
    *,
    category: str = "general",
) -> str:
    """Verbatim instruction assembly from the router training pipeline."""
    conversation_lines = [
        f"{message.role.title()}: {message.content}"
        for message in context.history
        if message.role in {"user", "assistant"} and message.content
    ]
    action_lines = [
        f"- {record.node_id} [{record.status}]: "
        f"{record.output if record.output not in (None, '') else 'No output provided'}"
        for record in context.prior_executions
    ]
    nodes_context = "\n".join(
        f"[{index}]: {name}" for index, name in enumerate(candidate_names)
    )
    newline = chr(10)
    return (
        f"Industry Category: [{category}]\n"
        f"Current FSM State: [{context.current_state}]\n"
        "Conversation History:\n"
        f"{newline.join(conversation_lines) if conversation_lines else '(none)'}\n"
        "Prior Graph Actions:\n"
        f"{newline.join(action_lines) if action_lines else '(none)'}\n"
        f'User Intent: "{context.intent}"\n'
        f"Available Candidates:\n{nodes_context}"
    )


def build_output_schema(candidate_indices: list[int]) -> dict[str, Any]:
    """Verbatim constrained-decoding schema from ``train/infrence.py``."""
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "minLength": 1},
            "topology": {
                "type": "string",
                "enum": ["parallel", "sequential", "fallback"],
            },
            "execution_plan": {
                "type": "array",
                "minItems": 1,
                "maxItems": len(candidate_indices),
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": len(candidate_indices),
                    "uniqueItems": True,
                    "items": {
                        "type": "integer",
                        "enum": candidate_indices,
                    },
                },
            },
        },
        "required": ["reasoning", "topology", "execution_plan"],
        "additionalProperties": False,
    }


class SlmRouter:
    """Routes via any :class:`Provider` speaking the trained router contract."""

    def __init__(self, provider: Provider, *, category: str = "general"):
        self.provider = provider
        self.category = category

    async def route(
        self, context: RunContext, candidates: list[Candidate]
    ) -> RoutingPlan:
        fallback_positions = [
            index for index, candidate in enumerate(candidates) if candidate.is_fallback
        ]
        if len(fallback_positions) != 1:
            raise RouterError("candidates must contain exactly one dedicated fallback")
        fallback_position = fallback_positions[0]

        actionable = [
            index
            for index, candidate in enumerate(candidates)
            if not candidate.is_fallback
        ][:MAX_ACTIONABLE]

        # Prompt slots 0..8 are actionable (padded with dummies), slot 9 is
        # always the rejection node — the trained contract.
        slot_to_candidate: dict[int, int] = {
            slot: candidate_index for slot, candidate_index in enumerate(actionable)
        }
        names = [candidates[index].name for index in actionable]
        while len(names) < MAX_ACTIONABLE:
            names.append(f"Dummy Contingency Node {len(names)}")
        names.append(REJECTION_TEXT)
        slot_to_candidate[MAX_ACTIONABLE] = fallback_position

        instruction = build_instruction(context, names, category=self.category)
        prompt = PROMPT_STYLE.format(instruction=instruction)
        schema = build_output_schema(list(range(CANDIDATE_SLOTS)))

        raw = self.provider.generate(prompt, schema=schema)
        if inspect.isawaitable(raw):
            raw = await raw
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RouterError(f"router returned invalid JSON: {raw!r}") from exc

        return self._to_plan(payload, slot_to_candidate)

    def _to_plan(
        self, payload: dict[str, Any], slot_to_candidate: dict[int, int]
    ) -> RoutingPlan:
        try:
            wire_topology = payload["topology"]
            wire_plan = payload["execution_plan"]
            reasoning = payload.get("reasoning", "")
        except (KeyError, TypeError) as exc:
            raise RouterError(f"router plan is missing fields: {payload!r}") from exc

        execution_plan: list[list[int]] = []
        for step in wire_plan:
            mapped_step: list[int] = []
            for slot in step:
                if slot not in slot_to_candidate:
                    raise RouterError(
                        f"router selected a padding candidate slot: {slot}"
                    )
                mapped_step.append(slot_to_candidate[slot])
            execution_plan.append(mapped_step)

        if wire_topology == "fallback":
            topology = Topology.FALLBACK
        elif wire_topology == "parallel":
            topology = Topology.PARALLEL
        elif wire_topology == "sequential":
            # Hybrid is encoded on the wire as sequential with parallel steps.
            has_parallel_step = any(len(step) > 1 for step in execution_plan)
            topology = Topology.HYBRID if has_parallel_step else Topology.SEQUENTIAL
        else:
            raise RouterError(f"unknown topology from router: {wire_topology!r}")

        try:
            return RoutingPlan(
                reasoning=reasoning, topology=topology, execution_plan=execution_plan
            )
        except ValueError as exc:
            raise RouterError(f"router plan failed validation: {exc}") from exc
