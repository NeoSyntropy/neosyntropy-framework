"""Step execution: run nodes, collect proposals, never commit.

Adapted from ``neosyntropy_backend_cli/core/selection/executor.py``. The
executor runs one plan step at a time so the control manager can gate each
step's results with axioms before any state commits.
"""
from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from ..core.axiom import AxiomViolation
from ..core.context import RunContext
from ..core.graph import Graph
from ..core.models import Candidate, NodeResult
from ..core.node import Node, NodeContext
from ..providers.base import ProviderRegistry
from ..tools.calling import ParameterExtractor, ToolCallingLoop
from ..tools.registry import BoundTools, ToolRegistry


class TopologyExecutor:
    def __init__(
        self,
        providers: ProviderRegistry,
        tools: ToolRegistry | None = None,
        *,
        extractor: ParameterExtractor | None = None,
        tool_loop: ToolCallingLoop | None = None,
    ):
        self.providers = providers
        self.tools = tools or ToolRegistry()
        self.tool_loop = tool_loop or ToolCallingLoop(extractor=extractor)

    async def execute_step(
        self,
        indices: list[int],
        candidates: list[Candidate],
        graph: Graph,
        context: RunContext,
    ) -> list[NodeResult]:
        results = list(
            await asyncio.gather(
                *(
                    self._run_node(graph.nodes[candidates[index].node_id], context)
                    for index in indices
                )
            )
        )
        # Single-node steps default to moving onto the node's own state; the
        # transition was already plan-validated. The fallback node is exempt:
        # it is a safe stop and never moves the workflow.
        if len(results) == 1 and results[0].next_state is None:
            definition = graph.nodes[results[0].node_id]
            if not definition.is_fallback:
                results[0] = results[0].model_copy(
                    update={"next_state": results[0].node_id}
                )
        return results

    async def _run_node(self, definition: Node, context: RunContext) -> NodeResult:
        node_context = NodeContext(
            run=context,
            node=definition,
            tools=BoundTools(
                registry=self.tools,
                allowed=definition.tools,
                node_id=definition.id,
            ),
        )
        try:
            if definition.handler is not None:
                raw = definition.handler(node_context)
                if inspect.isawaitable(raw):
                    raw = await raw
            elif definition.tools:
                # Provider-backed node with tools: reason, trigger, extract
                # arguments under schema, execute under the allow-list.
                loop = await self.tool_loop.run(
                    provider=self.providers.get(definition.provider),
                    messages=build_messages(definition, context, node_context.tools),
                    tools=node_context.tools,
                )
                raw = NodeResult(
                    node_id=definition.id,
                    output=loop.text or None,
                    tool_calls=loop.records,
                )
            else:
                provider = self.providers.get(definition.provider)
                output = provider.generate(render_prompt(definition, context))
                if inspect.isawaitable(output):
                    output = await output
                raw = NodeResult(node_id=definition.id, output=output)
        except AxiomViolation:
            # A broken axiom is a rejection, not a node failure.
            raise
        except Exception as exc:
            return NodeResult(
                node_id=definition.id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        return self._normalize(definition, raw)

    @staticmethod
    def _normalize(definition: Node, raw: Any) -> NodeResult:
        if raw is None:
            return NodeResult(node_id=definition.id)
        if isinstance(raw, NodeResult):
            if raw.node_id != definition.id:
                raise ValueError(
                    f"node {definition.id!r} returned result for {raw.node_id!r}"
                )
            return raw
        return NodeResult(node_id=definition.id, output=raw)


def render_prompt(
    definition: Node, context: RunContext, tools: BoundTools | None = None
) -> str:
    """Assemble the scoped prompt a provider-backed node receives.

    Small by construction: the node's own instructions, the request context,
    and only this node's tools — never a mega-prompt of every rule and tool.
    """
    instructions = definition.prompt or definition.description or definition.name
    sections = [
        f"{instructions}\n",
        f'User Intent: "{context.intent}"',
        f"Current State: {context.current_state}",
        f"State: {json.dumps(context.state, sort_keys=True, default=str)}",
    ]
    if tools is not None and tools.names():
        catalog = "\n".join(
            f"- {spec.name}: {spec.description}" for spec in tools.specs()
        )
        sections.append(
            "\nAvailable tools:\n"
            f"{catalog}\n"
            "To use one, end your reply with its trigger token and no arguments: "
            "<TOOL:tool_name>"
        )
    return "\n".join(sections)


def build_messages(
    definition: Node, context: RunContext, tools: BoundTools | None = None
) -> list[dict[str, str]]:
    """Seed the tool-calling conversation: prior history plus this node's task."""
    messages = [
        {"role": message.role, "content": message.content} for message in context.history
    ]
    messages.append({"role": "user", "content": render_prompt(definition, context, tools)})
    return messages
