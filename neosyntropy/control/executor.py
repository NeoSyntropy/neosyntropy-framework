"""Step execution: run nodes, collect proposals, never commit.

The executor runs one plan step at a time so the control manager can gate each
step's results before any state commits.

Prompt assembly is a backend concern. The framework only declares node
``prompt`` / tools / schema and forwards run context; the backend builds the
model-facing prompt.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from ..core.context import RunContext
from ..core.graph import FSM
from ..core.models import Candidate, NodeResult
from ..core.node import (
    REASONING_TEXT_KEY,
    TOOL_EVIDENCE_KEY,
    Node,
    NodeContext,
)
from ..providers.base import ProviderRegistry
from ..tools.communication.calling import ToolCallingLoop, expects_json_object
from ..tools.core.registry import BoundTools, ToolNotAllowedError, ToolRegistry


class TopologyExecutor:
    def __init__(
        self,
        providers: ProviderRegistry,
        tools: ToolRegistry | None = None,
        *,
        tool_loop: ToolCallingLoop | None = None,
    ):
        self.providers = providers
        self.tools = tools or ToolRegistry()
        self.tool_loop = tool_loop or ToolCallingLoop()

    async def execute_step(
        self,
        indices: list[int],
        candidates: list[Candidate],
        graph: FSM,
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
        output_schema = definition.output_schema or None
        bound_tools = BoundTools(
            registry=self.tools,
            allowed=definition.tools,
            node_id=definition.id,
        )
        node_context = NodeContext(
            run=context,
            node=definition,
            tools=bound_tools,
        )

        try:
            if definition.handler is not None and definition.kind == "handler":
                raw = definition.handler(node_context)
                if inspect.isawaitable(raw):
                    raw = await raw
            elif definition.mode == "reasoning" and definition.tools:
                loop = await self.tool_loop.run(
                    provider=self.providers.get(definition.provider),
                    messages=_seed_messages(definition, context),
                    tools=bound_tools,
                    output_schema=output_schema,
                    node=definition,
                    context=context,
                )
                text = loop.text or None
                raw = NodeResult(
                    node_id=definition.id,
                    output=(
                        _parse_structured(text)
                        if text is not None and expects_json_object(output_schema)
                        else text
                    ),
                    tool_calls=loop.records,
                )
            else:
                provider = self.providers.get(definition.provider)
                output = _provider_generate(
                    provider,
                    definition=definition,
                    context=context,
                    schema=output_schema,
                    tools=bound_tools,
                )
                if inspect.isawaitable(output):
                    output = await output
                
                from ..core.models import GenerateResult
                if isinstance(output, GenerateResult):
                    output_text = output.text
                else:
                    output_text = output

                if output_schema is not None and expects_json_object(output_schema):
                    output_value = _parse_structured(output_text)
                    
                    if definition.handler is not None and definition.kind == "schema":
                        # Validate the JSON into the pydantic model if available
                        if definition.output_model:
                            validated = definition.output_model(**output_value)
                            raw_out = definition.handler(validated)
                        else:
                            raw_out = definition.handler(output_value)
                            
                        if inspect.isawaitable(raw_out):
                            raw_out = await raw_out
                        output_value = raw_out
                else:
                    output_value = output_text
                    
                raw = NodeResult(node_id=definition.id, output=output_value)
        except ToolNotAllowedError:
            raise
        except Exception as exc:
            code = getattr(exc, "code", None)
            message = str(exc)
            if code == "inference_warming" or "still warming" in message.lower():
                error = message
            else:
                error = f"{type(exc).__name__}: {exc}"
            return NodeResult(
                node_id=definition.id,
                status="failed",
                error=error,
            )

        return self._normalize(definition, raw)

    @staticmethod
    def _normalize(definition: Node, raw: Any) -> NodeResult:
        if raw is None:
            result = NodeResult(node_id=definition.id)
        elif isinstance(raw, NodeResult):
            if raw.node_id != definition.id:
                raise ValueError(
                    f"node {definition.id!r} returned result for {raw.node_id!r}"
                )
            result = raw
        else:
            result = NodeResult(node_id=definition.id, output=raw)

        return _with_reasoning_handoff(definition, result)


def declared_prompt(definition: Node) -> str:
    """Return the node-authored prompt text (declaration only, no assembly)."""
    return definition.prompt or definition.description or definition.name


def _seed_messages(definition: Node, context: RunContext) -> list[dict[str, str]]:
    messages = [
        {"role": message.role, "content": message.content} for message in context.history
    ]
    # Run input is assembled into the first backend prompt via ``context``, but
    # local history drives extractors / later turns — keep the user request here.
    if context.input:
        messages.append(
            {
                "role": "user",
                "content": json.dumps(context.input, sort_keys=True, separators=(",", ":")),
            }
        )
    messages.append({"role": "user", "content": declared_prompt(definition)})
    return messages


def _provider_generate(
    provider: Any,
    *,
    definition: Node,
    context: RunContext,
    schema: dict[str, Any] | None,
    tools: BoundTools | None = None,
) -> Any:
    generate = provider.generate
    try:
        params = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: dict[str, Any] = {"schema": schema}
    if "node" in params:
        kwargs["node"] = definition
    if "context" in params:
        kwargs["context"] = context
    if "tools" in params and tools is not None:
        kwargs["tools"] = tools
    return generate(declared_prompt(definition), **kwargs)


def _parse_structured(output: Any) -> Any:
    if not isinstance(output, str):
        return output
    return json.loads(output)


def _with_reasoning_handoff(definition: Node, result: NodeResult) -> NodeResult:
    """Persist reasoning notes + tool evidence for a following schema node.

    Handlers may already set these keys; the framework only fills gaps so
    CombineNode / ReasoningNode → SchemaNode pairs share one convention.
    """
    if result.status != "succeeded" or definition.mode != "reasoning":
        return result

    updates = dict(result.state_updates)
    changed = False

    if REASONING_TEXT_KEY not in updates:
        text: str | None = None
        if isinstance(result.output, str) and result.output.strip():
            text = result.output
        if text is not None:
            updates[REASONING_TEXT_KEY] = text
            changed = True

    if TOOL_EVIDENCE_KEY not in updates and result.tool_calls:
        evidence = [
            {"tool": record.tool, "result": record.result}
            for record in result.tool_calls
            if record.ok and not record.denied
        ]
        if evidence:
            updates[TOOL_EVIDENCE_KEY] = evidence
            changed = True

    if not changed:
        return result
    return result.model_copy(update={"state_updates": updates})
