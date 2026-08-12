"""Model-driven tool calling for provider-backed nodes.

Uses native foundation model tool calls.
1. **Request** — The provider receives the tool list (JSON Schema) and prompt.
2. **Execute** — The call goes through the node's bound tools, so the
   allow-list is enforced fail-closed.
3. **Reinject** — The outcome is appended as a ``tool`` message and reasoning
   continues.

A model proposing an undeclared or unknown tool is a proposal, not
permission: the call is denied, never executed, and the refusal is reinjected
so the model can recover. Every attempt — denied, failed, or successful —
lands in :attr:`~neosyntropy.core.models.NodeResult.tool_calls` so the
audit trail stays complete.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..core.models import ToolCallRecord, GenerateResult
from ..providers.base import Provider
from .registry import BoundTools, ToolNotAllowedError


def normalize_messages(messages: Sequence[dict[str, str]]) -> str:
    """``ROLE: content`` rendering used for legacy reasoner fallback."""
    return "\n".join(
        f"{message.get('role', 'unknown').upper()}: {message.get('content', '')}"
        for message in messages
    )


async def _generate(
    provider: Provider,
    prompt: str,
    schema: dict | None = None,
    tools: BoundTools | None = None,
    **extra: Any,
) -> GenerateResult:
    kwargs: dict[str, Any] = {"schema": schema}
    try:
        params = inspect.signature(provider.generate).parameters
    except (TypeError, ValueError):
        params = {}
    if "tools" in params and tools is not None:
        kwargs["tools"] = tools
    for key, value in extra.items():
        if key in params and value is not None:
            kwargs[key] = value
    output = provider.generate(prompt, **kwargs)
    if inspect.isawaitable(output):
        output = await output
    if isinstance(output, GenerateResult):
        return output
    return GenerateResult(text=str(output), tool_calls=[])


@dataclass
class ToolLoopResult:
    text: str
    messages: list[dict[str, str]] = field(default_factory=list)
    records: list[ToolCallRecord] = field(default_factory=list)


class ToolCallingLoop:
    """Runs reason → native trigger → validate → execute → reinject.

    Bounded by construction: at most ``max_tool_calls`` executed calls and a
    hard turn limit, with duplicate calls rejected, so a model cannot loop
    forever or replay the same side effect.
    """

    def __init__(
        self,
        *,
        max_tool_calls: int = 4,
        extractor: Any = None,
        confidence_threshold: float = 0.55,
    ):
        self.max_tool_calls = max_tool_calls

    async def run(
        self,
        *,
        provider: Provider,
        messages: Sequence[dict[str, str]],
        tools: BoundTools,
        output_schema: dict[str, Any] | None = None,
        node: Any = None,
        context: Any = None,
    ) -> ToolLoopResult:
        history = [dict(message) for message in messages]
        visible_parts: list[str] = []
        records: list[ToolCallRecord] = []
        seen: set[str] = set()
        executed = 0
        turns = 0
        provider_kwargs: dict[str, Any] = {"node": node} if node is not None else {}

        while turns < self.max_tool_calls * 2 + 2:
            turns += 1
            if turns == 1 and node is not None and context is not None:
                result = await _generate(
                    provider,
                    _reasoning_prompt(history),
                    node=node,
                    context=context,
                    tools=tools,
                )
            else:
                result = await _generate(
                    provider, _reasoning_prompt(history), tools=tools, **provider_kwargs
                )
            
            if result.text.strip():
                visible_parts.append(result.text.strip())
                history.append({"role": "assistant", "content": result.text.strip()})

            if not result.tool_calls:
                break

            for call in result.tool_calls:
                if executed >= self.max_tool_calls:
                    history.append({"role": "tool", "content": "Tool-call limit reached."})
                    break

                if call.tool not in tools:
                    records.append(ToolCallRecord(tool=call.tool, denied=True, error="not allowed"))
                    history.append(
                        {
                            "role": "tool",
                            "content": (
                                f"Tool '{call.tool}' is not available here. "
                                f"Available tools: {list(tools.names()) or 'none'}."
                            ),
                        }
                    )
                    continue

                fingerprint = hashlib.sha256(
                    json.dumps(
                        {"tool": call.tool, "arguments": call.arguments}, sort_keys=True
                    ).encode()
                ).hexdigest()
                
                if fingerprint in seen:
                    history.append(
                        {"role": "tool", "content": "Duplicate tool call rejected."}
                    )
                    continue
                seen.add(fingerprint)

                try:
                    invocation = tools.try_invoke(call.tool, call.arguments)
                except ToolNotAllowedError:
                    records.append(
                        ToolCallRecord(
                            tool=call.tool,
                            arguments=call.arguments,
                            denied=True,
                            error="not allowed",
                        )
                    )
                    history.append(
                        {"role": "tool", "content": f"Tool '{call.tool}' is not available."}
                    )
                    continue
                except Exception as exc:
                    records.append(ToolCallRecord(tool=call.tool, ok=False, error=str(exc)))
                    history.append(
                        {"role": "tool", "content": f"Argument extraction failed: {exc}"}
                    )
                    continue

                executed += 1
                records.append(
                    ToolCallRecord(
                        tool=call.tool,
                        arguments=call.arguments,
                        confidence=call.confidence,
                        ok=invocation.ok,
                        result=invocation.result,
                        error=invocation.error,
                        latency_ms=invocation.latency_ms,
                    )
                )
                history.append(
                    {
                        "role": "tool",
                        "content": json.dumps(
                            {
                                "tool": invocation.tool,
                                "ok": invocation.ok,
                                "result": invocation.result,
                                "error": invocation.error,
                            },
                            separators=(",", ":"),
                            default=str,
                        ),
                    }
                )
                
            if executed >= self.max_tool_calls:
                break
        else:
            history.append({"role": "tool", "content": "Turn limit reached."})

        if expects_json_object(output_schema):
            structured = await _generate(
                provider,
                _structured_output_prompt(history),
                schema=output_schema,
                **provider_kwargs,
            )
            visible_parts = [structured.text]
            history.append({"role": "assistant", "content": structured.text})
        elif not visible_parts:
            closing = await _generate(
                provider,
                _reasoning_prompt(history),
                **provider_kwargs,
            )
            if closing.text.strip():
                visible_parts.append(closing.text.strip())
                history.append({"role": "assistant", "content": closing.text.strip()})

        return ToolLoopResult(
            text="\n\n".join(part for part in visible_parts if part).strip(),
            messages=history,
            records=records,
        )


def expects_json_object(schema: dict[str, Any] | None) -> bool:
    """True when the output contract is structured JSON (not plain text)."""
    if not schema:
        return False
    schema_type = schema.get("type")
    if schema_type == "string":
        return False
    if schema_type == "object" or "properties" in schema:
        return True
    return schema_type is None


def _reasoning_prompt(history: Sequence[dict[str, str]]) -> str:
    return normalize_messages(history) + "\nASSISTANT:"


def _structured_output_prompt(history: Sequence[dict[str, str]]) -> str:
    return (
        normalize_messages(history)
        + "\nReturn the final answer as one JSON object matching the supplied schema."
        "\nJSON:"
    )
