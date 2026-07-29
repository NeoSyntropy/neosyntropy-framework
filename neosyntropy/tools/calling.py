"""Model-driven tool calling for provider-backed nodes.

Reasoning and argument extraction are split, preserving the trained edge
contracts:

1. **Trigger** — the reasoner emits ``<TOOL:tool_name>`` with no arguments.
2. **Extract** — a parameter extractor fills a JSON object constrained by the
   tool's pydantic schema.
3. **Validate** — arguments are validated against the args model *before* the
   tool runs.
4. **Execute** — the call goes through the node's bound tools, so the
   allow-list is enforced fail-closed.
5. **Reinject** — the outcome is appended as a ``tool`` message and reasoning
   continues.

A model proposing an undeclared or unknown tool is a proposal, not
permission: the call is denied, never executed, and the refusal is reinjected
so the model can recover. Every attempt — denied, failed, or successful —
lands in :attr:`~neosyntropy.core.models.NodeResult.tool_calls`, so axioms
can gate on tool usage and the audit trail stays complete.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from ..core.models import ToolCall, ToolCallRecord
from ..providers.base import Provider
from .registry import BoundTools, ToolNotAllowedError, ToolRegistry

# The trained handoff token. Changing this breaks trained reasoner weights.
TOOL_TRIGGER = re.compile(r"<TOOL:([a-z][a-z0-9_]*)>")

# Verbatim edge-extractor SFT instruction line, so trained extractors see the
# prompt they were trained on.
EDGE_SFT_INSTRUCTION = "You extract tool arguments for a customer-support classifier."


class ExtractionError(RuntimeError):
    """The extractor could not produce schema-valid arguments."""


def parse_tool_trigger(text: str) -> tuple[str, str | None]:
    """Split generated text into visible prose and an optional tool name."""
    match = TOOL_TRIGGER.search(text)
    if match is None:
        return text.strip(), None
    return text[: match.start()].strip(), match.group(1)


def normalize_messages(messages: Sequence[dict[str, str]]) -> str:
    """``ROLE: content`` rendering used for extractor input (train/serve parity)."""
    return "\n".join(
        f"{message.get('role', 'unknown').upper()}: {message.get('content', '')}"
        for message in messages
    )


def knowledge_text(tool: str, conversation: str) -> str:
    """Soft-prefix knowledge string expected by GIST-style extractors."""
    return f"tool={tool}\n{conversation}"


def tool_json_schema(args_model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema suitable for constrained decoding of tool arguments."""
    schema = args_model.model_json_schema()
    schema.setdefault("type", "object")
    schema["additionalProperties"] = False
    if "properties" in schema and "required" not in schema:
        schema["required"] = list(schema["properties"])
    return schema


def build_extraction_prompt(
    tool: str,
    args_model: type[BaseModel],
    conversation: str,
    *,
    instruction: str = EDGE_SFT_INSTRUCTION,
) -> str:
    """Verbatim edge extraction prompt shape (only the lead line is tunable)."""
    fields = ", ".join(tool_json_schema(args_model).get("properties", {}))
    return (
        f"{instruction}\n"
        f"Tool: {tool}\n"
        f"Required JSON keys: {fields}\n"
        f"Conversation:\n{conversation}\n"
        "Respond with a single JSON object only.\n"
        "JSON:"
    )


@runtime_checkable
class ParameterExtractor(Protocol):
    async def extract(
        self, messages: Sequence[dict[str, str]], tool: str
    ) -> ToolCall: ...


async def _generate(provider: Provider, prompt: str, schema: dict | None = None) -> str:
    output = provider.generate(prompt, schema=schema)
    if inspect.isawaitable(output):
        output = await output
    return str(output)


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ProviderParameterExtractor:
    """Extracts tool arguments with any :class:`Provider`.

    The tool's JSON schema is passed to ``generate`` so providers that support
    constrained decoding emit a valid object by construction; the result is
    validated against the pydantic args model either way. Confidence is 1.0
    for schema-valid output — the gate that matters here is validation, not a
    score. Swap in a trained extractor (which reports real confidence) via the
    :class:`ParameterExtractor` protocol.
    """

    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        *,
        instruction: str = EDGE_SFT_INSTRUCTION,
    ):
        self.provider = provider
        self.registry = registry
        self.instruction = instruction

    async def extract(
        self, messages: Sequence[dict[str, str]], tool: str
    ) -> ToolCall:
        spec = self.registry.get(tool)
        prompt = build_extraction_prompt(
            tool,
            spec.args_model,
            normalize_messages(messages),
            instruction=self.instruction,
        )
        raw = await _generate(self.provider, prompt, tool_json_schema(spec.args_model))
        match = _JSON_OBJECT.search(raw)
        if match is None:
            raise ExtractionError(f"no JSON object in extractor output: {raw!r}")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"invalid JSON from extractor: {raw!r}") from exc
        try:
            validated = spec.args_model.model_validate(payload)
        except ValidationError as exc:
            raise ExtractionError(f"arguments failed {tool} schema: {exc}") from exc
        return ToolCall(tool=tool, arguments=validated.model_dump(), confidence=1.0)


@dataclass
class ToolLoopResult:
    text: str
    messages: list[dict[str, str]] = field(default_factory=list)
    records: list[ToolCallRecord] = field(default_factory=list)


class ToolCallingLoop:
    """Runs reason → trigger → extract → validate → execute → reinject.

    Bounded by construction: at most ``max_tool_calls`` executed calls and a
    hard turn limit, with duplicate calls rejected, so a model cannot loop
    forever or replay the same side effect.
    """

    def __init__(
        self,
        *,
        extractor: ParameterExtractor | None = None,
        confidence_threshold: float = 0.55,
        max_tool_calls: int = 4,
    ):
        self.extractor = extractor
        self.confidence_threshold = confidence_threshold
        self.max_tool_calls = max_tool_calls

    async def run(
        self,
        *,
        provider: Provider,
        messages: Sequence[dict[str, str]],
        tools: BoundTools,
    ) -> ToolLoopResult:
        extractor = self.extractor or ProviderParameterExtractor(
            provider, tools.registry
        )
        history = [dict(message) for message in messages]
        visible_parts: list[str] = []
        records: list[ToolCallRecord] = []
        seen: set[str] = set()
        executed = 0
        turns = 0

        while turns < self.max_tool_calls * 2 + 2:
            turns += 1
            raw = await _generate(provider, _reasoning_prompt(history))
            visible, tool = parse_tool_trigger(raw)
            if visible:
                visible_parts.append(visible)
                history.append({"role": "assistant", "content": visible})
            if tool is None:
                break

            trigger = f"<TOOL:{tool}>"
            if history and history[-1].get("role") == "assistant":
                history[-1]["content"] = f"{history[-1]['content']} {trigger}".strip()
            else:
                history.append({"role": "assistant", "content": trigger})

            # Allow-list gate: a proposal is not permission.
            if tool not in tools:
                records.append(ToolCallRecord(tool=tool, denied=True, error="not allowed"))
                history.append(
                    {
                        "role": "tool",
                        "content": (
                            f"Tool '{tool}' is not available here. "
                            f"Available tools: {list(tools.names()) or 'none'}."
                        ),
                    }
                )
                continue

            if executed >= self.max_tool_calls:
                history.append({"role": "tool", "content": "Tool-call limit reached."})
                break

            try:
                call = await extractor.extract(history, tool)
            except (ExtractionError, KeyError) as exc:
                records.append(ToolCallRecord(tool=tool, ok=False, error=str(exc)))
                history.append(
                    {"role": "tool", "content": f"Argument extraction failed: {exc}"}
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

            if call.confidence < self.confidence_threshold:
                records.append(
                    ToolCallRecord(
                        tool=call.tool,
                        arguments=call.arguments,
                        confidence=call.confidence,
                        ok=False,
                        error=f"Low-confidence arguments ({call.confidence:.3f})",
                    )
                )
                history.append(
                    {"role": "tool", "content": "Arguments were not confident enough."}
                )
                continue

            try:
                invocation = tools.try_invoke(call.tool, call.arguments)
            except ToolNotAllowedError:
                # The allow-list is authoritative even if a custom extractor
                # rewrote the tool name.
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
        else:
            history.append({"role": "tool", "content": "Turn limit reached."})

        return ToolLoopResult(
            text=" ".join(visible_parts).strip(),
            messages=history,
            records=records,
        )


def _reasoning_prompt(history: Sequence[dict[str, str]]) -> str:
    return normalize_messages(history) + "\nASSISTANT:"
