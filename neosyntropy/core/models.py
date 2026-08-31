"""Strict typed models shared across the framework.

The request/plan/result contracts are ported from the proven
``neosyntropy_backend_cli`` runtime and must stay strict (``extra="forbid"``)
so schema drift is rejected instead of silently accepted.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Topology(str, Enum):
    """How planned steps may group nodes.

    ``HYBRID`` exists for validator compatibility; the trained semantic router
    encodes hybrid plans as ``sequential`` with parallel inner steps (stable
    wire format), and the backend maps that shape to ``HYBRID``.
    """

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


class Message(RuntimeModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionRecord(RuntimeModel):
    """Evidence of a previously executed node (input for prerequisites)."""

    node_id: str
    status: Literal["succeeded", "failed", "fallback"]
    output: Any = None
    state_updates: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class RunRequest(RuntimeModel):
    """One control-cycle request. Input is evidence, not authority."""

    input: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    current_state: str = ""
    history: list[Message] = Field(default_factory=list)
    prior_executions: list[ExecutionRecord] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Candidate(RuntimeModel):
    """A request-local, selectable view of a node presented to the router."""

    node_id: str
    name: str
    description: str = ""
    score: float = 0.0
    prerequisites: tuple[str, ...] = ()
    is_fallback: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class RoutingPlan(RuntimeModel):
    """Router proposal. Proposal is not permission."""

    reasoning: str = ""
    topology: Topology
    execution_plan: list[list[StrictInt]]

    @model_validator(mode="after")
    def plan_is_nonempty(self) -> RoutingPlan:
        if not self.execution_plan or any(not step for step in self.execution_plan):
            raise ValueError("execution_plan must contain non-empty steps")
        return self


class ToolCall(RuntimeModel):
    """A parameter extractor's output: typed arguments for one tool.

    This is the stable extractor contract — trained edge extractors return
    exactly this shape (tool name, arguments, confidence).
    """

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ToolCallRecord(RuntimeModel):
    """Audit record for one attempted tool call inside a node.

    ``denied`` marks a tool the model proposed but was not allowed to use;
    denied calls never execute.
    """

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ok: bool = False
    denied: bool = False
    result: Any = None
    error: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)


class GenerateResult(RuntimeModel):
    """The outcome of a single LLM generation that may include tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class NodeResult(RuntimeModel):
    """What a node proposes. State changes commit only after gates pass."""

    node_id: str
    status: Literal["succeeded", "failed", "fallback"] = "succeeded"
    output: Any = None
    state_updates: dict[str, Any] = Field(default_factory=dict)
    next_state: str | None = None
    error: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class ExecutionStepResult(RuntimeModel):
    step: int = Field(ge=0)
    results: list[NodeResult]


class GateCheck(RuntimeModel):
    """Audit record of one gate evaluation (validator, guard, or built-in)."""

    name: str
    stage: Literal["plan", "result"]
    passed: bool
    message: str = ""
    node_id: str | None = None


class AuditRecord(RuntimeModel):
    """Every control cycle emits one; reviews check a graph path, not a transcript."""

    request_id: str
    input: dict[str, Any] = Field(default_factory=dict)
    initial_state: str
    final_state: str
    plan: RoutingPlan | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    gate_checks: list[GateCheck] = Field(default_factory=list)
    steps: list[ExecutionStepResult] = Field(default_factory=list)
    committed_transitions: list[str] = Field(default_factory=list)
    rejected: bool = False
    rejection: str | None = None
    created_at: float = Field(default_factory=time.time)


class RunResult(RuntimeModel):
    """Outcome of one control cycle.

    A rejection (illegal plan, illegal transition, failed guard) is a normal,
    non-exceptional outcome: ``rejected=True``, no state change was committed
    for the offending step, and the audit record explains why.

    When the backend owns control, ``plan`` and ``candidates`` are omitted so
    routing internals never leak to the client.
    """

    request_id: str
    plan: RoutingPlan | None = None
    candidates: list[Candidate] = Field(default_factory=list, max_length=10)
    steps: list[ExecutionStepResult]
    final_state: str
    state: dict[str, Any] = Field(default_factory=dict)
    completed: bool
    rejected: bool = False
    rejection: str | None = None
    audit: AuditRecord
