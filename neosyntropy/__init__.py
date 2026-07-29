"""NeoSyntropy: a deterministic control layer for AI workflows.

Models can propose what should happen next; a finite-state graph defines what
is allowed to happen. Five primitives span the problem space:

- :class:`Node` — executable capability (not a workflow position)
- :class:`Edge` — one permitted movement between states
- :class:`Axiom` — an invariant enforced fail-closed before anything commits
- :class:`Group` — organization without a second control path
- :class:`ControlManager` — the pipeline that owns the whole cycle

Quickstart::

    from neosyntropy import ControlManager, Edge, Graph, axiom, node

    @node(id="VerifyIdentity")
    def verify_identity(ctx):
        return ctx.result(state_updates={"verified": True})

    @node(id="OutOfScope", is_fallback=True)
    def out_of_scope(ctx):
        return ctx.result(output="I can't help with that.")

    @axiom(name="MustBeVerified", nodes=("IssueRefund",))
    def must_be_verified(ctx, proposal):
        return proposal.state.get("verified", False)

    graph = Graph(
        nodes=[verify_identity, out_of_scope],
        edges=[Edge(source="Start", target="VerifyIdentity", label="first")],
        axioms=[must_be_verified],
    )
    result = ControlManager(graph).run({"intent": "refund order 123"})
"""
from __future__ import annotations

from .backend import (
    BackendCandidateSelector,
    BackendClient,
    BackendError,
    BackendProvider,
    BackendRouter,
)
from .control.executor import TopologyExecutor
from .control.logging import DecisionLogger, JsonlDecisionLogger
from .control.manager import ControlManager
from .control.selector import CandidateSelector, LexicalCandidateSelector
from .control.validator import PlanValidationError, PlanValidator
from .core.axiom import (
    Axiom,
    AxiomEngine,
    AxiomViolation,
    OutputAxiom,
    Proposal,
    axiom,
)
from .core.context import ContextBuilder, RunContext
from .core.edge import EDGE_LABEL_PRIORITY, Edge, TransitionTable
from .core.graph import END, START, Graph, GraphValidationError
from .core.group import Group
from .core.models import (
    AuditRecord,
    AxiomCheck,
    Candidate,
    ExecutionRecord,
    ExecutionStepResult,
    Message,
    NodeResult,
    RoutingPlan,
    RunRequest,
    RunResult,
    ToolCall,
    ToolCallRecord,
    Topology,
)
from .core.node import Node, NodeContext, node
from .core.state import StateConflictError, StateManager
from .observability import BackendTelemetryReporter, RunObserver, graph_manifest
from .providers.base import DeterministicProvider, Provider, ProviderRegistry
from .providers.callable import CallableProvider
from .routing.base import Router, RouterError
from .routing.deterministic import DeterministicRouter
from .routing.slm import SlmRouter
from .tools.calling import (
    TOOL_TRIGGER,
    ExtractionError,
    ParameterExtractor,
    ProviderParameterExtractor,
    ToolCallingLoop,
    ToolLoopResult,
    parse_tool_trigger,
    tool_json_schema,
)
from .tools.registry import (
    DEFAULT_REGISTRY,
    BoundTools,
    RegisteredTool,
    ToolInvocation,
    ToolNotAllowedError,
    ToolRegistry,
    registered_tools,
    tool,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_REGISTRY",
    "EDGE_LABEL_PRIORITY",
    "END",
    "START",
    "TOOL_TRIGGER",
    "AuditRecord",
    "Axiom",
    "AxiomCheck",
    "AxiomEngine",
    "AxiomViolation",
    "BackendCandidateSelector",
    "BackendClient",
    "BackendError",
    "BackendProvider",
    "BackendRouter",
    "BackendTelemetryReporter",
    "BoundTools",
    "CallableProvider",
    "Candidate",
    "CandidateSelector",
    "ContextBuilder",
    "ControlManager",
    "DecisionLogger",
    "DeterministicProvider",
    "DeterministicRouter",
    "Edge",
    "ExecutionRecord",
    "ExecutionStepResult",
    "ExtractionError",
    "Graph",
    "GraphValidationError",
    "Group",
    "JsonlDecisionLogger",
    "LexicalCandidateSelector",
    "Message",
    "Node",
    "NodeContext",
    "NodeResult",
    "OutputAxiom",
    "ParameterExtractor",
    "PlanValidationError",
    "PlanValidator",
    "Proposal",
    "Provider",
    "ProviderParameterExtractor",
    "ProviderRegistry",
    "RegisteredTool",
    "Router",
    "RouterError",
    "RoutingPlan",
    "RunContext",
    "RunObserver",
    "RunRequest",
    "RunResult",
    "SlmRouter",
    "StateConflictError",
    "StateManager",
    "ToolCall",
    "ToolCallRecord",
    "ToolCallingLoop",
    "ToolInvocation",
    "ToolLoopResult",
    "ToolNotAllowedError",
    "ToolRegistry",
    "Topology",
    "TopologyExecutor",
    "TransitionTable",
    "axiom",
    "graph_manifest",
    "node",
    "parse_tool_trigger",
    "registered_tools",
    "tool",
    "tool_json_schema",
]
