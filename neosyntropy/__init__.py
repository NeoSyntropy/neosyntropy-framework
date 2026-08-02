"""NeoSyntropy: a deterministic control layer for AI workflows.



Models can propose what should happen next; a finite-state graph defines what

is allowed to happen. Core primitives:



- :class:`Node` — executable capability (not a workflow position)

- :class:`Edge` — one permitted movement between states

- :class:`Group` — organization without a second control path

- :class:`ControlManager` — the pipeline that owns the whole cycle



Quickstart::



    from neosyntropy import ControlManager, Graph, edge_deterministic, edge_fallback, node

    from neosyntropy import EmptyOutput, OpenInput, TextOutput



    @node(id="VerifyIdentity", input_schema=OpenInput, output_schema=EmptyOutput)

    def verify_identity(ctx):

        return ctx.result(output={}, state_updates={"verified": True})



    @node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)

    def out_of_scope(ctx):

        return ctx.result(output={"message": "I can't help with that."})



    graph = Graph(

        nodes=[verify_identity, out_of_scope],

        edges=[

            edge_deterministic("Start", "VerifyIdentity"),

            edge_fallback("Start", "OutOfScope"),

        ],

    )

    result = ControlManager(graph).run({"intent": "refund order 123"})

"""

from __future__ import annotations



from .backend import (

    BackendClient,

    BackendError,

    BackendProvider,

)

from .control.executor import TopologyExecutor

from .control.logging import DecisionLogger, JsonlDecisionLogger

from .control.manager import ControlManager

from .control.validator import PlanValidationError, PlanValidator

from .core.context import ContextBuilder, RunContext

from .core.edge import (

    Edge,

    EdgeKind,

    EdgeTargetKind,

    TransitionTable,

    edge_deterministic,

    edge_fallback,

    edge_semantic,

)

from .core.graph import END, START, Graph, GraphValidationError

from .core.group import Group

from .core.models import (

    AuditRecord,

    Candidate,

    ExecutionRecord,

    ExecutionStepResult,

    GateCheck,

    Message,

    NodeResult,

    RoutingPlan,

    RunRequest,

    RunResult,

    ToolCall,

    ToolCallRecord,

    Topology,

)

from .core.node import Node, NodeContext, NodeMode, node

from .core.schemas import (
    EmptyInput,
    EmptyOutput,
    OpenInput,
    OpenOutput,
    TextOutput,
    input_model_schema,
    strict_model_schema,
)

from .core.state import StateConflictError, StateManager

from .observability import BackendTelemetryReporter, RunObserver, graph_manifest

from .providers.base import Provider, ProviderRegistry

from .providers.callable import CallableProvider

from .routing.base import Router, RouterError

from .routing.deterministic import DeterministicRouter

from .routing.semantic import SemanticRouter

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

    "END",

    "START",

    "TOOL_TRIGGER",

    "AuditRecord",

    "BackendClient",

    "BackendError",

    "BackendProvider",

    "BackendTelemetryReporter",

    "BoundTools",

    "CallableProvider",

    "Candidate",

    "ContextBuilder",

    "ControlManager",

    "DecisionLogger",

    "DeterministicRouter",

    "Edge",

    "EdgeKind",

    "EdgeTargetKind",

    "EmptyInput",
    "OpenInput",
    "EmptyOutput",

    "ExecutionRecord",

    "ExecutionStepResult",

    "ExtractionError",

    "GateCheck",

    "Graph",

    "GraphValidationError",

    "Group",

    "JsonlDecisionLogger",

    "Message",

    "Node",

    "NodeContext",

    "NodeMode",

    "NodeResult",

    "OpenOutput",

    "ParameterExtractor",

    "PlanValidationError",

    "PlanValidator",

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

    "SemanticRouter",

    "StateConflictError",

    "StateManager",

    "TextOutput",

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

    "edge_deterministic",

    "edge_fallback",

    "edge_semantic",

    "graph_manifest",

    "node",

    "parse_tool_trigger",

    "registered_tools",

    "input_model_schema",

    "strict_model_schema",

    "tool",

    "tool_json_schema",

]


