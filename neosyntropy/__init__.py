"""NeoSyntropy: a deterministic control layer for AI workflows.



Models can propose what should happen next; a finite-state graph defines what

is allowed to happen. Core primitives:



- :class:`Node` — executable capability (not a workflow position)

- :class:`Edge` — one permitted movement between states

- :class:`Group` — named node collection; optional entry, routers, and edges

- :class:`ControlManager` — the pipeline that owns the whole cycle



Quickstart::



    from neosyntropy import Client, EmptyOutput, OpenInput, TextOutput, Workflow, node



    @node(id="VerifyIdentity", input_schema=OpenInput, output_schema=EmptyOutput)

    def verify_identity(ctx):

        return ctx.result(output={}, state_updates={"verified": True})



    @node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)

    def out_of_scope(ctx):

        return ctx.result(output={"message": "I can't help with that."})



    fsm = Workflow([verify_identity], fallback=out_of_scope)

    client = Client(api_key="...")
    client.create_project(name="Support Bot", slug="support-bot")
    # Or pass project_id=... explicitly if you already have one.

    result = fsm.run({"text": "refund order 123"}, client=client)

"""

from __future__ import annotations



from .backend import (

    DEFAULT_API_URL,

    BackendClient,

    BackendError,

    BackendProvider,

    Client,

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

from .core.graph import (
    END,
    FSM,
    FSMValidationError,
    Graph,
    GraphValidationError,
    Workflow,
)

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

from .core.decorators import function_calling, workflow
from .core.node import (
    COMBINE_SCHEMA_SUFFIX,
    REASONING_OUTPUT_SCHEMA,
    REASONING_TEXT_KEY,
    TOOL_EVIDENCE_KEY,
    CombineNode,
    Node,
    NodeContext,
    NodeKind,
    NodeMode,
    ReasoningNode,
    ReasoningStep,
    SchemaStep,
    SchemaNode,
    node,
    retrieval_node,
)

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

from .monitor.base import RunObserver
from .monitor.run.observer import BackendTelemetryReporter
from .monitor.graph.manifest import (
    control_graph_manifest,
    graph_manifest,
)

from .providers.base import Provider, ProviderRegistry

from .providers.callable import CallableProvider

from .backend import BackendSemanticRouter

from .core.routing.base import Router, RouterError

from .core.routing.deterministic import DeterministicRouter
from .core.routing.semantic import SemanticRouter

from .core.routing.preferred import PreferredPathRouter

from .tools.communication.calling import (

    ToolCallingLoop,



    ToolCallingLoop,
    ToolLoopResult,
)

from .tools.coding.ast_tools import (

    AnalyzeFileArgs,

    AstAnalyzer,

    AstTools,

    AstToolsError,

    FindBareExceptsArgs,

)

from .tools.coding.coding_tools import (

    CodingTools,

    CodingToolsError,

    CodingWorkspace,

    DEFAULT_ALLOWED_COMMANDS,

    EditFileArgs,

    FindArgs,

    GrepArgs,

    LsArgs,

    ReadFileArgs,

    RunShellArgs,

    WriteFileArgs,

)



from .tools.core.registry import (

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

    "ToolCallingLoop",

    "AuditRecord",

    "BackendClient",

    "BackendError",

    "BackendProvider",

    "BackendTelemetryReporter",

    "Client",

    "DEFAULT_API_URL",

    "BoundTools",

    "AnalyzeFileArgs",

    "AstAnalyzer",

    "AstTools",

    "AstToolsError",

    "FindBareExceptsArgs",

    "CodingTools",

    "CodingToolsError",

    "CodingWorkspace",

    "DEFAULT_ALLOWED_COMMANDS",

    "EditFileArgs",

    "FindArgs",

    "GrepArgs",

    "LsArgs",

    "ReadFileArgs",

    "RunShellArgs",

    "WriteFileArgs",

    "CallableProvider",

    "COMBINE_SCHEMA_SUFFIX",

    "Candidate",

    "CombineNode",

    "ContextBuilder",

    "ControlManager",

    "DecisionLogger",

    "BackendSemanticRouter",

    "DeterministicRouter",

    "PreferredPathRouter",

    "Edge",

    "EdgeKind",

    "EdgeTargetKind",

    "EmptyInput",
    "OpenInput",
    "EmptyOutput",

    "ExecutionRecord",

    "ExecutionStepResult",



    "GateCheck",

    "FSM",

    "FSMValidationError",

    "Graph",

    "GraphValidationError",

    "Group",

    "Workflow",

    "JsonlDecisionLogger",

    "Message",

    "Node",

    "NodeContext",

    "NodeKind",

    "NodeMode",

    "NodeResult",

    "OpenOutput",

    "REASONING_OUTPUT_SCHEMA",

    "REASONING_TEXT_KEY",

    "ReasoningNode",

    "ReasoningStep",

    "SchemaStep",

    "SchemaNode",

    "TOOL_EVIDENCE_KEY",

    "PlanValidationError",

    "PlanValidator",

    "Provider",

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

    "control_graph_manifest",
    "graph_manifest",

    "node",
    "retrieval_node",
    "function_calling",
    "workflow",
    "registered_tools",
    "input_model_schema",
    "strict_model_schema",
    "tool",


]
