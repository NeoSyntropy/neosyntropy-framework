"""Shared helpers for cookbook copies under ``tests/scenarios``.

Each cookbook example is registered here, wrapped as a scenario, and run
offline (no live backend). ``BACKEND_APIS`` on the spec is the list of
backend endpoints that the same cookbook should exercise when it is run
against a real API with a ``Client``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from neosyntropy.core.models import RoutingPlan
from tests.scenarios.stores import SqliteDatabase

ROOT = Path(__file__).resolve().parents[2]
COOKBOOK_ROOT = ROOT / "cookbook"
PROVIDER = "offline/cookbook"


# ---------------------------------------------------------------------------
# Backend API catalog (what to validate when a cookbook runs against the API)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendApi:
    method: str
    path: str
    when: str
    purpose: str


def _api(method: str, path: str, when: str, purpose: str) -> BackendApi:
    return BackendApi(method=method, path=path, when=when, purpose=purpose)


PROJECT_APIS = (
    _api(
        "GET",
        "/observability/projects",
        "cookbook_main",
        "ensure_project lists projects owned by the API key",
    ),
    _api(
        "POST",
        "/observability/projects",
        "cookbook_main",
        "create the cookbook project when the slug is new (409 reuse on retry)",
    ),
)

INFERENCE_APIS = (
    _api(
        "POST",
        "/framework/inference",
        "cookbook_main",
        "SchemaNode / ReasoningNode / LLM validators generate through BackendProvider",
    ),
)

FUNCTION_CONCEPT_APIS = (
    _api(
        "POST",
        "/observability/projects/{project_id}/functions",
        "cookbook_main",
        "register @function_calling / @workflow concept manifests",
    ),
)

MONITOR_APIS = (
    _api(
        "POST",
        "/observability/projects/{project_id}/graphs",
        "monitor",
        "register the source-free graph structure for the console",
    ),
    _api(
        "POST",
        "/telemetry/runs",
        "monitor",
        "open an observability run when NEOSYNTROPY_MONITOR=TRUE",
    ),
    _api(
        "POST",
        "/telemetry/runs/{run_id}/events",
        "monitor",
        "step_started / plan_proposed / step_completed (and tools_executed)",
    ),
    _api(
        "POST",
        "/telemetry/runs/{run_id}/finish",
        "monitor",
        "close the observability run with final_state",
    ),
)

REMOTE_GRAPH_APIS = (
    _api(
        "POST",
        "/observability/projects/{project_id}/code-artifacts/missing",
        "remote_execution",
        "ask which handler/tool bundles the project is missing",
    ),
    _api(
        "PUT",
        "/observability/projects/{project_id}/code-artifacts/{sha256}",
        "remote_execution",
        "upload gzip code bundles for Python handlers and tools",
    ),
    _api(
        "PUT",
        "/observability/projects/{project_id}/graphs/{graph_id}/recovery",
        "remote_execution",
        "publish the executable recovery revision",
    ),
    _api(
        "GET",
        "/observability/projects/{project_id}/graphs/{graph_id}",
        "remote_execution",
        "fetch the stored graph after publish",
    ),
    _api(
        "GET",
        "/observability/projects/{project_id}/graphs/{graph_id}/artifacts",
        "remote_execution",
        "list code artifacts associated with the graph",
    ),
    _api(
        "GET",
        "/observability/projects/{project_id}/graphs/{graph_id}/artifacts/{artifact_id}/download",
        "remote_execution",
        "download bundle bytes for the local snapshot",
    ),
)

CONTROL_APIS = (
    _api(
        "POST",
        "/control/runs",
        "remote_execution",
        "start a backend-owned control run (graph + request)",
    ),
    _api(
        "POST",
        "/control/runs/{run_id}/results",
        "remote_execution",
        "submit node results (or client_rejection) for the next step",
    ),
)

ROUTER_APIS = (
    _api(
        "POST",
        "/framework/router",
        "remote_execution",
        "legacy semantic routing; prefer routing inside POST /control/runs",
    ),
)

LOCAL_ONLY_APIS = (
    _api(
        "NONE",
        "local-only",
        "cookbook_main",
        "this cookbook does not call the backend; validate the local toolkit only",
    ),
)


def _llm_fsm_apis(*, tools: bool = False, semantic: bool = False) -> tuple[BackendApi, ...]:
    apis = list(PROJECT_APIS + INFERENCE_APIS + MONITOR_APIS + REMOTE_GRAPH_APIS + CONTROL_APIS)
    if tools:
        apis.append(
            _api(
                "POST",
                "/telemetry/runs/{run_id}/events",
                "monitor",
                "tools_executed events must match in-process tool_calls",
            )
        )
    if semantic:
        apis.extend(ROUTER_APIS)
    return tuple(apis)


def _python_fsm_apis() -> tuple[BackendApi, ...]:
    return PROJECT_APIS + MONITOR_APIS + REMOTE_GRAPH_APIS + CONTROL_APIS


def _decorator_apis() -> tuple[BackendApi, ...]:
    return (
        PROJECT_APIS
        + FUNCTION_CONCEPT_APIS
        + INFERENCE_APIS
        + MONITOR_APIS
        + REMOTE_GRAPH_APIS
        + CONTROL_APIS
    )


# ---------------------------------------------------------------------------
# Cookbook specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CookbookSpec:
    id: str
    title: str
    cookbook: str
    kind: str
    mermaid: str
    backend_apis: tuple[BackendApi, ...]
    expected_nodes: tuple[str, ...] = ()
    payload_model: str = ""
    payload_kwargs: dict[str, Any] = field(default_factory=dict)
    graph_attr: str = "build_fsm"
    use_first_candidate_router: bool = False
    require_end: bool = True


SPECS: dict[str, CookbookSpec] = {}


def _spec(spec: CookbookSpec) -> CookbookSpec:
    SPECS[spec.id] = spec
    return spec


_spec(
    CookbookSpec(
        id="cookbook_python_node",
        title="Cookbook: Python node",
        cookbook="fsm/python_node_example.py",
        kind="fsm",
        mermaid="ValidateOrder --> FormatResponse --> End",
        backend_apis=_python_fsm_apis(),
        expected_nodes=("ValidateOrder", "FormatResponse"),
        payload_model="OrderInput",
        payload_kwargs={"order_id": "ORD-001", "customer": "Alice", "amount": 149.99},
    )
)
_spec(
    CookbookSpec(
        id="cookbook_schema_node",
        title="Cookbook: schema node",
        cookbook="fsm/schema_node_example.py",
        kind="fsm",
        mermaid="ExtractTicket --> End",
        backend_apis=_llm_fsm_apis(),
        expected_nodes=("ExtractTicket",),
        payload_model="TicketInput",
        payload_kwargs={
            "text": "Hi, I'm María Garcia. Please update my billing email to maria@example.com."
        },
    )
)
_spec(
    CookbookSpec(
        id="cookbook_reasoning_prompt_tools",
        title="Cookbook: reasoning node with prompt tools",
        cookbook="fsm/reasoning_node_prompt_tools_example.py",
        kind="fsm",
        mermaid="RouteIntent --> End",
        backend_apis=_llm_fsm_apis(tools=True),
        expected_nodes=("RouteIntent",),
        payload_model="SupportRequest",
        payload_kwargs={"intent": "My package is late and I need help with the shipment."},
    )
)
_spec(
    CookbookSpec(
        id="cookbook_reasoning_steps",
        title="Cookbook: reasoning node steps",
        cookbook="fsm/reasoning_node_steps_example.py",
        kind="fsm",
        mermaid=(
            "SupportDecision_step_0 --> SupportDecision_step_1 --> SupportDecision_step_2 --> End"
        ),
        backend_apis=_llm_fsm_apis(tools=True),
        expected_nodes=(
            "SupportDecision_step_0",
            "SupportDecision_step_1",
            "SupportDecision_step_2",
        ),
        payload_model="CaseInput",
        payload_kwargs={
            "text": "I need to change my delivery address before the package is shipped."
        },
    )
)
_spec(
    CookbookSpec(
        id="cookbook_semantic_router_parallel",
        title="Cookbook: semantic router (parallel)",
        cookbook="fsm/semantic_router_parallel_example.py",
        kind="fsm",
        mermaid="CaptureRequest --> SupportIntent --> BillingHelp --> End",
        backend_apis=_llm_fsm_apis(semantic=True),
        expected_nodes=("CaptureRequest",),
        payload_model="SupportRequest",
        payload_kwargs={"text": "I want help updating the billing email on my invoice."},
        use_first_candidate_router=True,
    )
)
_spec(
    CookbookSpec(
        id="cookbook_semantic_router_sequential",
        title="Cookbook: semantic router (sequential)",
        cookbook="fsm/semantic_router_sequential_example.py",
        kind="fsm",
        mermaid=(
            "CaptureRequest --> SupportIntent --> InvestigateBilling --> ResolveRequest --> End"
        ),
        backend_apis=_llm_fsm_apis(semantic=True),
        expected_nodes=("CaptureRequest",),
        payload_model="SupportRequest",
        payload_kwargs={"text": "I need to change my shipping address before the order goes out."},
        use_first_candidate_router=True,
    )
)
_spec(
    CookbookSpec(
        id="cookbook_node_kpi",
        title="Cookbook: node KPI",
        cookbook="kpi/node_kpi_example.py",
        kind="fsm",
        mermaid="SummarizeText --> ScoreSummary --> End",
        backend_apis=_llm_fsm_apis(),
        expected_nodes=("SummarizeText", "ScoreSummary"),
        payload_model="ArticleInput",
        payload_kwargs={
            "text": (
                "Researchers at MIT have developed a new battery technology that charges "
                "in under five minutes and lasts three times longer than current lithium-ion cells."
            )
        },
    )
)
_spec(
    CookbookSpec(
        id="cookbook_group_kpi",
        title="Cookbook: group KPI",
        cookbook="kpi/group_kpi_example.py",
        kind="fsm",
        mermaid="ExtractTicket --> ClassifyUrgency --> ScoreTriage --> End",
        backend_apis=_llm_fsm_apis(),
        expected_nodes=("ExtractTicket", "ClassifyUrgency", "ScoreTriage"),
        payload_model="TicketInput",
        payload_kwargs={
            "text": (
                "Hi, I'm James. My account is locked and I can't reset my password. "
                "My email is james@example.com."
            )
        },
    )
)
_spec(
    CookbookSpec(
        id="cookbook_fsm_path_kpi",
        title="Cookbook: FSM path KPI",
        cookbook="kpi/fsm_path_kpi_example.py",
        kind="fsm",
        mermaid="ParseQuery --> GenerateAnswer --> PathScore --> End",
        backend_apis=_llm_fsm_apis(),
        expected_nodes=("ParseQuery", "GenerateAnswer", "PathScore"),
        payload_model="QueryInput",
        payload_kwargs={"question": "What are the main causes of climate change?"},
    )
)
_spec(
    CookbookSpec(
        id="cookbook_node_validation",
        title="Cookbook: node validation",
        cookbook="validation/node_validation_example.py",
        kind="fsm",
        mermaid="SummarizeText --> ValidateSummary --> End",
        backend_apis=_llm_fsm_apis(),
        expected_nodes=("SummarizeText", "ValidateSummary"),
        payload_model="ArticleInput",
        payload_kwargs={
            "text": (
                "Researchers at MIT have developed a new battery technology that charges "
                "in under five minutes and lasts three times longer than current lithium-ion cells."
            )
        },
        require_end=False,
    )
)
_spec(
    CookbookSpec(
        id="cookbook_group_path_validation",
        title="Cookbook: group path validation",
        cookbook="validation/group_path_validation_example.py",
        kind="fsm",
        mermaid="ExtractTicket --> ClassifyUrgency --> ValidateTriage --> End",
        backend_apis=_llm_fsm_apis(),
        expected_nodes=("ExtractTicket", "ClassifyUrgency", "ValidateTriage"),
        payload_model="TicketInput",
        payload_kwargs={
            "text": (
                "Hi, I'm James. My account is locked and I can't reset my password. "
                "My email is james@example.com."
            )
        },
    )
)
_spec(
    CookbookSpec(
        id="cookbook_fsm_path_validation",
        title="Cookbook: FSM path validation",
        cookbook="validation/fsm_path_validation_example.py",
        kind="fsm",
        mermaid="ParseQuery --> GenerateAnswer --> AuditPath --> End",
        backend_apis=_llm_fsm_apis(),
        expected_nodes=("ParseQuery", "GenerateAnswer", "AuditPath"),
        payload_model="QueryInput",
        payload_kwargs={"question": "What are the main causes of climate change?"},
        require_end=False,
    )
)
_spec(
    CookbookSpec(
        id="cookbook_function_calling",
        title="Cookbook: @function_calling",
        cookbook="decorators/function_calling_example.py",
        kind="decorator_functions",
        mermaid="ExtractParams --> greet / summarize --> End",
        backend_apis=_decorator_apis(),
        expected_nodes=("greet",),
        payload_model="UserRequest",
        payload_kwargs={"text": "Please greet María in Spanish"},
        graph_attr="build_workflow",
    )
)
_spec(
    CookbookSpec(
        id="cookbook_workflow_reasoning",
        title="Cookbook: @workflow reasoning",
        cookbook="decorators/workflow_reasoning_example.py",
        kind="decorator_workflow",
        mermaid="lookup_sku --> check_stock --> ExtractParams --> place_order",
        backend_apis=_decorator_apis()
        + (
            _api(
                "POST",
                "/telemetry/runs/{run_id}/events",
                "monitor",
                "tools_executed for lookup_sku and check_stock",
            ),
        ),
        expected_nodes=("place_order",),
        payload_model="UserRequest",
        payload_kwargs={"text": "We need 3 laptops for the sales team"},
        graph_attr="build_workflow",
    )
)
_spec(
    CookbookSpec(
        id="cookbook_filesystem",
        title="Cookbook: filesystem tools",
        cookbook="tools/filesystem_example.py",
        kind="filesystem",
        mermaid="write_file --> ls --> read_file",
        backend_apis=LOCAL_ONLY_APIS,
    )
)
_spec(
    CookbookSpec(
        id="cookbook_web_search",
        title="Cookbook: web search tools",
        cookbook="tools/web_search_example.py",
        kind="web_search",
        mermaid="web_search --> extract_text",
        backend_apis=LOCAL_ONLY_APIS,
    )
)
_spec(
    CookbookSpec(
        id="cookbook_email",
        title="Cookbook: email tools",
        cookbook="tools/email_example.py",
        kind="email",
        mermaid="email_user --> SMTP",
        backend_apis=LOCAL_ONLY_APIS,
    )
)
_spec(
    CookbookSpec(
        id="cookbook_knowledge_retrieval",
        title="Cookbook: FileSystemKnowledge retrieval",
        cookbook="knowledge/retrieval_example.py",
        kind="knowledge_retrieval",
        mermaid="FileSystemKnowledge.search --> hits",
        backend_apis=LOCAL_ONLY_APIS,
    )
)
_spec(
    CookbookSpec(
        id="cookbook_knowledge_transform",
        title="Cookbook: FileSystemKnowledge transform",
        cookbook="knowledge/transform_example.py",
        kind="knowledge_transform",
        mermaid="FileSystemKnowledge --> transform --> summaries",
        backend_apis=LOCAL_ONLY_APIS,
    )
)

COOKBOOK_EXAMPLE_FILES = tuple(spec.cookbook for spec in SPECS.values())


def cookbook_example_paths() -> list[Path]:
    return sorted(
        path for path in COOKBOOK_ROOT.rglob("*_example.py") if "skills" not in path.parts
    )


# ---------------------------------------------------------------------------
# Offline provider / router (copied from the cookbook roundtrip fixture)
# ---------------------------------------------------------------------------


class CookbookSchemaProvider:
    def generate(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        if not schema:
            return "offline reasoning"
        return json.dumps(_schema_value(schema, schema), sort_keys=True)


class FirstCandidateRouter:
    async def route(self, context: Any, candidates: list[Any]) -> RoutingPlan:
        return RoutingPlan(
            reasoning="deterministic cookbook fixture",
            topology="sequential",
            execution_plan=[[0]],
        )


def _schema_value(schema: Any, root: dict[str, Any], name: str = "") -> Any:
    if not isinstance(schema, dict):
        return None
    if "$ref" in schema:
        current: Any = root
        for part in str(schema["$ref"]).removeprefix("#/").split("/"):
            current = current[part]
        return _schema_value(current, root, name)
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    for branch in schema.get("anyOf") or schema.get("oneOf") or ():
        if isinstance(branch, dict) and branch.get("type") != "null":
            return _schema_value(branch, root, name)
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        return {
            key: _schema_value(value, root, key)
            for key, value in (schema.get("properties") or {}).items()
        }
    if kind == "array":
        item = _schema_value(schema.get("items") or {}, root, name)
        if isinstance(item, str):
            return [item, f"{item} two", f"{item} three"]
        return [item]
    if kind == "boolean":
        return True
    if kind == "integer":
        return 3
    if kind == "number":
        return 0.8
    lowered = name.lower()
    if "email" in lowered:
        return "cookbook@example.com"
    if "sku" in lowered:
        return "sku_laptop_pro"
    if "warehouse" in lowered or "location" in lowered:
        return "wh_east"
    if "language" in lowered:
        return "es"
    if "order_id" in lowered:
        return "order-1"
    if "headline" in lowered:
        return "New battery technology charges in minutes and lasts longer"
    if "lane" in lowered:
        return "shipping"
    if "text" in lowered or "intent" in lowered or "request" in lowered:
        return "Please process this valid cookbook support request."
    return "cookbook"


def load_cookbook(relative: str) -> ModuleType:
    path = COOKBOOK_ROOT / relative
    name = "_scenario_cookbook_" + relative.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cookbook_db() -> SqliteDatabase:
    db = SqliteDatabase(name="cookbook_deliveries")
    db.execute(
        "CREATE TABLE cookbook_runs ("
        "scenario_id TEXT PRIMARY KEY, "
        "final_state TEXT, "
        "rejected INTEGER NOT NULL, "
        "rejection TEXT)"
    )
    db.execute(
        "CREATE TABLE node_outputs ("
        "scenario_id TEXT NOT NULL, "
        "node_id TEXT NOT NULL, "
        "status TEXT NOT NULL, "
        "output TEXT)"
    )
    db.execute(
        "CREATE TABLE tool_calls ("
        "scenario_id TEXT NOT NULL, "
        "node_id TEXT NOT NULL, "
        "tool TEXT NOT NULL, "
        "ok INTEGER NOT NULL)"
    )
    db.execute(
        "CREATE TABLE local_artifacts ("
        "scenario_id TEXT NOT NULL, "
        "kind TEXT NOT NULL, "
        "name TEXT NOT NULL, "
        "detail TEXT NOT NULL)"
    )
    db.execute(
        "CREATE TABLE backend_apis ("
        "scenario_id TEXT NOT NULL, "
        "method TEXT NOT NULL, "
        "path TEXT NOT NULL, "
        "when_to_validate TEXT NOT NULL, "
        "purpose TEXT NOT NULL)"
    )
    return db


def record_backend_apis(db: SqliteDatabase, spec: CookbookSpec) -> None:
    for api in spec.backend_apis:
        db.insert_row(
            "backend_apis",
            {
                "scenario_id": spec.id,
                "method": api.method,
                "path": api.path,
                "when_to_validate": api.when,
                "purpose": api.purpose,
            },
        )


def persist_run(db: SqliteDatabase, spec_id: str, result: Any) -> None:
    db.insert_row(
        "cookbook_runs",
        {
            "scenario_id": spec_id,
            "final_state": getattr(result, "final_state", None),
            "rejected": 1 if getattr(result, "rejected", False) else 0,
            "rejection": str(getattr(result, "rejection", "") or ""),
        },
    )
    for step in getattr(result, "steps", []) or []:
        for item in getattr(step, "results", []) or []:
            db.insert_row(
                "node_outputs",
                {
                    "scenario_id": spec_id,
                    "node_id": item.node_id,
                    "status": item.status,
                    "output": json.dumps(item.output, default=str),
                },
            )
            for call in getattr(item, "tool_calls", []) or []:
                db.insert_row(
                    "tool_calls",
                    {
                        "scenario_id": spec_id,
                        "node_id": item.node_id,
                        "tool": call.tool,
                        "ok": 1 if call.ok else 0,
                    },
                )


def _providers_for(graph: Any) -> dict[str, CookbookSchemaProvider]:
    providers = {provider for node in graph.nodes.values() if (provider := node.provider)}
    providers.update(
        provider
        for router in graph.routers.values()
        if (provider := getattr(router, "provider", None))
    )
    providers.add(PROVIDER)
    return {name: CookbookSchemaProvider() for name in providers}


def _payload(module: ModuleType, spec: CookbookSpec) -> Any:
    if not spec.payload_model:
        return None
    model = getattr(module, spec.payload_model)
    return model(**spec.payload_kwargs)


def _run_graph(
    graph: Any, payload: Any, *, tools: Any = None, first_candidate: bool = False
) -> Any:
    if first_candidate:
        router: Any = FirstCandidateRouter()
    else:
        from neosyntropy.core.routing.preferred import PreferredPathRouter

        router = PreferredPathRouter(graph)
    return graph.run(
        payload,
        state={},
        tools=tools,
        providers=_providers_for(graph),
        router=router,
        client=None,
    )


@dataclass
class CookbookScenario:
    spec: CookbookSpec
    db: SqliteDatabase
    module: ModuleType | None = None
    fsm: Any = None
    tools: Any = None
    graphs: list[Any] = field(default_factory=list)

    def run(self, payload: Any = None) -> Any:
        spec = self.spec
        if spec.kind == "filesystem":
            return self._run_filesystem()
        if spec.kind == "web_search":
            return self._run_web_search()
        if spec.kind == "email":
            return self._run_email()
        if spec.kind == "knowledge_retrieval":
            return self._run_knowledge_retrieval()
        if spec.kind == "knowledge_transform":
            return self._run_knowledge_transform()
        if spec.kind == "decorator_functions":
            return self._run_decorator_functions(payload)
        if spec.kind == "decorator_workflow":
            return self._run_decorator_workflow(payload)
        request = payload if payload is not None else _payload(self.module, spec)
        result = _run_graph(
            self.fsm,
            request,
            tools=self.tools,
            first_candidate=spec.use_first_candidate_router,
        )
        persist_run(self.db, spec.id, result)
        return result

    def _run_decorator_functions(self, payload: Any) -> Any:
        greet, summarize = self.graphs
        request = payload if payload is not None else _payload(self.module, self.spec)
        result = _run_graph(greet.__neosyntropy_fsm__, request)
        persist_run(self.db, self.spec.id, result)
        second = _run_graph(
            summarize.__neosyntropy_fsm__,
            self.module.UserRequest(text="Give me a summary of quantum computing in 20 words"),
        )
        persist_run(self.db, f"{self.spec.id}:summarize", second)
        return result

    def _run_decorator_workflow(self, payload: Any) -> Any:
        function = self.graphs[0]
        request = payload if payload is not None else _payload(self.module, self.spec)
        result = _run_graph(function.__neosyntropy_fsm__, request, tools=self.tools)
        persist_run(self.db, self.spec.id, result)
        return result

    def _run_filesystem(self) -> dict[str, Any]:
        from tempfile import TemporaryDirectory

        from neosyntropy.tools.coding.coding_tools import CodingTools

        with TemporaryDirectory(prefix="neosyntropy-cookbook-fs-") as temp_dir:
            registry = CodingTools(
                base_dir=temp_dir,
                enable_read_file=True,
                enable_write_file=True,
                enable_find=True,
                enable_ls=True,
                enable_edit_file=False,
                enable_run_shell=False,
            ).register()
            registry.invoke(
                "write_file",
                {
                    "file_path": "notes.txt",
                    "contents": "NeoSyntropy cookbook\n- write files\n- read files\n",
                },
            )
            listed = registry.invoke("ls", {"path": None, "limit": 20}).result
            notes = registry.invoke(
                "read_file", {"file_path": "notes.txt", "offset": 0, "limit": 20}
            ).result
        self.db.insert_row(
            "local_artifacts",
            {
                "scenario_id": self.spec.id,
                "kind": "file",
                "name": "notes.txt",
                "detail": str(notes),
            },
        )
        self.db.insert_row(
            "cookbook_runs",
            {
                "scenario_id": self.spec.id,
                "final_state": "End",
                "rejected": 0,
                "rejection": "",
            },
        )
        return {"listed": listed, "notes": notes}

    def _run_web_search(self) -> dict[str, Any]:
        fixture = [
            {
                "title": "pathlib — Object-oriented filesystem paths",
                "href": "https://docs.python.org/3/library/pathlib.html",
            }
        ]
        self.db.insert_row(
            "local_artifacts",
            {
                "scenario_id": self.spec.id,
                "kind": "search",
                "name": "Python pathlib Path documentation",
                "detail": json.dumps(fixture),
            },
        )
        self.db.insert_row(
            "cookbook_runs",
            {
                "scenario_id": self.spec.id,
                "final_state": "End",
                "rejected": 0,
                "rejection": "",
            },
        )
        return {"results": fixture}

    def _run_email(self) -> dict[str, Any]:
        payload = {
            "subject": "NeoSyntropy cookbook email",
            "body": "This message was sent by the NeoSyntropy cookbook email example.",
            "sent": False,
        }
        self.db.insert_row(
            "local_artifacts",
            {
                "scenario_id": self.spec.id,
                "kind": "email",
                "name": "intended_message",
                "detail": json.dumps(payload),
            },
        )
        self.db.insert_row(
            "cookbook_runs",
            {
                "scenario_id": self.spec.id,
                "final_state": "End",
                "rejected": 0,
                "rejection": "",
            },
        )
        return payload

    def _run_knowledge_retrieval(self) -> list[Any]:
        from neosyntropy.knowledge.document import Document
        from tests.scenarios.stores import InMemoryVectorDb, ScenarioKnowledge

        corpus = {
            "incident_status.txt": (
                "Incident status: the payment queue is delayed.\n"
                "Owner: finance operations.\n"
                "Next step: confirm the retry job is running.\n"
            ),
            "support_notes.txt": (
                "Support notes: customer asked about renewal timing.\n"
                "Suggested answer: explain the billing date and the grace period.\n"
            ),
            "shipping_update.txt": (
                "Shipping update: the parcel is in transit.\n"
                "Expected delivery: tomorrow afternoon.\n"
            ),
        }
        vector_db = InMemoryVectorDb(name="cookbook_fs")
        knowledge = ScenarioKnowledge(vector_db=vector_db, name="filesystem_corpus")
        knowledge.insert([Document(name=name, content=content) for name, content in corpus.items()])
        hits = knowledge.search("renewal")
        for doc in hits:
            self.db.insert_row(
                "local_artifacts",
                {
                    "scenario_id": self.spec.id,
                    "kind": "document",
                    "name": str(doc.name),
                    "detail": doc.content,
                },
            )
        self.db.insert_row(
            "cookbook_runs",
            {
                "scenario_id": self.spec.id,
                "final_state": "End",
                "rejected": 0,
                "rejection": "",
            },
        )
        return hits

    def _run_knowledge_transform(self) -> list[Any]:
        from collections.abc import Iterable

        from neosyntropy.knowledge.document import Document
        from tests.scenarios.stores import InMemoryVectorDb, ScenarioKnowledge

        documents = [
            Document(
                name="customer_policy.txt",
                content=(
                    "Policy: renewals are billed on the first of the month.\n"
                    "Grace period: 7 days after the due date.\n"
                ),
            ),
            Document(
                name="ops_playbook.txt",
                content=(
                    "Playbook: if billing retries fail, notify support and finance.\n"
                    "Escalation: open an incident if the queue is blocked for 30 minutes.\n"
                ),
            ),
        ]

        def summarize_documents(raw_data: Iterable[Document], **kwargs: object) -> list[Document]:
            summaries: list[Document] = []
            for doc in raw_data:
                lines = [line.strip() for line in doc.content.splitlines() if line.strip()]
                first_line = lines[0] if lines else doc.content.strip()
                summaries.append(
                    Document(
                        name=f"{doc.name}.summary",
                        content=f"{doc.name}: {first_line}",
                        meta_data={
                            "source_name": doc.name,
                            "source_lines": len(lines),
                            "word_count": len(doc.content.split()),
                        },
                    )
                )
            return summaries

        class SearchSource:
            def __init__(self, docs: list[Document], query: str) -> None:
                self.docs = docs
                self.query = query

            def load(self, **kwargs: object) -> list[Document]:
                needle = self.query.lower()
                return [doc for doc in self.docs if needle in doc.content.lower()]

        destination = ScenarioKnowledge(
            vector_db=InMemoryVectorDb(name="summaries"), name="summary_knowledge"
        )
        transformed = ScenarioKnowledge(
            transform=summarize_documents, name="summary_knowledge"
        ).transform(source=SearchSource(documents, "billing"), destination=destination)
        for doc in transformed:
            self.db.insert_row(
                "local_artifacts",
                {
                    "scenario_id": self.spec.id,
                    "kind": "summary",
                    "name": str(doc.name),
                    "detail": doc.content,
                },
            )
        self.db.insert_row(
            "cookbook_runs",
            {
                "scenario_id": self.spec.id,
                "final_state": "End",
                "rejected": 0,
                "rejection": "",
            },
        )
        return list(transformed)


def build_from_spec(spec_id: str) -> CookbookScenario:
    spec = SPECS[spec_id]
    db = cookbook_db()
    record_backend_apis(db, spec)
    module: ModuleType | None = None
    tools = None
    graphs: list[Any] = []
    fsm = None
    if spec.kind not in {"web_search", "email", "knowledge_retrieval", "knowledge_transform"}:
        module = load_cookbook(spec.cookbook)
        tools = module.build_tools() if hasattr(module, "build_tools") else None
        if spec.kind == "decorator_functions":
            graphs.extend(module.build_workflow(client=None, provider=PROVIDER))
        elif spec.kind == "decorator_workflow":
            graphs.append(module.build_workflow(client=None, tools=tools, provider=PROVIDER))
        elif spec.kind == "fsm":
            fsm = (
                module.build_fsm()
                if spec.id == "cookbook_python_node"
                else module.build_fsm(PROVIDER)
            )
    return CookbookScenario(
        spec=spec,
        db=db,
        module=module,  # type: ignore[arg-type]
        fsm=fsm,
        tools=tools,
        graphs=graphs,
    )
