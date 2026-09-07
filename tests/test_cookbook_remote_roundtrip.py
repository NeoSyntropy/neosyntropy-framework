from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from neosyntropy import FSM
from neosyntropy.core.models import RoutingPlan
from neosyntropy.monitor.graph.manifest import graph_manifest
from neosyntropy.remote import graph_manifest_with_bundles

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = "offline/cookbook"
COOKBOOKS = (
    "fsm/python_node_example.py",
    "fsm/schema_node_example.py",
    "fsm/reasoning_node_prompt_tools_example.py",
    "fsm/reasoning_node_steps_example.py",
    "fsm/semantic_router_parallel_example.py",
    "fsm/semantic_router_sequential_example.py",
    "kpi/node_kpi_example.py",
    "kpi/group_kpi_example.py",
    "kpi/fsm_path_kpi_example.py",
    "validation/node_validation_example.py",
    "validation/group_path_validation_example.py",
    "validation/fsm_path_validation_example.py",
    "decorators/function_calling_example.py",
    "decorators/workflow_reasoning_example.py",
)


@dataclass
class _Case:
    id: str
    graph: FSM
    tools: Any = None


class _BundleClient:
    def __init__(self, packaged: Any, structure: dict[str, Any]) -> None:
        self.packaged = packaged
        self.structure = structure

    def get_graph(self, graph_id: str) -> dict[str, Any]:
        return {
            "id": graph_id,
            "project_id": "cookbook-project",
            "manifest": self.structure,
            "recovery_manifest": self.packaged.manifest,
        }

    def get_graph_code_bundles(
        self, graph_id: str
    ) -> dict[str, tuple[dict[str, Any], bytes]]:
        revision = self.packaged.manifest["revision"]
        return {
            digest: (
                {
                    "structure_hash": self.packaged.manifest["structure_hash"],
                    "revision": revision,
                },
                bundle,
            )
            for digest, bundle in self.packaged.bundles.items()
        }


class _SchemaProvider:
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


class _FirstCandidateRouter:
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
        return [_schema_value(schema.get("items") or {}, root, name)]
    if kind == "boolean":
        return True
    if kind == "integer":
        return 1
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
    if "text" in lowered or "intent" in lowered or "request" in lowered:
        return "Please process this valid cookbook support request."
    return "cookbook"


def _load_module(relative: str) -> ModuleType:
    path = ROOT / "cookbook" / relative
    name = "_cookbook_roundtrip_" + relative.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cases(relative: str) -> list[_Case]:
    module = _load_module(relative)
    tools = module.build_tools() if hasattr(module, "build_tools") else None
    if relative == "decorators/function_calling_example.py":
        functions = module.build_workflow(client=None, provider=PROVIDER)
        return [
            _Case(
                f"{relative}:{function.__name__}",
                function.__neosyntropy_fsm__,
            )
            for function in functions
        ]
    if relative == "decorators/workflow_reasoning_example.py":
        function = module.build_workflow(
            client=None,
            tools=tools,
            provider=PROVIDER,
        )
        return [
            _Case(
                relative,
                function.__neosyntropy_fsm__,
                tools,
            )
        ]
    graph = (
        module.build_fsm()
        if relative == "fsm/python_node_example.py"
        else module.build_fsm(PROVIDER)
    )
    return [_Case(relative, graph, tools)]


def _structure(graph: FSM) -> dict[str, Any]:
    return {
        "entry": graph.entry_id,
        "nodes": {
            node_id: {
                "kind": node.kind,
                "mode": node.mode,
                "provider": node.provider,
                "prerequisites": tuple(node.prerequisites),
                "handler": callable(node.handler),
            }
            for node_id, node in graph.nodes.items()
        },
        "edges": [
            (
                edge.source,
                edge.target,
                edge.kind,
                edge.target_kind,
                edge.description,
                callable(edge.guard),
            )
            for edge in graph.edges
        ],
        "groups": {
            name: {
                "entry": group.entry_id(),
                "parent": group.parent,
                "description": group.description,
                "metadata": group.metadata,
            }
            for name, group in graph.groups.items()
        },
        "routers": {
            name: {
                "type": type(router).__name__,
                "provider": getattr(router, "provider", None),
                "category": getattr(router, "category", None),
                "reasoning": getattr(router, "reasoning", None),
            }
            for name, router in graph.routers.items()
        },
        "decorator": getattr(graph, "decorator", None),
        "allow_unlisted_transitions": graph.allow_unlisted_transitions,
    }


def _result_shape(result: Any) -> dict[str, Any]:
    return {
        "final_state": result.final_state,
        "rejected": result.rejected,
        "rejection": result.rejection,
        "state": result.state,
        "steps": [
            [
                {
                    "node_id": item.node_id,
                    "status": item.status,
                    "output": item.output,
                    "state_updates": item.state_updates,
                    "next_state": item.next_state,
                    "tools": [
                        {
                            "tool": call.tool,
                            "ok": call.ok,
                            "denied": call.denied,
                            "result": call.result,
                            "error": call.error,
                        }
                        for call in item.tool_calls
                    ],
                }
                for item in step.results
            ]
            for step in result.steps
        ],
    }


@pytest.mark.parametrize("relative", COOKBOOKS)
def test_every_cookbook_fsm_round_trips_structure_and_behavior(
    relative: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "NEOSYNTROPY_API_URL",
        "NEOSYNTROPY_API_KEY",
        "NEOSYNTROPY_ACCESS_TOKEN",
        "NEOSYNTROPY_PROJECT_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEO_REMOTE_EXECUTION", "TRUE")

    for case in _cases(relative):
        packaged = graph_manifest_with_bundles(case.graph, case.tools)
        assert packaged.manifest["schema_version"] == 3
        assert packaged.manifest["recoverable"] is True
        loaded = FSM.load(
            "graph-1",
            client=_BundleClient(
                packaged,
                graph_manifest(case.graph, case.tools),
            ),
        )

        assert _structure(loaded) == _structure(case.graph), case.id

        providers = {
            provider
            for node in case.graph.nodes.values()
            if (provider := node.provider)
        }
        providers.update(
            provider
            for router in case.graph.routers.values()
            if (provider := getattr(router, "provider", None))
        )
        provider_map = {name: _SchemaProvider() for name in providers}
        request = _schema_value(case.graph.input_schema, case.graph.input_schema)

        authored_result = case.graph.run(
            request,
            state={},
            tools=case.tools,
            providers=provider_map,
            router=_FirstCandidateRouter(),
        )
        loaded_result = loaded.run(
            request,
            state={},
            providers=provider_map,
            router=_FirstCandidateRouter(),
        )
        assert _result_shape(loaded_result) == _result_shape(authored_result), case.id
