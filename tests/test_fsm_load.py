from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from neosyntropy import (
    FSM,
    DeterministicRouter,
    Group,
    OpenInput,
    SchemaNode,
    SemanticRouter,
    TextOutput,
    ToolRegistry,
    edge_deterministic,
    edge_fallback,
)
from neosyntropy.backend import BackendClient, Client
from neosyntropy.cloud.monitor.graph.manifest import graph_manifest
from neosyntropy.cloud.remote import graph_manifest_with_bundles, load_bundle_callable


@pytest.fixture(autouse=True)
def _enable_remote_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO_REMOTE_EXECUTION", "TRUE")


def _schema_node(node_id: str, *, fallback: bool = False):
    return SchemaNode(
        id=node_id,
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt=f"Run {node_id}",
        is_fallback=fallback,
    )


def test_from_manifest_round_trips_compiled_semantic_graph() -> None:
    target = _schema_node("Handle")
    fallback = _schema_node("Fallback", fallback=True)
    router = SemanticRouter(
        id="Intent",
        input_schema=OpenInput,
        routes={"handle": target},
        fallback_node=fallback,
    )
    authored = FSM(
        entry=router,
        nodes=[target, fallback],
        routers=[router],
        edges=[edge_deterministic("Handle", "End")],
    )

    loaded = FSM.from_manifest(graph_manifest(authored))

    assert loaded.entry_id == authored.entry_id
    assert set(loaded.nodes) == set(authored.nodes)
    assert loaded.router_ids == authored.router_ids
    assert {
        (edge.source, edge.target, edge.kind, edge.target_kind)
        for edge in loaded.edges
    } == {
        (edge.source, edge.target, edge.kind, edge.target_kind)
        for edge in authored.edges
    }


def test_from_manifest_does_not_namespace_compiled_group_nodes_twice() -> None:
    member = _schema_node("DoA")
    group = Group(
        name="skill_a",
        entry=member,
        nodes=[member],
        edges=[edge_deterministic("DoA", "End")],
    )
    fallback = _schema_node("Fallback", fallback=True)
    authored = FSM(
        entry="skill_a__DoA",
        nodes=[fallback],
        groups=[group],
    )

    loaded = FSM.from_manifest(graph_manifest(authored))

    assert "skill_a__DoA" in loaded.nodes
    assert "skill_a__skill_a__DoA" not in loaded.nodes
    assert loaded.groups["skill_a"].entry_id() == "skill_a__DoA"


def test_from_manifest_hydrates_python_handler() -> None:
    manifest = {
        "entry": "Echo",
        "input_schema": {"type": "object"},
        "nodes": [
            {
                "id": "Echo",
                "name": "Echo",
                "kind": "handler",
                "mode": "schema_extraction",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "is_fallback": False,
                "handler_code": {
                    "entry_file": "/app/echo.py",
                    "function_name": "echo",
                    "extractable": True,
                    "vfs": {
                        "/app/echo.py": (
                            "from neosyntropy import node\n"
                            "@node(id='Echo', input_schema={'type': 'object'}, "
                            "output_schema={'type': 'object'})\n"
                            "def echo(ctx):\n"
                            "    return ctx.result(output={'loaded': True})\n"
                        )
                    },
                },
            },
            {
                "id": "Fallback",
                "name": "Fallback",
                "kind": "schema",
                "mode": "schema_extraction",
                "prompt": "fallback",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "is_fallback": True,
            },
        ],
        "edges": [
            {"source": "Echo", "target": "End", "kind": "deterministic"},
            {"source": "Echo", "target": "Fallback", "kind": "fallback"},
        ],
        "groups": [],
        "routers": [],
    }

    loaded = FSM.from_manifest(manifest)

    class Context:
        def result(self, **kwargs: Any) -> dict[str, Any]:
            return kwargs

    assert loaded.nodes["Echo"].handler is not None
    assert loaded.nodes["Echo"].handler(Context()) == {"output": {"loaded": True}}
    result = loaded.run({})
    assert result.final_state == "End"
    assert result.steps[0].results[0].output == {"loaded": True}


def test_load_fetches_manifest_and_preserves_graph_id() -> None:
    from neosyntropy.cloud.monitor._manifest import structure_hash
    from neosyntropy.cloud.remote import recovery_revision

    manifest = {
        "schema_version": 3,
        "entry": "Only",
        "input_schema": {"type": "object"},
        "nodes": [
            {
                "id": "Only",
                "kind": "schema",
                "mode": "schema_extraction",
                "prompt": "only",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            {
                "id": "Fallback",
                "kind": "schema",
                "mode": "schema_extraction",
                "prompt": "fallback",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "is_fallback": True,
            },
        ],
        "edges": [
            {"source": "Only", "target": "End", "kind": "deterministic"},
            {"source": "Only", "target": "Fallback", "kind": "fallback"},
        ],
    }
    manifest["structure_hash"] = structure_hash(manifest)
    recovery = {
        **manifest,
        "code_artifacts": [],
        "recoverable": True,
    }
    recovery["revision"] = recovery_revision(recovery)

    class FakeClient:
        def get_graph(self, graph_id: str) -> dict[str, Any]:
            assert graph_id == "graph-1"
            return {
                "id": graph_id,
                "project_id": "project-1",
                "manifest": manifest,
                "recovery_manifest": recovery,
            }

    loaded = FSM.load("graph-1", client=FakeClient())

    assert loaded.graph_id == "graph-1"
    assert loaded.entry_id == "Only"


def test_load_rejects_missing_manifest() -> None:
    class FakeClient:
        def get_graph(self, graph_id: str) -> dict[str, Any]:
            return {"id": graph_id, "manifest": None}

    with pytest.raises(ValueError, match="does not contain a manifest"):
        FSM.load("graph-1", client=FakeClient())


def test_client_get_graph_without_project_binds_returned_project() -> None:
    client = Client(api_key="nsk_test")

    def fake_get(self: BackendClient, path: str) -> dict[str, Any]:
        assert path == "/observability/graphs/graph-1"
        return {"id": "graph-1", "project_id": "project-1", "manifest": {}}

    with patch.object(BackendClient, "_get", fake_get):
        graph = client.get_graph("graph-1")

    assert graph["id"] == "graph-1"
    assert client.project_id == "project-1"
    assert client._as_backend().project_id == "project-1"


def test_client_get_graph_with_project_uses_scoped_route() -> None:
    client = Client(api_key="nsk_test", project_id="project-1")

    def fake_get(self: BackendClient, path: str) -> dict[str, Any]:
        assert path == "/observability/projects/project-1/graphs/graph-1"
        return {"id": "graph-1", "project_id": "project-1", "manifest": {}}

    with patch.object(BackendClient, "_get", fake_get):
        graph = client.get_graph("graph-1")

    assert graph["project_id"] == "project-1"


def test_client_code_bundle_upload_and_download_are_project_scoped() -> None:
    client = Client(api_key="nsk_test", project_id="project-1")
    bundles = {"a" * 64: b"existing", "b" * 64: b"missing"}
    association = {
        "structure_hash": "structure-1",
        "artifact": {"id": "artifact-1", "sha256": "b" * 64},
    }

    with (
        patch.object(
            BackendClient,
            "missing_code_artifacts",
            return_value=["b" * 64],
        ) as missing,
        patch.object(BackendClient, "upload_code_artifact") as upload,
        patch.object(
            BackendClient,
            "get_graph_artifacts",
            return_value=[association],
        ),
        patch.object(
            BackendClient,
            "get_project_graph",
            return_value={"id": "graph-1", "project_id": "project-1"},
        ),
        patch.object(
            BackendClient,
            "download_code_artifact",
            return_value=b"bundle",
        ) as download,
    ):
        client.upload_code_bundles(bundles)
        loaded = client.get_graph_code_bundles("graph-1")

    missing.assert_called_once_with("project-1", list(bundles))
    upload.assert_called_once_with("project-1", "b" * 64, b"missing")
    download.assert_called_once_with(
        "project-1",
        "artifact-1",
        graph_id="graph-1",
    )
    assert loaded["b" * 64] == (association, b"bundle")


class _BundleClient:
    def __init__(self, packaged: Any, structure: dict[str, Any]) -> None:
        self.packaged = packaged
        self.structure = structure

    def get_graph(self, graph_id: str) -> dict[str, Any]:
        return {
            "id": graph_id,
            "project_id": "project-1",
            "manifest": self.structure,
            "recovery_manifest": self.packaged.manifest,
        }

    def get_graph_code_bundles(
        self, graph_id: str
    ) -> dict[str, tuple[dict[str, Any], bytes]]:
        assert graph_id == "graph-1"
        structure_hash = self.packaged.manifest["structure_hash"]
        return {
            digest: ({"structure_hash": structure_hash}, bundle)
            for digest, bundle in self.packaged.bundles.items()
        }


def _source_module(tmp_path: Path, name: str, source: str) -> types.ModuleType:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def test_remote_load_hydrates_tool_registry(tmp_path: Path) -> None:
    module = _source_module(
        tmp_path,
        "_remote_tool",
        (
            "from neosyntropy import tool\n"
            "from pydantic import BaseModel\n"
            "class LookupArgs(BaseModel):\n"
            "    value: str\n"
            "def register_tools(registry):\n"
            "    @tool(registry=registry)\n"
            "    def lookup(args: LookupArgs):\n"
            "        return {'value': args.value.upper()}\n"
        ),
    )
    registry = ToolRegistry()
    module.register_tools(registry)
    reason = __import__("neosyntropy").ReasoningNode(
        id="Reason",
        input_schema={"type": "object"},
        tools=["lookup"],
        prompt="Use lookup.",
    )
    fallback = _schema_node("Fallback", fallback=True)
    graph = FSM(
        entry=reason,
        nodes=[reason, fallback],
        edges=[
            edge_deterministic("Reason", "End"),
            edge_fallback("Reason", "Fallback"),
        ],
    )
    try:
        packaged = graph_manifest_with_bundles(graph, registry)
        loaded = FSM.load(
            "graph-1", client=_BundleClient(packaged, graph_manifest(graph, registry))
        )
    finally:
        sys.modules.pop(module.__name__, None)

    invocation = loaded.tool_registry.invoke("lookup", {"value": "remote"})
    assert invocation.ok is True
    assert invocation.result == {"value": "REMOTE"}
    override = ToolRegistry()
    assert loaded._control_manager(tools=override).tools is override


def test_remote_load_restores_guard_and_functional_adapter(tmp_path: Path) -> None:
    module = _source_module(
        tmp_path,
        "_remote_validation",
        (
            "from neosyntropy.core.validation import functional_validation_node\n"
            "def allows(state):\n"
            "    return state.get('allowed') is True\n"
            "@functional_validation_node(id='Check', output_key='accepted')\n"
            "def check(ctx):\n"
            "    return ctx.state.get('allowed') is True\n"
        ),
    )
    fallback = _schema_node("Fallback", fallback=True)
    graph = FSM(
        entry=module.check,
        nodes=[module.check, fallback],
        edges=[
            edge_deterministic("Check", "End", guard=module.allows),
            edge_fallback("Check", "Fallback"),
        ],
    )
    try:
        packaged = graph_manifest_with_bundles(graph)
        loaded = FSM.load(
            "graph-1", client=_BundleClient(packaged, graph_manifest(graph))
        )
    finally:
        sys.modules.pop(module.__name__, None)

    guarded = next(edge for edge in loaded.edges if edge.kind == "deterministic")
    assert guarded.guard_allows({"allowed": True}) is True
    assert guarded.guard_allows({"allowed": False}) is False
    result = loaded.run({}, state={"allowed": True})
    assert result.final_state == "End"
    assert result.steps[0].results[0].output == {
        "valid": True,
        "reason": "",
    }


def test_remote_load_mirrors_verified_server_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    module = _source_module(
        tmp_path,
        "_remote_snapshot_handler",
        (
            "from neosyntropy import node\n"
            "@node(id='Handle', input_schema={'type': 'object'}, "
            "output_schema={'type': 'object'})\n"
            "def handle(ctx):\n"
            "    return ctx.result(output={'ok': True})\n"
        ),
    )
    fallback = _schema_node("Fallback", fallback=True)
    graph = FSM(
        entry=module.handle,
        nodes=[module.handle, fallback],
        edges=[
            edge_deterministic("Handle", "End"),
            edge_fallback("Handle", "Fallback"),
        ],
    )
    structure = graph_manifest(graph)
    packaged = graph_manifest_with_bundles(graph)
    record = {
        "id": "graph-1",
        "project_id": "project-1",
        "manifest": structure,
        "recovery_manifest": packaged.manifest,
    }
    associations = [
        {
            "structure_hash": packaged.manifest["structure_hash"],
            "revision": packaged.manifest["revision"],
            "artifact": {"id": digest, "sha256": digest},
        }
        for digest in packaged.bundles
    ]
    snapshot = {
        "graph": record,
        "artifacts": associations,
        "bundles": packaged.bundles,
    }
    client = Client(api_key="nsk_test", project_id="project-1")
    try:
        with (
            patch.object(client, "get_graph", return_value=record),
            patch.object(client, "get_graph_snapshot", return_value=snapshot),
        ):
            loaded = FSM.load("graph-1", client=client)
        with (
            patch.object(client, "get_graph", return_value=record),
            patch.object(
                client,
                "get_graph_snapshot",
                side_effect=AssertionError("cache hit must not download"),
            ),
        ):
            cached = FSM.load("graph-1", client=client)
    finally:
        sys.modules.pop(module.__name__, None)

    assert loaded.graph_id == "graph-1"
    assert cached.graph_id == "graph-1"
    graph_dir = tmp_path / ".neosyntropy/project-1/graphs/graph-1"
    assert (graph_dir / "structure.json").is_file()
    assert (graph_dir / "recovery.json").is_file()
    for digest, bundle in packaged.bundles.items():
        assert (graph_dir / "artifacts" / f"{digest}.json.gz").read_bytes() == bundle


def test_remote_load_restores_deterministic_router_predicate(
    tmp_path: Path,
) -> None:
    module = _source_module(
        tmp_path,
        "_remote_router",
        (
            "def allowed(ctx):\n"
            "    return ctx.state.get('allowed') is True\n"
        ),
    )
    target = _schema_node("Target")
    fallback = _schema_node("Fallback", fallback=True)
    router = DeterministicRouter(
        id="Route",
        input_schema={"type": "object"},
        rules=[(module.allowed, target)],
    )
    graph = FSM(
        entry=router,
        nodes=[target, fallback],
        routers=[router],
        edges=[
            edge_deterministic("Target", "End"),
            edge_fallback("Route", "Fallback"),
        ],
    )
    try:
        packaged = graph_manifest_with_bundles(graph)
        loaded = FSM.load(
            "graph-1", client=_BundleClient(packaged, graph_manifest(graph))
        )
    finally:
        sys.modules.pop(module.__name__, None)

    match = loaded.first_matching_deterministic("Route", {"allowed": True})
    assert match is not None
    assert match.target == "Target"
    assert loaded.first_matching_deterministic("Route", {"allowed": False}) is None


def test_remote_load_rejects_structure_and_bundle_tampering(tmp_path: Path) -> None:
    module = _source_module(
        tmp_path,
        "_remote_integrity",
        "def handler(ctx):\n    return ctx.result(output={'ok': True})\n",
    )
    from neosyntropy.core.node.base import Node

    node = Node(
        id="Handler",
        handler=module.handler,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    fallback = _schema_node("Fallback", fallback=True)
    graph = FSM(
        entry=node,
        nodes=[node, fallback],
        edges=[
            edge_deterministic("Handler", "End"),
            edge_fallback("Handler", "Fallback"),
        ],
    )
    try:
        packaged = graph_manifest_with_bundles(graph)
    finally:
        sys.modules.pop(module.__name__, None)

    changed_structure = graph_manifest(graph)
    changed_structure["entry"] = "Changed"
    tampered_structure = types.SimpleNamespace(
        manifest=packaged.manifest,
        bundles=packaged.bundles,
    )
    with pytest.raises(ValueError, match="structure hash"):
        FSM.load(
            "graph-1",
            client=_BundleClient(tampered_structure, changed_structure),
        )

    digest = next(iter(packaged.bundles))
    tampered_bundles = dict(packaged.bundles)
    tampered_bundles[digest] += b"tampered"
    tampered_code = types.SimpleNamespace(
        manifest=packaged.manifest,
        bundles=tampered_bundles,
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        FSM.load(
            "graph-1",
            client=_BundleClient(tampered_code, graph_manifest(graph)),
        )


def test_bundle_hydration_restores_local_modules_and_typed_bindings() -> None:
    payload = {
        "schema_version": 1,
        "entry_file": "/app/pkg/entry.py",
        "vfs": {
            "/app/pkg/entry.py": (
                "from pkg.constants import BASE\n"
                "def execute(value):\n"
                "    from pkg.lazy import increment\n"
                "    return increment(BASE + value + offsets[0]), marker\n"
            ),
            "/app/pkg/constants.py": "BASE = 10\n",
            "/app/pkg/lazy.py": "def increment(value):\n    return value + 1\n",
        },
        "callable": {"name": "execute"},
        "bindings": {
            "offsets": {"type": "tuple", "items": [2]},
            "marker": {"type": "bytes", "hex": "6f6b"},
        },
    }

    loaded, root = load_bundle_callable(payload, artifact_id="sha256:local")

    assert Path(root).is_dir()
    assert loaded(3) == (16, b"ok")


def test_load_rejects_incompatible_runtime_before_downloading_code() -> None:
    from neosyntropy.cloud.monitor._manifest import structure_hash
    from neosyntropy.cloud.remote import recovery_revision

    structure = {
        "schema_version": 3,
        "entry": "Only",
        "nodes": [],
        "edges": [],
    }
    structure["structure_hash"] = structure_hash(structure)
    manifest = {
        **structure,
        "runtime_compat": {"python": {"requires": ">=99.0"}},
        "code_artifacts": [
            {
                "id": "sha256:unused",
                "sha256": "0" * 64,
                "extractable": True,
                "runtime_compat": {"python": {"requires": ">=99.0"}},
            }
        ],
    }
    manifest["revision"] = recovery_revision(manifest)

    class IncompatibleClient:
        downloaded = False

        def get_graph(self, graph_id: str) -> dict[str, Any]:
            return {
                "id": graph_id,
                "manifest": structure,
                "recovery_manifest": manifest,
            }

        def get_graph_code_bundles(self, graph_id: str) -> dict[str, Any]:
            self.downloaded = True
            return {}

    client = IncompatibleClient()
    with pytest.raises(ValueError, match="requires Python"):
        FSM.load("graph-1", client=client)
    assert client.downloaded is False


def test_from_v3_manifest_restores_metadata_handlers_and_router_definitions() -> None:
    def schema_handler(ctx: Any) -> Any:
        return ctx.result(output={"handled": True})

    def route_allowed(ctx: Any) -> bool:
        return ctx.state.get("allowed") is True

    manifest = {
        "schema_version": 3,
        "entry": "Route",
        "input_schema": {"type": "object"},
        "nodes": [
            {
                "id": "Route",
                "name": "Route",
                "kind": "router",
                "input_schema": {"type": "object"},
            },
            {
                "id": "Intent",
                "name": "Intent",
                "kind": "router",
                "input_schema": {"type": "object"},
            },
            {
                "id": "Work",
                "kind": "schema",
                "mode": "schema_extraction",
                "provider": "vertex/custom",
                "prerequisites": ["Ready"],
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "group": "ops",
                "implementation": {"artifact_ref": "schema-ref"},
            },
            {
                "id": "Fallback",
                "kind": "schema",
                "mode": "schema_extraction",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "is_fallback": True,
            },
        ],
        "edges": [
            {
                "source": "Route",
                "target": "Intent",
                "kind": "deterministic",
                "description": "Route rule[0] -> Intent",
                "guard_ref": "predicate-ref",
            },
            {"source": "Intent", "target": "Work", "kind": "semantic"},
            {"source": "Intent", "target": "Fallback", "kind": "fallback"},
            {"source": "Work", "target": "End", "kind": "deterministic"},
        ],
        "groups": [
            {
                "name": "ops",
                "description": "Operations",
                "metadata": {"owner": "platform"},
                "entry": "Work",
                "namespace": True,
            }
        ],
        "routers": ["Route", "Intent"],
        "routers_detail": [
            {
                "id": "Route",
                "type": "deterministic",
                "state_id": "Route",
                "input_schema": {"type": "object"},
                "rules": [
                    {
                        "index": 0,
                        "target": "Intent",
                        "target_kind": "router",
                        "predicate_ref": "predicate-ref",
                    }
                ],
            },
            {
                "id": "Intent",
                "type": "semantic",
                "state_id": "Intent",
                "input_schema": {"type": "object"},
                "routes": {
                    "work": {"target": "Work", "target_kind": "node"}
                },
                "fallback": {"target": "Fallback", "target_kind": "node"},
                "category": "support",
                "provider": "vertex/router",
            },
        ],
        "allow_unlisted_transitions": True,
        "decorator": "workflow",
        "flags": {"audited": True},
    }

    loaded = FSM.from_manifest(
        manifest,
        code_callables={
            "schema-ref": schema_handler,
            "predicate-ref": route_allowed,
        },
    )

    assert loaded.nodes["Work"].handler is schema_handler
    assert loaded.nodes["Work"].provider == "vertex/custom"
    assert loaded.nodes["Work"].prerequisites == ("Ready",)
    assert loaded.groups["ops"].description == "Operations"
    assert loaded.groups["ops"].metadata == {"owner": "platform"}
    assert loaded.routers["Intent"].category == "support"
    assert loaded.routers["Intent"].provider == "vertex/router"
    assert loaded.first_matching_deterministic("Route", {"allowed": True}) is not None
    assert loaded.decorator == "workflow"
    assert loaded.flags == {"audited": True}


def test_remote_load_restores_lambda_router_predicate(tmp_path: Path) -> None:
    module = _source_module(
        tmp_path,
        "_remote_lambda_router",
        "predicate = lambda ctx: ctx.state.get('route') == 'target'\n",
    )
    target = _schema_node("Target")
    fallback = _schema_node("Fallback", fallback=True)
    router = DeterministicRouter(
        id="Route",
        input_schema={"type": "object"},
        rules=[(module.predicate, target)],
    )
    graph = FSM(
        entry=router,
        nodes=[target, fallback],
        routers=[router],
        edges=[
            edge_deterministic("Target", "End"),
            edge_fallback("Route", "Fallback"),
        ],
    )
    try:
        packaged = graph_manifest_with_bundles(graph)
        loaded = FSM.load(
            "graph-1", client=_BundleClient(packaged, graph_manifest(graph))
        )
    finally:
        sys.modules.pop(module.__name__, None)

    match = loaded.first_matching_deterministic("Route", {"route": "target"})
    assert match is not None
    assert match.target == "Target"
