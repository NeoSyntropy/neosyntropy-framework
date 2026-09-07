"""FSM: nodes + edges + groups, validated at construction.

The FSM is the single source of permission. Search/selection is not
permission, router proposals are not permission — only edges listed here
(or an explicit ``allow_unlisted_transitions=True``) permit a transition.

Edge kinds:

- ``deterministic`` — auto-commit when exactly one guard matches
- ``semantic`` — scopes the semantic router to a node or group
- ``fallback`` — used when neither of the above yields a route

``entry`` is required. It is a node or router with ``input_schema``; that
schema is the workflow entry contract (derived onto ``FSM.input_schema``).
Runs begin at ``entry.id`` — there is no synthetic Start vertex.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any

import jsonschema
from pydantic import BaseModel

from .edge import Edge, TransitionTable, edge_deterministic, edge_fallback
from .group import Group, expand_authored_groups, flatten_group_tree
from .node import CombineNode, Node

END = "End"
_FORBIDDEN_START = "Start"

FSMNode = Node | CombineNode


def _flatten_nodes(
    items: Iterable[FSMNode],
) -> tuple[list[Node], list[Edge]]:
    """Expand CombineNode authoring units into concrete nodes + link edges."""
    nodes: list[Node] = []
    auto_edges: list[Edge] = []
    for item in items:
        if isinstance(item, CombineNode):
            expanded, links = item.expand()
            nodes.extend(expanded)
            auto_edges.extend(links)
        elif isinstance(item, Node):
            nodes.append(item)
        else:
            raise FSMValidationError(
                [f"FSM nodes must be Node or CombineNode; got {type(item)!r}"]
            )
    return nodes, auto_edges


def _resolve_routers(
    routers: Iterable[Any],
    entry: Any,
) -> tuple[list[Any], set[str], list[Edge], list[Node]]:
    """Collect router declarations (including nested) and compile edges.

    High-reasoning :class:`SemanticRouter` units also expand a reasoning
    node plus a link edge; those are returned as the fourth tuple item.
    ``router_ids`` uses the actual router state (``{id}.Schema`` when high).
    """
    from .routing.declarations import (
        collect_nested_routers,
        compile_routers,
    )
    from .routing.deterministic import DeterministicRouter
    from .routing.semantic import SemanticRouter

    roots = list(routers)
    if entry is not None and isinstance(entry, (DeterministicRouter, SemanticRouter)):
        if entry not in roots and not any(r.id == entry.id for r in roots):
            roots.insert(0, entry)

    if not roots:
        return [], set(), [], []

    collected: dict[str, Any] = {}
    for root in roots:
        if not isinstance(root, (DeterministicRouter, SemanticRouter)):
            raise FSMValidationError(
                [f"routers must be DeterministicRouter or SemanticRouter; got {type(root)!r}"]
            )
        for item in collect_nested_routers(root):
            collected[item.id] = item

    ordered = list(collected.values())
    extra_nodes: list[Node] = []
    extra_edges: list[Edge] = []
    router_ids: set[str] = set()
    for item in ordered:
        if isinstance(item, SemanticRouter):
            nodes, links = item.expand()
            extra_nodes.extend(nodes)
            extra_edges.extend(links)
            router_ids.add(item.router_state_id)
        else:
            router_ids.add(item.id)
    return ordered, router_ids, [*extra_edges, *compile_routers(ordered)], extra_nodes


class FSMValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid FSM: " + "; ".join(errors))


def _resolve_entry_id(entry: Any) -> str:
    """Return the state id for an authored entry (router, node, or id)."""
    from .routing.deterministic import DeterministicRouter
    from .routing.semantic import SemanticRouter

    if isinstance(entry, (DeterministicRouter, SemanticRouter)):
        return entry.id
    if isinstance(entry, CombineNode):
        return entry.id
    if isinstance(entry, Node):
        return entry.id
    if isinstance(entry, str) and entry.strip():
        return entry
    raise FSMValidationError(
        [f"entry must be a router, node, or id; got {type(entry)!r}"]
    )


def _entry_contract(
    entry: Any,
    entry_id: str,
    nodes: dict[str, Node],
    routers: dict[str, Any],
) -> tuple[dict[str, Any], type[BaseModel] | None]:
    """Derive the FSM entry input_schema from the entry node or router."""
    from .routing.deterministic import DeterministicRouter
    from .routing.semantic import SemanticRouter

    if isinstance(entry, (DeterministicRouter, SemanticRouter)):
        schema = entry.json_schema
        if not schema:
            raise FSMValidationError(
                [
                    f"entry router {entry_id!r} requires input_schema "
                    "(it is the workflow entry contract)"
                ]
            )
        return dict(schema), entry.input_model

    if isinstance(entry, CombineNode):
        node = nodes.get(entry.id)
        if node is None or not node.input_schema:
            raise FSMValidationError(
                [f"entry CombineNode {entry_id!r} requires input_schema"]
            )
        return dict(node.input_schema), node.input_model

    if isinstance(entry, Node):
        if not entry.input_schema:
            raise FSMValidationError(
                [f"entry node {entry_id!r} requires input_schema"]
            )
        return dict(entry.input_schema), entry.input_model

    # String id — resolve against assembled nodes/routers.
    if entry_id in routers:
        router = routers[entry_id]
        schema = getattr(router, "json_schema", None)
        if not schema:
            raise FSMValidationError(
                [
                    f"entry router {entry_id!r} requires input_schema "
                    "(it is the workflow entry contract)"
                ]
            )
        return dict(schema), getattr(router, "input_model", None)
    node = nodes.get(entry_id)
    if node is None:
        raise FSMValidationError(
            [f"entry {entry_id!r} is not a known node or router"]
        )
    if not node.input_schema:
        raise FSMValidationError(
            [f"entry node {entry_id!r} requires input_schema"]
        )
    return dict(node.input_schema), node.input_model


def _load_manifest_handler(node_id: str, raw_code: Any) -> Any:
    """Compile the extracted entry file and return its original Python handler."""
    if not isinstance(raw_code, Mapping) or not raw_code.get("extractable"):
        raise FSMValidationError(
            [f"handler node {node_id!r} has no extractable handler_code"]
        )
    vfs = raw_code.get("vfs")
    if not isinstance(vfs, Mapping) or not vfs:
        raise FSMValidationError(
            [f"handler node {node_id!r} has no handler source in handler_code"]
        )
    entry_file = str(raw_code.get("entry_file") or "")
    if entry_file not in vfs:
        entry_file = next(iter(vfs))
    source = vfs.get(entry_file)
    function_name = str(raw_code.get("function_name") or "").strip()
    if not isinstance(source, str) or not function_name:
        raise FSMValidationError(
            [f"handler node {node_id!r} has incomplete handler_code"]
        )
    namespace: dict[str, Any] = {
        "__file__": entry_file,
        "__name__": f"_neosyntropy_loaded_{PurePosixPath(entry_file).stem}",
        "__package__": None,
    }
    try:
        exec(compile(source, entry_file, "exec"), namespace)
    except Exception as exc:
        raise FSMValidationError(
            [f"could not load handler node {node_id!r}: {exc}"]
        ) from exc
    loaded = namespace.get(function_name)
    handler = loaded.handler if isinstance(loaded, Node) else loaded
    if not callable(handler):
        raise FSMValidationError(
            [f"handler node {node_id!r} did not define {function_name!r}"]
        )
    return handler


def _artifact_ref(raw: Any) -> str | None:
    if isinstance(raw, str) and raw:
        return raw
    if not isinstance(raw, Mapping):
        return None
    value = raw.get("artifact_ref") or raw.get("implementation_ref")
    return str(value) if isinstance(value, str) and value else None


def _deny_loaded_predicate(_state: Any) -> bool:
    """Fail closed when inspecting a manifest without hydrated code."""
    return False


def _node_implementation_ref(raw_node: Mapping[str, Any]) -> str | None:
    return (
        _artifact_ref(raw_node.get("implementation"))
        or _artifact_ref(raw_node.get("implementation_ref"))
    )


def _adapt_loaded_handler(
    node_id: str,
    raw_node: Mapping[str, Any],
    handler: Any,
) -> Any:
    """Recreate framework-owned functional adapters around user callbacks."""
    implementation = raw_node.get("implementation")
    if not isinstance(implementation, Mapping):
        return handler
    adapter = implementation.get("adapter")
    if not isinstance(adapter, Mapping):
        return handler
    adapter_kind = adapter.get("kind") or adapter.get("type")
    if adapter_kind not in {
        "closure_adapter",
        "functional_validation",
        "functional_kpi",
    }:
        return handler
    callable_meta = adapter.get("callable")
    module = (
        str(callable_meta.get("module") or "")
        if isinstance(callable_meta, Mapping)
        else ""
    )
    bindings = adapter.get("bindings")
    bindings = bindings if isinstance(bindings, Mapping) else {}
    output_key = str(bindings.get("output_key") or "")
    common = {
        "id": node_id,
        "name": str(raw_node.get("name") or node_id),
        "description": str(raw_node.get("description") or ""),
        "input_schema": raw_node.get("input_schema") or {"type": "object"},
        "prerequisites": tuple(raw_node.get("prerequisites") or ()),
        "group": raw_node.get("group"),
        "metadata": dict(raw_node.get("metadata") or {}),
    }
    adapter_name = str(adapter.get("name") or adapter.get("adapter") or "")
    if adapter_kind == "functional_validation" or (
        adapter_kind == "closure_adapter"
        and (module.endswith(".validation.node") or adapter_name == "functional_validation")
    ):
        from .validation import functional_validation_node

        rebuilt = functional_validation_node(
            **common,
            output_key=output_key or "valid",
        )(handler)
        return rebuilt.handler
    if adapter_kind == "functional_kpi" or (
        adapter_kind == "closure_adapter"
        and (module.endswith(".kpi.node") or adapter_name == "functional_kpi")
    ):
        from .kpi import functional_kpi_node

        rebuilt = functional_kpi_node(
            **common,
            output_key=output_key or "score",
        )(handler)
        return rebuilt.handler
    return handler


def _json_schema_type(
    raw: Any,
    *,
    root: Mapping[str, Any] | None = None,
    name: str = "RemoteValue",
) -> Any:
    """Best-effort Python type for a recovered JSON Schema field."""
    if not isinstance(raw, Mapping):
        return Any
    root = root or raw
    reference = raw.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/"):
        resolved: Any = root
        try:
            for part in reference[2:].split("/"):
                resolved = resolved[part]
        except (KeyError, TypeError):
            return Any
        return _json_schema_type(resolved, root=root, name=name)
    alternatives = raw.get("anyOf") or raw.get("oneOf")
    if isinstance(alternatives, list):
        choices = [
            _json_schema_type(item, root=root, name=name)
            for item in alternatives
            if isinstance(item, Mapping) and item.get("type") != "null"
        ]
        return choices[0] if choices else Any
    kind = raw.get("type")
    if kind == "object" or isinstance(raw.get("properties"), Mapping):
        return _json_schema_model(name, raw, root=root)
    if kind == "array":
        return list[
            _json_schema_type(
                raw.get("items"),
                root=root,
                name=f"{name}Item",
            )
        ]
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }.get(kind, Any)


def _json_schema_model(
    name: str,
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any] | None = None,
) -> Any:
    from pydantic import ConfigDict, create_model

    root = root or schema
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    required = set(schema.get("required") or ())
    fields = {
        str(field): (
            _json_schema_type(
                field_schema,
                root=root,
                name=f"{name}_{field}",
            ),
            ... if field in required else None,
        )
        for field, field_schema in properties.items()
    }
    return create_model(
        name,
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _loaded_tool_registry(
    manifest: Mapping[str, Any],
    callables: Mapping[str, Any],
) -> Any:
    from ..tools.core.registry import RegisteredTool, ToolRegistry

    catalog = {
        str(item.get("name")): item
        for item in manifest.get("tools") or ()
        if isinstance(item, Mapping) and item.get("name")
    }
    links: dict[str, str] = {}
    for raw_node in manifest.get("nodes") or ():
        if not isinstance(raw_node, Mapping):
            continue
        implementation = raw_node.get("implementation")
        if not isinstance(implementation, Mapping):
            continue
        tools = implementation.get("tools")
        if not isinstance(tools, Mapping):
            continue
        for name, raw_link in tools.items():
            ref = _artifact_ref(raw_link)
            if ref:
                links.setdefault(str(name), ref)

    registry = ToolRegistry()
    for name, ref in sorted(links.items()):
        handler = callables.get(ref)
        if not callable(handler):
            raise FSMValidationError(
                [f"tool {name!r} has no extractable remote code artifact"]
            )
        item = catalog.get(name, {})
        schema = item.get("input_schema")
        schema = schema if isinstance(schema, Mapping) else {}
        args_model = _json_schema_model(f"Remote_{name}_Args", schema)
        registry.register(
            RegisteredTool(
                name=name,
                description=str(item.get("description") or ""),
                args_model=args_model,
                handler=handler,
                json_schema=dict(schema),
                return_schema=(
                    dict(item["output_schema"])
                    if isinstance(item.get("output_schema"), Mapping)
                    else None
                ),
            )
        )
    return registry


def _load_manifest_routers(
    stubs: Mapping[str, Mapping[str, Any]],
    edges: list[Edge],
    providers: Any = None,
    *,
    details: Any = None,
    callables: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Restore router declarations for manifest parity without recompiling edges."""
    from .routing.deterministic import DeterministicRouter
    from .routing.semantic import SemanticRouter

    loaded: dict[str, Any] = {}
    providers = providers if isinstance(providers, Mapping) else {}
    hydrate_code = callables is not None
    callables = callables or {}
    detail_items = (
        list(details.values()) if isinstance(details, Mapping) else list(details or ())
    )
    for detail in detail_items:
        if not isinstance(detail, Mapping):
            continue
        router_id = str(
            detail.get("id") or detail.get("name") or detail.get("router_id") or ""
        ).strip()
        state_id = str(
            detail.get("state_id")
            or detail.get("router_state_id")
            or router_id
        ).strip()
        if not router_id:
            continue
        stub = stubs.get(state_id) or stubs.get(router_id) or {}
        router_type = str(
            detail.get("type") or detail.get("kind") or ""
        ).removesuffix("_router")
        schema = (
            detail.get("input_schema")
            or stub.get("input_schema")
            or {"type": "object"}
        )
        common = {
            "id": router_id,
            "description": str(
                detail.get("description") or stub.get("description") or ""
            ),
            "input_schema": schema,
            "group": detail.get("group", stub.get("group")),
        }
        if router_type == "deterministic":
            raw_rules = detail.get("rules") or ()
            rules = []
            for index, raw_rule in enumerate(raw_rules):
                if not isinstance(raw_rule, Mapping):
                    continue
                reference = _artifact_ref(
                    raw_rule.get("predicate")
                    or raw_rule.get("predicate_ref")
                    or raw_rule.get("guard_ref")
                )
                if reference is None:
                    predicates = detail.get("predicate_refs")
                    if isinstance(predicates, list) and index < len(predicates):
                        reference = _artifact_ref(predicates[index])
                predicate = callables.get(reference or "")
                if not callable(predicate):
                    if hydrate_code:
                        raise FSMValidationError(
                            [
                                f"deterministic router {router_id!r} predicate "
                                f"{reference!r} is unavailable"
                            ]
                        )
                    predicate = _deny_loaded_predicate
                target = (
                    raw_rule.get("target")
                    or raw_rule.get("target_id")
                    or raw_rule.get("to")
                )
                if not target:
                    raise FSMValidationError(
                        [f"deterministic router {router_id!r} rule requires target"]
                    )
                rules.append((predicate, str(target)))
            if not rules:
                # v3 may keep predicate refs on the implementation link and
                # targets in the explicit rule records.
                implementation = detail.get("implementation")
                refs = (
                    implementation.get("predicates")
                    if isinstance(implementation, Mapping)
                    else ()
                )
                targets = [
                    item.get("target")
                    for item in raw_rules
                    if isinstance(item, Mapping) and item.get("target")
                ]
                for reference, target in zip(refs or (), targets, strict=False):
                    predicate = callables.get(str(reference))
                    if callable(predicate):
                        rules.append((predicate, str(target)))
            if rules:
                loaded[router_id] = DeterministicRouter(rules=rules, **common)
                continue
        if router_type == "semantic":
            raw_routes = detail.get("routes") or {}
            if isinstance(raw_routes, Mapping):
                routes = {
                    str(label): (
                        str(target.get("target") or target.get("id"))
                        if isinstance(target, Mapping)
                        else str(target)
                    )
                    for label, target in raw_routes.items()
                }
            else:
                routes = {
                    str(item.get("label") or item.get("name")): str(
                        item.get("target") or item.get("target_id")
                    )
                    for item in raw_routes
                    if isinstance(item, Mapping)
                    and (item.get("label") or item.get("name"))
                    and (item.get("target") or item.get("target_id"))
                }
            fallback = detail.get("fallback") or detail.get("fallback_node")
            if isinstance(fallback, Mapping):
                fallback = fallback.get("target") or fallback.get("id")
            loaded[router_id] = SemanticRouter(
                **common,
                routes=routes,
                fallback_node=str(fallback) if fallback else None,
                category=str(detail.get("category") or "general"),
                provider=str(
                    detail.get("provider")
                    or providers.get(state_id)
                    or "neosyntropy/base"
                ),
                reasoning=str(detail.get("reasoning") or "low"),
                tools=tuple(detail.get("tools") or stub.get("tools") or ()),
                prompt=str(detail.get("prompt") or stub.get("prompt") or ""),
            )

    restored_states = {
        str(getattr(router, "router_state_id", router.id)) for router in loaded.values()
    }
    for state_id, stub in stubs.items():
        if state_id in restored_states or str(stub.get("name") or state_id) in loaded:
            continue
        original_id = str(stub.get("name") or state_id)
        semantic = [
            edge
            for edge in edges
            if edge.source == state_id and edge.kind == "semantic"
        ]
        fallback = next(
            (
                edge.target
                for edge in edges
                if edge.source == state_id and edge.kind == "fallback"
            ),
            None,
        )
        if semantic:
            routes = {
                (edge.description.split(":", 1)[1].split("->", 1)[0].strip()
                 if ":" in edge.description and "->" in edge.description
                 else edge.target): edge.target
                for edge in semantic
            }
            router = SemanticRouter(
                id=original_id,
                routes=routes,
                fallback_node=fallback,
                description=str(stub.get("description") or ""),
                input_schema=stub.get("input_schema") or {"type": "object"},
                group=stub.get("group"),
                provider=str(providers.get(state_id) or "neosyntropy/base"),
                reasoning="high" if state_id != original_id else "low",
                tools=tuple(stub.get("tools") or ()),
                prompt=str(stub.get("prompt") or ""),
            )
            loaded[original_id] = router
            continue
        loaded[original_id] = SimpleNamespace(
            id=original_id,
            router_state_id=state_id,
            description=str(stub.get("description") or ""),
            prompt=stub.get("prompt"),
            tools=tuple(stub.get("tools") or ()),
            json_schema=stub.get("input_schema") or {"type": "object"},
            group=stub.get("group"),
            provider=str(providers.get(state_id) or "neosyntropy/base"),
        )
    return loaded


class FSM:
    @classmethod
    def load(cls, graph_id: str, *, client: Any) -> FSM:
        """Load structure and verified code blobs for a stored graph UUID."""
        from .._features import remote_execution_enabled

        if not remote_execution_enabled():
            raise RuntimeError(
                "remote graph loading is disabled; set "
                "NEO_REMOTE_EXECUTION=TRUE to enable it"
            )
        if client is None or not hasattr(client, "get_graph"):
            raise TypeError("client must be a NeoSyntropy Client")
        graph_id = str(graph_id).strip()
        if not graph_id:
            raise ValueError("graph_id is required")
        record = client.get_graph(graph_id)
        if not isinstance(record, Mapping):
            raise FSMValidationError(["backend returned an invalid graph record"])
        structure_manifest = record.get("manifest")
        recovery_manifest = record.get("recovery_manifest")
        manifest = (
            recovery_manifest
            if isinstance(recovery_manifest, Mapping)
            else structure_manifest
        )
        if not isinstance(manifest, Mapping):
            raise FSMValidationError(["stored graph does not contain a manifest"])
        if (
            not isinstance(recovery_manifest, Mapping)
            and "code_artifacts" not in manifest
        ):
            raise FSMValidationError(
                ["stored graph does not have a remotely recoverable revision"]
            )
        code_callables: dict[str, Any] = {}
        bundle_roots: list[str] = []
        graph_snapshot: Mapping[str, Any] | None = None
        snapshot_needs_write = False
        raw_artifacts = manifest.get("code_artifacts")
        from ..cloud.monitor._manifest import structure_hash as monitor_structure_hash
        from ..cloud.remote import recovery_revision

        expected_structure_hash = str(manifest.get("structure_hash") or "")
        expected_revision = str(manifest.get("revision") or "")
        if (
            isinstance(recovery_manifest, Mapping)
            and (
                not isinstance(structure_manifest, Mapping)
                or monitor_structure_hash(structure_manifest)
                != expected_structure_hash
            )
        ):
            raise FSMValidationError(["stored graph structure hash is invalid"])
        if not expected_revision or recovery_revision(manifest) != expected_revision:
            raise FSMValidationError(["stored graph recovery revision is invalid"])
        from ..backend import Client as PublicClient

        if isinstance(client, PublicClient):
            from ..cloud.remote.snapshot import read_graph_snapshot

            graph_snapshot = read_graph_snapshot(record)
        if raw_artifacts:
            from ..cloud.remote import (
                CodeBundleError,
                decode_code_bundle,
                load_bundle_callable,
                validate_manifest_compatibility,
            )

            try:
                validate_manifest_compatibility(manifest)
                for raw_ref in raw_artifacts:
                    if isinstance(raw_ref, Mapping):
                        validate_manifest_compatibility(
                            {
                                "schema_version": manifest.get("schema_version", 1),
                                "runtime_compat": raw_ref.get("runtime_compat"),
                                "dependency_lock": raw_ref.get("dependency_lock"),
                            }
                        )
            except CodeBundleError as exc:
                raise FSMValidationError(
                    [f"stored graph runtime is incompatible: {exc}"]
                ) from exc

            if not hasattr(client, "get_graph_code_bundles"):
                raise TypeError(
                    "client must support graph code artifact downloads"
                )
            if isinstance(client, PublicClient):
                if graph_snapshot is None:
                    graph_snapshot = client.get_graph_snapshot(
                        str(record.get("id") or graph_id),
                        graph_record=record,
                    )
                    snapshot_needs_write = True
                snapshot_associations = graph_snapshot.get("artifacts", [])
                snapshot_bundles = graph_snapshot.get("bundles", {})
                downloaded = {
                    str(association["artifact"]["sha256"]): (
                        association,
                        snapshot_bundles[
                            str(association["artifact"]["sha256"])
                        ],
                    )
                    for association in snapshot_associations
                }
            else:
                downloaded = client.get_graph_code_bundles(
                    str(record.get("id") or graph_id)
                )
            if not isinstance(downloaded, Mapping):
                raise FSMValidationError(
                    ["backend returned invalid graph code artifacts"]
                )
            decoded_artifacts: list[tuple[str, str, Mapping[str, Any]]] = []
            for raw_ref in raw_artifacts:
                if not isinstance(raw_ref, Mapping):
                    continue
                artifact_id = str(raw_ref.get("id") or "")
                digest = raw_ref.get("sha256")
                required = bool(raw_ref.get("required", True))
                if not digest or raw_ref.get("extractable") is False:
                    if required and raw_ref.get("publishable") is False:
                        reason = str(raw_ref.get("reason") or "artifact is unavailable")
                        raise FSMValidationError(
                            [f"required code artifact {artifact_id!r} is unavailable: {reason}"]
                        )
                    continue
                stored = downloaded.get(str(digest))
                if (
                    not isinstance(stored, tuple)
                    or len(stored) != 2
                    or not isinstance(stored[1], bytes)
                ):
                    raise FSMValidationError(
                        [f"code artifact {digest} is missing from stored graph"]
                    )
                association, bundle = stored
                if isinstance(association, Mapping):
                    associated_hash = str(
                        association.get("structure_hash") or ""
                    )
                    if associated_hash and associated_hash != expected_structure_hash:
                        raise FSMValidationError(
                            [f"code artifact {digest} belongs to another graph revision"]
                        )
                    associated_revision = str(
                        association.get("revision") or ""
                    )
                    if (
                        associated_revision
                        and associated_revision != expected_revision
                    ):
                        raise FSMValidationError(
                            [f"code artifact {digest} has a mismatched revision"]
                        )
                try:
                    payload = decode_code_bundle(
                        bundle, expected_sha256=str(digest)
                    )
                except CodeBundleError as exc:
                    raise FSMValidationError(
                        [f"could not hydrate code artifact {digest}: {exc}"]
                    ) from exc
                decoded_artifacts.append((artifact_id, str(digest), payload))

            pending = decoded_artifacts
            while pending:
                deferred: list[tuple[str, str, Mapping[str, Any]]] = []
                progress = False
                last_error: CodeBundleError | None = None
                for artifact_id, digest, payload in pending:
                    try:
                        validate_manifest_compatibility(
                            {
                                "schema_version": manifest.get("schema_version", 1),
                                "runtime_compat": payload.get("runtime_compat"),
                                "dependency_lock": payload.get("dependency_lock"),
                            }
                        )
                        loaded, root = load_bundle_callable(
                            payload,
                            artifact_id=artifact_id,
                            callables=code_callables,
                        )
                    except CodeBundleError as exc:
                        if "callable binding" in str(exc) and "unavailable" in str(exc):
                            deferred.append((artifact_id, digest, payload))
                            last_error = exc
                            continue
                        raise FSMValidationError(
                            [f"could not hydrate code artifact {digest}: {exc}"]
                        ) from exc
                    code_callables[artifact_id] = loaded
                    bundle_roots.append(root)
                    progress = True
                if deferred and not progress:
                    raise FSMValidationError(
                        [f"could not resolve code artifact bindings: {last_error}"]
                    )
                pending = deferred

        fsm = cls.from_manifest(manifest, code_callables=code_callables)
        fsm.graph_id = str(record.get("id") or graph_id)
        from ..backend import BackendClient, Client

        fsm._remote_client = (
            client if isinstance(client, (Client, BackendClient)) else None
        )
        fsm._code_bundle_roots = bundle_roots
        if isinstance(client, Client):
            if graph_snapshot is None:
                graph_snapshot = client.get_graph_snapshot(
                    fsm.graph_id,
                    graph_record=record,
                )
                snapshot_needs_write = True
            from ..cloud.remote.snapshot import write_graph_snapshot

            if snapshot_needs_write:
                write_graph_snapshot(
                    graph_snapshot["graph"],
                    graph_snapshot["artifacts"],
                    graph_snapshot["bundles"],
                )
        return fsm

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        code_callables: Mapping[str, Any] | None = None,
    ) -> FSM:
        """Reconstruct an FSM from its already-compiled graph manifest."""
        if not isinstance(manifest, Mapping):
            raise TypeError("manifest must be a mapping")

        graph = cls.__new__(cls)
        graph.groups = {}
        for raw_group in manifest.get("groups") or []:
            if not isinstance(raw_group, Mapping) or not raw_group.get("name"):
                continue
            group = Group(
                name=str(raw_group["name"]),
                description=str(raw_group.get("description") or ""),
                metadata=dict(raw_group.get("metadata") or {}),
                entry=raw_group.get("entry"),
                parent=raw_group.get("parent"),
                # Manifest ids are already compiled; never namespace entry twice.
                namespace=False,
            )
            object.__setattr__(
                group, "_namespace", bool(raw_group.get("namespace", False))
            )
            graph.groups[group.name] = group

        graph.nodes = {}
        router_stubs: dict[str, Mapping[str, Any]] = {}
        for raw_node in manifest.get("nodes") or []:
            if not isinstance(raw_node, Mapping):
                continue
            node_id = str(raw_node.get("id") or "").strip()
            if not node_id:
                raise FSMValidationError(["manifest node requires id"])
            if raw_node.get("kind") == "router":
                router_stubs[node_id] = raw_node
                continue
            handler = None
            implementation_ref = _node_implementation_ref(raw_node)
            if implementation_ref and code_callables is not None:
                handler = code_callables.get(implementation_ref)
                if not callable(handler):
                    artifact = next(
                        (
                            item
                            for item in manifest.get("code_artifacts") or ()
                            if isinstance(item, Mapping)
                            and item.get("id") == implementation_ref
                        ),
                        {},
                    )
                    reason = (
                        str(artifact.get("reason") or "artifact is unavailable")
                        if isinstance(artifact, Mapping)
                        else "artifact is unavailable"
                    )
                    raise FSMValidationError(
                        [
                            f"node {node_id!r} has no extractable remote "
                            f"code artifact: {reason}"
                        ]
                    )
                handler = _adapt_loaded_handler(node_id, raw_node, handler)
            elif raw_node.get("kind") == "handler":
                if raw_node.get("handler_code"):
                    handler = _load_manifest_handler(
                        node_id, raw_node.get("handler_code")
                    )
                elif code_callables is not None:
                    raise FSMValidationError(
                        [f"handler node {node_id!r} has no implementation reference"]
                    )
            try:
                node = Node(
                    id=node_id,
                    name=str(raw_node.get("name") or node_id),
                    description=str(raw_node.get("description") or ""),
                    provider=str(raw_node.get("provider") or "neosyntropy/base"),
                    prompt=str(raw_node.get("prompt") or ""),
                    prerequisites=tuple(raw_node.get("prerequisites") or ()),
                    tools=tuple(raw_node.get("tools") or ()),
                    mode=raw_node.get("mode"),
                    kind=raw_node.get("kind"),
                    input_schema=raw_node.get("input_schema") or {},
                    output_schema=raw_node.get("output_schema") or {},
                    group=raw_node.get("group"),
                    is_fallback=bool(raw_node.get("is_fallback")),
                    metadata=dict(raw_node.get("metadata") or {}),
                    handler=handler,
                )
            except (TypeError, ValueError) as exc:
                raise FSMValidationError(
                    [f"invalid manifest node {node_id!r}: {exc}"]
                ) from exc
            if callable(handler) and node.kind == "schema":
                object.__setattr__(
                    node,
                    "output_model",
                    _json_schema_model(
                        f"Remote_{node_id}_Output",
                        node.output_schema,
                    ),
                )
            if node.id in graph.nodes:
                raise FSMValidationError([f"duplicate node id {node.id!r}"])
            graph.nodes[node.id] = node

        graph.edges = []
        raw_edges: list[Mapping[str, Any]] = []
        for raw_edge in manifest.get("edges") or []:
            try:
                if isinstance(raw_edge, Mapping):
                    raw_edges.append(raw_edge)
                graph.edges.append(
                    raw_edge
                    if isinstance(raw_edge, Edge)
                    else Edge.model_validate(raw_edge)
                )
            except (TypeError, ValueError) as exc:
                raise FSMValidationError([f"invalid manifest edge: {exc}"]) from exc
        if code_callables is not None:
            from .routing.deterministic import _wrap_guard

            for edge, raw_edge in zip(graph.edges, raw_edges, strict=False):
                guard_ref = raw_edge.get("guard_ref")
                if not guard_ref:
                    continue
                guard = code_callables.get(str(guard_ref))
                if not callable(guard):
                    raise FSMValidationError(
                        [f"deterministic edge guard {guard_ref!r} is unavailable"]
                    )
                if " rule[" in edge.description:
                    guard = _wrap_guard(guard)
                object.__setattr__(edge, "guard", guard)

        graph.router_ids = {
            str(router_id)
            for router_id in manifest.get("routers") or ()
            if str(router_id).strip()
        }
        graph.router_ids.update(router_stubs)
        graph.routers = _load_manifest_routers(
            router_stubs,
            graph.edges,
            manifest.get("router_providers"),
            details=manifest.get("routers_detail"),
            callables=code_callables,
        )
        graph.router_ids.update(
            str(getattr(router, "router_state_id", router.id))
            for router in graph.routers.values()
        )
        graph.entry_id = str(manifest.get("entry") or "").strip()
        graph.input_schema = dict(manifest.get("input_schema") or {})
        graph.input_model = None
        graph.allow_unlisted_transitions = bool(
            manifest.get("allow_unlisted_transitions", False)
        )
        graph.decorator = manifest.get("decorator")
        graph.flags = dict(manifest.get("flags") or {})
        graph.runtime_compat = manifest.get("runtime_compat")
        graph.dependency_lock = manifest.get("dependency_lock")
        graph.tool_registry = (
            _loaded_tool_registry(manifest, code_callables)
            if code_callables is not None
            else None
        )
        graph._validate(validate_reachability=True)
        return graph

    def __init__(
        self,
        *,
        nodes: Iterable[FSMNode],
        entry: Any,
        edges: Iterable[Edge | dict[str, Any]] = (),
        groups: Iterable[Group | str] = (),
        routers: Iterable[Any] = (),
        allow_unlisted_transitions: bool = False,
        validate_reachability: bool = True,
    ):
        if entry is None:
            raise FSMValidationError(
                ["FSM requires entry= (a node or router with input_schema)"]
            )

        from .routing.deterministic import DeterministicRouter
        from .routing.semantic import SemanticRouter

        self.groups: dict[str, Group] = {}
        resolved_groups: list[Group] = []
        for group in groups:
            resolved = group if isinstance(group, Group) else Group(name=group)
            for item in flatten_group_tree([resolved]):
                if item.name not in self.groups:
                    self.groups[item.name] = item
                    resolved_groups.append(item)

        # Groups used as semantic route targets may also author nodes/routers.
        for root in [*routers, entry]:
            if not isinstance(root, (DeterministicRouter, SemanticRouter)):
                continue
            from .routing.declarations import collect_nested_routers

            for item in collect_nested_routers(root):
                if not isinstance(item, SemanticRouter):
                    continue
                for target in item.routes.values():
                    if isinstance(target, Group):
                        for nested in flatten_group_tree([target]):
                            if nested.name not in self.groups:
                                self.groups[nested.name] = nested
                                resolved_groups.append(nested)

        try:
            group_nodes, group_routers, group_edges = expand_authored_groups(
                resolved_groups
            )
        except ValueError as exc:
            raise FSMValidationError([str(exc)]) from exc

        flat_nodes, auto_edges = _flatten_nodes([*nodes, *group_nodes])
        self.nodes: dict[str, Node] = {}
        for item in flat_nodes:
            if item.id in self.nodes:
                raise FSMValidationError([f"duplicate node id {item.id!r}"])
            self.nodes[item.id] = item

        resolved_routers, router_ids, router_edges, router_nodes = _resolve_routers(
            [*routers, *group_routers], entry
        )
        for item in router_nodes:
            if item.id in self.nodes:
                raise FSMValidationError([f"duplicate node id {item.id!r}"])
            self.nodes[item.id] = item
        self.routers = {item.id: item for item in resolved_routers}
        self.router_ids = set(router_ids)

        overlap = self.router_ids & set(self.nodes)
        if overlap:
            raise FSMValidationError(
                [f"router id clashes with node id: {sorted(overlap)}"]
            )

        self.entry_id = _resolve_entry_id(entry)
        if self.entry_id == _FORBIDDEN_START:
            raise FSMValidationError(
                ["entry cannot be the removed synthetic 'Start' state"]
            )
        if (
            self.entry_id not in self.nodes
            and self.entry_id not in self.router_ids
        ):
            raise FSMValidationError(
                [f"entry {self.entry_id!r} is not a known node or router"]
            )

        self.input_schema, self.input_model = _entry_contract(
            entry, self.entry_id, self.nodes, self.routers
        )

        declared = [
            edge if isinstance(edge, Edge) else Edge.model_validate(edge)
            for edge in edges
        ]
        # Router/Combine/group auto edges first; author edges may still add more.
        self.edges: list[Edge] = [
            *auto_edges,
            *router_edges,
            *group_edges,
            *declared,
        ]
        self.allow_unlisted_transitions = allow_unlisted_transitions

        # Groups referenced by nodes are auto-registered.
        for item in self.nodes.values():
            if item.group and item.group not in self.groups:
                self.groups[item.group] = Group(name=item.group)

        self._validate(validate_reachability)

    # -- validation ---------------------------------------------------------

    def _validate(self, validate_reachability: bool) -> None:
        errors: list[str] = []
        if not self.nodes:
            errors.append("graph must define at least one node")
        if not self.entry_id:
            errors.append("FSM requires entry=")
        if not self.input_schema:
            errors.append(
                "FSM entry must declare input_schema"
            )

        known_nodes = set(self.nodes) | set(self.router_ids) | {END}
        for edge in self.edges:
            if edge.source == _FORBIDDEN_START or edge.target == _FORBIDDEN_START:
                errors.append(
                    "edges must not use the removed synthetic 'Start' state; "
                    "set entry= instead"
                )
                continue
            if edge.source not in known_nodes:
                errors.append(
                    f"edge source {edge.source!r} is not a known node, "
                    f"router, or End"
                )
            if edge.target_kind == "group":
                if edge.target not in self.groups:
                    errors.append(
                        f"semantic edge targets unknown group {edge.target!r}"
                    )
            elif edge.target not in known_nodes:
                errors.append(
                    f"edge target {edge.target!r} is not a known node, "
                    f"router, or End"
                )

        fallback_count = sum(item.is_fallback for item in self.nodes.values())
        if fallback_count != 1:
            errors.append("graph must define exactly one dedicated fallback node")

        missing_output = sorted(
            item.id for item in self.nodes.values() if not item.output_schema
        )
        if missing_output:
            errors.append(
                f"every node requires output_schema; missing on: {missing_output}"
            )
        missing_input = sorted(
            item.id for item in self.nodes.values() if not item.input_schema
        )
        if missing_input:
            errors.append(
                f"every node requires input_schema; missing on: {missing_input}"
            )

        for group in self.groups.values():
            group_entry = group.entry_id()
            if group_entry is not None and group_entry not in known_nodes:
                errors.append(
                    f"group {group.name!r} entry {group_entry!r} is not a known "
                    f"node or router"
                )

        if errors:
            raise FSMValidationError(errors)

        if validate_reachability and self.edges:
            self._validate_reachability(errors)
        if errors:
            raise FSMValidationError(errors)

    def _concrete_targets(self, edge: Edge) -> set[str]:
        if edge.target_kind == "group":
            group = self.groups.get(edge.target)
            group_entry = group.entry_id() if group is not None else None
            if group_entry is not None:
                return {group_entry}
            return {node.id for node in self.nodes_in_group(edge.target)}
        return {edge.target}

    def _validate_reachability(self, errors: list[str]) -> None:
        """Entry/End discipline with group-target expansion.

        Applies only to vertices that participate in edges: every such vertex
        must be reachable from the declared entry, and (when End is used) must
        be able to reach End. The fallback node is exempt — it is a safe stop,
        not a path member. Capability-only nodes (no edges) are exempt as well;
        the plan validator fail-closes on them unless unlisted transitions
        are explicitly allowed.
        """
        participants: set[str] = {self.entry_id}
        forward: dict[str, set[str]] = {}
        backward: dict[str, set[str]] = {}
        for edge in self.edges:
            if edge.kind == "fallback":
                # Fallback edges are safe-stop exits, not primary path members.
                continue
            participants.add(edge.source)
            for target in self._concrete_targets(edge):
                participants.add(target)
                forward.setdefault(edge.source, set()).add(target)
                backward.setdefault(target, set()).add(edge.source)

        fallback_id = self.fallback_node.id
        reachable = _flood(self.entry_id, forward)
        unreachable = sorted(
            vertex
            for vertex in participants
            if vertex not in reachable
            and vertex not in {self.entry_id, fallback_id}
        )
        if unreachable:
            errors.append(
                f"states unreachable from entry {self.entry_id!r}: {unreachable}"
            )

        if END in participants:
            can_finish = _flood(END, backward)
            stuck = sorted(
                vertex
                for vertex in reachable
                if vertex not in can_finish and vertex not in {END, fallback_id}
            )
            if stuck:
                errors.append(f"states that cannot reach End: {stuck}")

    # -- entry contract -----------------------------------------------------

    def entry_input_error(self, payload: Mapping[str, Any]) -> str | None:
        """Return why ``payload`` fails the entry input contract, or None when it holds.

        Only meaningful for a run starting at ``entry_id``. The contract is
        derived from the entry node/router ``input_schema`` and validates the
        run ``input`` channel — not workflow ``state``.
        """
        if not self.input_schema:
            return "graph is missing required entry input_schema"
        try:
            jsonschema.validate(instance=dict(payload), schema=self.input_schema)
        except jsonschema.exceptions.ValidationError as exc:
            return f"entry input does not match the graph input schema: {exc.message}"
        except jsonschema.exceptions.SchemaError as exc:
            return f"graph input schema is invalid: {exc.message}"
        return None

    # -- permissions --------------------------------------------------------

    @property
    def fallback_node(self) -> Node:
        return next(item for item in self.nodes.values() if item.is_fallback)

    def transition_table(self) -> TransitionTable:
        group_entries = {
            name: entry
            for name, group in self.groups.items()
            if (entry := group.entry_id()) is not None
        }
        return TransitionTable.from_edges(
            self.edges,
            node_groups={node_id: node.group for node_id, node in self.nodes.items()},
            group_entries=group_entries,
            allow_unlisted_transitions=self.allow_unlisted_transitions,
        )

    def allows(self, source: str, target: str) -> bool:
        return self.transition_table().permits(source, target)

    def guard_allows(self, source: str, target: str, state: dict[str, Any]) -> bool:
        """Evaluate guards for a transition, fail-closed per edge.

        Deterministic edges own the gate for a concrete target: when any
        deterministic edge lists ``source -> target``, at least one of those
        guards must allow. Semantic edges cannot bypass a failing
        deterministic guard. When no listed edge matches (permissive graphs
        or default self-transitions), guards are vacuously satisfied.
        """
        matching = [
            edge
            for edge in self.edges
            if edge.source == source
            and (
                (edge.target_kind == "node" and edge.target == target)
                or (
                    edge.target_kind == "group"
                    and self.nodes.get(target) is not None
                    and self.nodes[target].group == edge.target
                )
            )
        ]
        if not matching:
            return True
        deterministic = [edge for edge in matching if edge.kind == "deterministic"]
        if deterministic:
            return any(edge.guard_allows(state) for edge in deterministic)
        return any(edge.guard_allows(state) for edge in matching)

    def outgoing(self, source: str, *, kind: str | None = None) -> list[Edge]:
        edges = [edge for edge in self.edges if edge.source == source]
        if kind is None:
            return edges
        return [edge for edge in edges if edge.kind == kind]

    def matching_deterministic(
        self, source: str, state: dict[str, Any]
    ) -> list[Edge]:
        """Deterministic edges whose guards allow ``state`` (fail-closed).

        Order matches compile / authoring order (DeterministicRouter: first
        matching rule wins — use :meth:`first_matching_deterministic`).
        """
        return [
            edge
            for edge in self.outgoing(source, kind="deterministic")
            if edge.guard_allows(state)
        ]

    def first_matching_deterministic(
        self, source: str, state: dict[str, Any]
    ) -> Edge | None:
        """First guard-allowed deterministic edge (DeterministicRouter semantics)."""
        for edge in self.outgoing(source, kind="deterministic"):
            if edge.guard_allows(state):
                return edge
        return None

    def semantic_candidate_ids(self, source: str) -> set[str] | None:
        """Node ids in scope for the semantic router from ``source``.

        Returns ``None`` when there are no outgoing semantic edges (caller
        should use the fallback edge). Returns an empty set when semantic
        edges exist but resolve to no actionable nodes.
        """
        semantic = self.outgoing(source, kind="semantic")
        if not semantic:
            return None
        scoped: set[str] = set()
        for edge in semantic:
            for target in self._concrete_targets(edge):
                if target in self.nodes and not self.nodes[target].is_fallback:
                    scoped.add(target)
        return scoped

    def fallback_target(self, source: str) -> str:
        """Target of the fallback edge from ``source``, else the fallback node."""
        edges = self.outgoing(source, kind="fallback")
        if len(edges) == 1:
            return edges[0].target
        if len(edges) > 1:
            # Ambiguous fallback declarations fail closed to the dedicated node.
            return self.fallback_node.id
        return self.fallback_node.id

    # -- organization -------------------------------------------------------

    def nodes_in_group(self, name: str) -> list[Node]:
        return [item for item in self.nodes.values() if item.group == name]

    def is_router_state(self, state_id: str) -> bool:
        """True when ``state_id`` is a compiled router (not an executable node)."""
        return state_id in self.router_ids

    # -- run (control cycle) ------------------------------------------------

    def run(
        self,
        request: Any = None,
        *,
        client: Any = None,
        tools: Any = None,
        state: Mapping[str, Any] | None = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> Any:
        """Run the FSM. Workflows begin at ``entry`` — do not pass a Start state.

        Application code::

            result = fsm.run(EntryInput(...), state={...}, client=client)

        ``request`` (or the first positional) is the run ``input`` and must match
        the entry ``input_schema``. ``state`` is a separate mutable workflow bag.

        By default ``until_end=True`` advances cycles until ``End`` or rejection.
        Pass ``until_end=False`` for a single control cycle (resume via
        ``current_state`` only when you intentionally continue mid-path).
        """
        return self._run_loop(
            request,
            client=client,
            tools=tools,
            state=state,
            until_end=until_end,
            max_cycles=max_cycles,
            **kwargs,
        )

    async def arun(
        self,
        request: Any = None,
        *,
        client: Any = None,
        tools: Any = None,
        state: Mapping[str, Any] | None = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> Any:
        """Async form of :meth:`run`."""
        import asyncio

        # Reuse the sync loop via a thread when callers already use arun with
        # until_end; single-cycle path stays fully async.
        payload = self._normalize_request(request, state=state)
        manager = self._control_manager(client=client, tools=tools, **kwargs)
        if not until_end:
            return await manager.arun(payload)

        return await asyncio.to_thread(
            self._run_loop,
            payload,
            client=client,
            tools=tools,
            until_end=True,
            max_cycles=max_cycles,
            **kwargs,
        )

    def run_batch(
        self,
        requests: list[Any],
        *,
        batch_size: int = 50,
        client: Any = None,
        tools: Any = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> list[Any]:
        """Run the FSM over a batch of requests concurrently using a thread pool."""
        from concurrent.futures import ThreadPoolExecutor

        results = []
        # Limit the number of concurrent threads to batch_size
        max_workers = min(batch_size, len(requests)) if requests else 1
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.run,
                    req,
                    client=client,
                    tools=tools,
                    until_end=until_end,
                    max_cycles=max_cycles,
                    **kwargs
                )
                for req in requests
            ]
            for future in futures:
                results.append(future.result())
        return results

    async def arun_batch(
        self,
        requests: list[Any],
        *,
        batch_size: int = 50,
        client: Any = None,
        tools: Any = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> list[Any]:
        """Async run the FSM over a batch of requests concurrently."""
        import asyncio

        async def _bounded_arun(semaphore: asyncio.Semaphore, req: Any):
            async with semaphore:
                return await self.arun(
                    req,
                    client=client,
                    tools=tools,
                    until_end=until_end,
                    max_cycles=max_cycles,
                    **kwargs
                )

        semaphore = asyncio.Semaphore(batch_size)
        tasks = [_bounded_arun(semaphore, req) for req in requests]
        return await asyncio.gather(*tasks)

    def _normalize_request(
        self,
        request: Any,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from pydantic import BaseModel

        from .models import RunRequest

        payload: dict[str, Any] = {}
        if request is None:
            payload = {"input": {}, "state": dict(state or {})}
        elif isinstance(request, RunRequest):
            payload = request.model_dump()
            if state is not None:
                payload["state"] = dict(state)
        elif isinstance(request, BaseModel):
            payload = {"input": request.model_dump(), "state": dict(state or {})}
        elif isinstance(request, Mapping):
            req_dict = dict(request)
            if "prior_executions" in req_dict or "current_state" in req_dict:
                payload = req_dict
                if "input" not in payload and "intent" in payload:
                    # Legacy resume payloads used intent= as free text.
                    legacy = payload.pop("intent")
                    payload["input"] = (
                        legacy if isinstance(legacy, dict) else {"text": str(legacy)}
                    )
                if state is not None:
                    payload["state"] = dict(state)
            elif "input" in req_dict or "state" in req_dict:
                payload = {
                    "input": dict(req_dict.get("input") or {}),
                    "state": dict(state if state is not None else req_dict.get("state") or {}),
                }
                for key in (
                    "current_state",
                    "prior_executions",
                    "request_id",
                    "metadata",
                    "history",
                ):
                    if key in req_dict:
                        payload[key] = req_dict[key]
            else:
                payload = {"input": req_dict, "state": dict(state or {})}
        else:
            raise TypeError(
                f"request must be RunRequest, BaseModel, mapping, or omitted; got {type(request)!r}"
            )

        # Workflows enter at the declared entry unless the caller resumes mid-path.
        payload.setdefault("current_state", self.entry_id)
        if not payload.get("current_state"):
            payload["current_state"] = self.entry_id
        payload.setdefault("prior_executions", [])
        if "input" not in payload:
            payload["input"] = {}
        elif not isinstance(payload["input"], dict):
            payload["input"] = {"text": str(payload["input"])}
        payload.setdefault("state", {})
        payload.pop("intent", None)
        return payload

    def _follow_terminal_edge(self, current: str, state: Mapping[str, Any]) -> str:
        """If the first matching deterministic exit is End, advance without another node run."""
        if current == END:
            return current
        matching = self.first_matching_deterministic(current, dict(state))
        if matching is not None and matching.target == END:
            return END
        return current

    def _run_loop(
        self,
        request: Any = None,
        *,
        client: Any = None,
        tools: Any = None,
        state: Mapping[str, Any] | None = None,
        until_end: bool = True,
        max_cycles: int = 32,
        **kwargs: Any,
    ) -> Any:
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")

        payload = self._normalize_request(request, state=state)
        manager = self._control_manager(client=client, tools=tools, **kwargs)

        if not until_end:
            return manager.run(payload)

        last = None
        prior: list[dict[str, Any]] = list(payload.get("prior_executions") or [])
        current = str(payload.get("current_state") or self.entry_id)
        initial_state = current
        snapshot = dict(payload.get("state") or {})
        run_input = dict(payload.get("input") or {})
        all_steps: list[Any] = []
        all_transitions: list[str] = []
        all_gates: list[Any] = []
        step_offset = 0

        for _ in range(max_cycles):
            cycle_request = {
                **payload,
                "input": run_input,
                "current_state": current,
                "state": snapshot,
                "prior_executions": prior,
            }
            last = manager.run(cycle_request)
            snapshot = dict(last.state)
            current = last.final_state
            for step in last.steps:
                all_steps.append(
                    step.model_copy(update={"step": step_offset + step.step})
                )
            step_offset += len(last.steps)
            all_transitions.extend(last.audit.committed_transitions)
            all_gates.extend(last.audit.gate_checks)
            prior = prior + [
                {
                    "node_id": item.node_id,
                    "status": item.status,
                    "output": item.output,
                    "state_updates": item.state_updates,
                }
                for step in last.steps
                for item in step.results
            ]
            if last.rejected:
                return last.model_copy(
                    update={
                        "steps": all_steps,
                        "audit": last.audit.model_copy(
                            update={
                                "initial_state": initial_state,
                                "steps": all_steps,
                                "committed_transitions": all_transitions,
                                "gate_checks": all_gates,
                            }
                        ),
                    }
                )
            previous = current
            current = self._follow_terminal_edge(current, snapshot)
            if current == END and previous != END:
                all_transitions.append(f"{previous}->{END}")
            if current == END:
                return last.model_copy(
                    update={
                        "final_state": END,
                        "completed": True,
                        "steps": all_steps,
                        "audit": last.audit.model_copy(
                            update={
                                "initial_state": initial_state,
                                "final_state": END,
                                "steps": all_steps,
                                "committed_transitions": all_transitions,
                                "gate_checks": all_gates,
                            }
                        ),
                    }
                )

        raise RuntimeError(
            f"FSM did not reach End within {max_cycles} cycles "
            f"(last state {current!r})"
        )

    def _control_manager(
        self,
        *,
        client: Any = None,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        from ..control.manager import ControlManager

        resolved_tools = (
            tools
            if tools is not None
            else getattr(self, "tool_registry", None)
        )
        resolved_client = (
            client
            if client is not None
            else getattr(self, "_remote_client", None)
        )
        return ControlManager(
            self,
            client=resolved_client,
            tools=resolved_tools,
            **kwargs,
        )


def _flood(origin: str, adjacency: dict[str, set[str]]) -> set[str]:
    seen = {origin}
    queue = deque([origin])
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency.get(vertex, ()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def _entry_id(item: FSMNode) -> str:
    """FSM state id to enter this authored unit."""
    return item.id


def _exit_id(item: FSMNode) -> str:
    """FSM state id that leaves this authored unit (CombineNode → ``{id}.Schema``)."""
    if isinstance(item, CombineNode):
        return item.schema_id
    return item.id


def Workflow(
    sequence: list[FSMNode] | tuple[FSMNode, ...],
    *,
    fallback: FSMNode | None = None,
    entry: Any = None,
) -> FSM:
    """Build a linear FSM from an ordered node list.

    Developers pass the happy-path sequence; ``Workflow`` sets ``entry`` to
    the first step, wires ``entry → … → End``, and the optional fallback.
    The entry node's ``input_schema`` is the workflow contract.

    ``CombineNode`` units expand as usual: the next step links from
    ``{id}.Schema``, not from the reasoning entry.
    """
    steps = list(sequence)
    if not steps:
        raise FSMValidationError(["Workflow sequence must contain at least one node"])
    if fallback is None:
        raise FSMValidationError(
            ["Workflow requires a fallback node (exactly one per FSM)"]
        )
    if isinstance(fallback, CombineNode):
        raise FSMValidationError(
            ["fallback cannot be a CombineNode; use SchemaNode(..., is_fallback=True)"]
        )
    if isinstance(fallback, Node) and not fallback.is_fallback:
        fallback = fallback.model_copy(update={"is_fallback": True})

    first = steps[0]
    edges: list[Edge] = []
    for index in range(len(steps) - 1):
        edges.append(
            edge_deterministic(_exit_id(steps[index]), _entry_id(steps[index + 1]))
        )
    edges.append(edge_deterministic(_exit_id(steps[-1]), END))
    if entry is None:
        first = steps[0]
        edges.append(edge_fallback(_entry_id(first), fallback.id))
        entry_target = first
    else:
        edges.append(edge_fallback(_entry_id(entry), fallback.id))
        entry_target = entry

    is_router = hasattr(entry_target, "routes") # Simple check for router
    return FSM(
        nodes=[*steps, fallback],
        routers=[entry_target] if is_router else [],
        entry=entry_target,
        edges=edges,
    )


# Backward-compatible aliases
Graph = FSM
GraphValidationError = FSMValidationError
