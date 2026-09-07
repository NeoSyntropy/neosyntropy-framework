"""Standalone callable, node and graph code extraction.

This module deliberately has no dependency on :mod:`neosyntropy.monitor`.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import sys
import textwrap
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from .bundles import (
    BUNDLE_MEDIA_TYPE,
    bundle_sha256,
    canonical_json_bytes,
    dependency_lock,
    gzip_json_bundle,
    runtime_compatibility,
)
from .schemas import CodeArtifactRef

if TYPE_CHECKING:
    from neosyntropy.core.node.base import Node
    from neosyntropy.tools.core.registry import ToolRegistry

_MAX_VFS_FILES = 200
_ROOT_MARKERS = {"pyproject.toml", "setup.py", "setup.cfg", ".git"}
_INSTALL_PARTS = {".venv", "venv", "site-packages", "dist-packages", "__pypackages__"}
_RUNTIME_PACKAGES = {"neosyntropy"}
_BUILTINS = frozenset(dir(__import__("builtins")))
try:
    _STDLIB = frozenset(sys.stdlib_module_names)  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover
    _STDLIB = frozenset()


class ToolVFS(TypedDict):
    entry_file: str
    vfs: dict[str, str]
    external_imports: list[str]
    extractable: bool


class NodeHandlerCode(TypedDict):
    node_id: str
    function_name: str
    function_module: str | None
    entry_file: str
    vfs: dict[str, str]
    external_imports: list[str]
    is_async: bool
    extractable: bool
    truncated: bool
    tool_names: list[str]
    tools_vfs: dict[str, ToolVFS]


class GraphCodeExtraction(TypedDict):
    code_artifacts: list[CodeArtifactRef]
    node_implementations: dict[str, dict[str, Any]]
    router_implementations: dict[str, dict[str, Any]]
    edge_guards: dict[str, str]
    bundles: dict[str, bytes]


class _ExtractionError(Exception):
    pass


def _find_project_root(file: Path) -> Path:
    current = file.resolve().parent
    package_boundary: Path | None = None
    for parent in (current, *current.parents):
        if any((parent / marker).exists() for marker in _ROOT_MARKERS):
            return parent
        if (parent / "__init__.py").exists():
            package_boundary = parent
        elif package_boundary is not None:
            return parent
    return package_boundary or current


def _module_candidates(name: str, root: Path) -> list[Path]:
    if not name:
        return []
    rel = Path(*name.split("."))
    candidates = (root / rel.with_suffix(".py"), root / rel / "__init__.py")
    return [item.resolve() for item in candidates if item.is_file()]


def _classify_import(module_name: str, base_dir: Path) -> str:
    root = module_name.split(".", 1)[0]
    if root in _STDLIB:
        return "stdlib"
    if root in _RUNTIME_PACKAGES:
        return "third_party"
    if _module_candidates(module_name, base_dir):
        return "local"
    try:
        spec = importlib.util.find_spec(root)
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.origin:
        return "third_party"
    origin = Path(spec.origin).resolve()
    try:
        relative = origin.relative_to(base_dir.resolve())
    except ValueError:
        return "third_party"
    return "third_party" if any(part in _INSTALL_PARTS for part in relative.parts) else "local"


def _resolve_import(node: ast.Import | ast.ImportFrom, caller: Path, root: Path) -> list[Path]:
    if isinstance(node, ast.Import):
        return [
            path
            for alias in node.names
            if _classify_import(alias.name, root) == "local"
            for path in _module_candidates(alias.name, root)
        ]
    module = node.module or ""
    if node.level:
        anchor = caller.parent
        for _ in range(node.level - 1):
            anchor = anchor.parent
        base = anchor
    else:
        if _classify_import(module, root) != "local":
            return []
        base = root
    paths = _module_candidates(module, base)
    for alias in node.names:
        if alias.name != "*":
            paths.extend(_module_candidates(f"{module}.{alias.name}".strip("."), base))
    return list(dict.fromkeys(paths))


def _loaded_names(node: ast.AST) -> tuple[set[str], dict[str, set[str]]]:
    local: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        local = {
            arg.arg
            for arg in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        }
        if node.args.vararg:
            local.add(node.args.vararg.arg)
        if node.args.kwarg:
            local.add(node.args.kwarg.arg)
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            local.add(child.id)
    loaded: set[str] = set()
    attrs: dict[str, set[str]] = {}
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id not in local:
                loaded.add(child.id)
        elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            if child.value.id not in local:
                attrs.setdefault(child.value.id, set()).add(child.attr)
    return loaded, attrs


class _Index:
    def __init__(self, path: Path, source: str, root: Path) -> None:
        self.path = path
        self.source = source
        self.tree = ast.parse(source, filename=str(path))
        self.defs: dict[str, ast.AST] = {}
        self.imports: dict[str, tuple[str, str | None, list[Path], ast.AST]] = {}
        self.stars: list[Path] = []
        for stmt in self.tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.defs[stmt.name] = stmt
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self.defs[target.id] = stmt
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                self.defs[stmt.target.id] = stmt
            elif isinstance(stmt, ast.Import):
                files = _resolve_import(stmt, path, root)
                for alias in stmt.names:
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    self.imports[bound] = (
                        _classify_import(alias.name, root),
                        None,
                        files,
                        stmt,
                    )
            elif isinstance(stmt, ast.ImportFrom):
                files = _resolve_import(stmt, path, root)
                kind = "local" if stmt.level else _classify_import(stmt.module or "", root)
                for alias in stmt.names:
                    if alias.name == "*":
                        self.stars.extend(files)
                    else:
                        self.imports[alias.asname or alias.name] = (
                            kind,
                            alias.name,
                            files,
                            stmt,
                        )
        # Tools and callbacks are commonly declared inside ``main``. Index
        # nested functions as fallback symbols so their source can still be
        # bundled independently from the authoring function.
        for nested in ast.walk(self.tree):
            if isinstance(nested, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.defs.setdefault(nested.name, nested)
            elif isinstance(nested, ast.Lambda):
                self.defs[_lambda_name(nested)] = nested


def _lambda_name(node: ast.Lambda) -> str:
    """Return a stable valid identifier for a source lambda."""
    identity = (
        f"{node.lineno}:{node.col_offset}:"
        f"{ast.dump(node, annotate_fields=True, include_attributes=False)}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"_neosyntropy_lambda_{digest}"


def _callable_symbol(fn: Callable[..., Any]) -> str:
    if getattr(fn, "__name__", "") != "<lambda>":
        return fn.__name__
    try:
        path = Path(inspect.getfile(fn)).resolve()
        parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        expression = next(
            node
            for node in ast.walk(parsed)
            if isinstance(node, ast.Lambda) and node.lineno == fn.__code__.co_firstlineno
        )
        return _lambda_name(expression)
    except (OSError, TypeError, SyntaxError, StopIteration):
        location = f"{fn.__code__.co_firstlineno}:{fn.__code__.co_name}"
        return f"_neosyntropy_lambda_{hashlib.sha256(location.encode()).hexdigest()[:16]}"


def _node_source(source: str, node: ast.AST, *, symbol: str | None = None) -> str:
    if isinstance(node, ast.Lambda):
        function = ast.FunctionDef(
            name=symbol or _lambda_name(node),
            args=node.args,
            body=[ast.Return(value=node.body)],
            decorator_list=[],
            returns=None,
            type_comment=None,
        )
        ast.fix_missing_locations(function)
        return ast.unparse(function).rstrip() + "\n"
    start = getattr(node, "lineno", 1)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        decorators = getattr(node, "decorator_list", ())
        if decorators:
            start = min(item.lineno for item in decorators)
    end = getattr(node, "end_lineno", start)
    chunk = "".join(source.splitlines(keepends=True)[start - 1 : end])
    return textwrap.dedent(chunk).rstrip() + "\n"


def _slice(index: _Index, symbols: set[str]) -> str:
    named_nodes = [(name, index.defs[name]) for name in symbols if name in index.defs]
    nodes = [node for _, node in named_nodes]
    used = set()
    for node in nodes:
        loaded, _ = _loaded_names(node)
        used.update(loaded)
    chunks: list[tuple[int, str]] = []
    seen: set[int] = set()
    for name in used:
        binding = index.imports.get(name)
        if binding is None or id(binding[3]) in seen:
            continue
        seen.add(id(binding[3]))
        stmt = binding[3]
        aliases = []
        for alias in stmt.names:  # type: ignore[attr-defined]
            bound = alias.asname or (
                alias.name.split(".", 1)[0] if isinstance(stmt, ast.Import) else alias.name
            )
            if bound in used or alias.name == "*":
                aliases.append(ast.alias(name=alias.name, asname=alias.asname))
        if aliases:
            clone: ast.AST = (
                ast.Import(names=aliases)
                if isinstance(stmt, ast.Import)
                else ast.ImportFrom(module=stmt.module, names=aliases, level=stmt.level)
            )
            chunks.append((stmt.lineno, ast.unparse(clone) + "\n"))
    chunks.extend(
        (
            getattr(node, "lineno", 0),
            _node_source(index.source, node, symbol=name),
        )
        for name, node in named_nodes
    )
    chunks.sort(key=lambda item: item[0])
    return "\n".join(text.rstrip() for _, text in chunks).rstrip() + "\n" if chunks else ""


def _build_vfs(
    fn: Callable[..., Any], base_dir: Path
) -> tuple[dict[str, str], list[str], bool, dict[str, str]]:
    try:
        entry = Path(inspect.getfile(fn)).resolve()
    except (OSError, TypeError) as exc:
        raise _ExtractionError(f"cannot locate source file: {exc}") from exc
    entry_symbol = _callable_symbol(fn)
    queue: deque[tuple[Path, str]] = deque([(entry, entry_symbol)])
    needed: dict[Path, set[str]] = {}
    indexes: dict[Path, _Index] = {}
    visited: set[tuple[Path, str]] = set()
    external: set[str] = set()
    truncated = False
    while queue:
        path, symbol = queue.popleft()
        if (path, symbol) in visited:
            continue
        if len({item[0] for item in visited} | {path}) > _MAX_VFS_FILES:
            truncated = True
            break
        visited.add((path, symbol))
        needed.setdefault(path, set()).add(symbol)
        try:
            index = indexes.setdefault(
                path,
                _Index(path, path.read_text(encoding="utf-8"), base_dir),
            )
        except (OSError, SyntaxError):
            continue
        if path == entry and getattr(fn, "__name__", "") != "<lambda>":
            first_line = getattr(getattr(fn, "__code__", None), "co_firstlineno", None)
            exact = next(
                (
                    candidate
                    for candidate in ast.walk(index.tree)
                    if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and candidate.name == symbol
                    and candidate.lineno == first_line
                ),
                None,
            )
            if exact is not None:
                index.defs[symbol] = exact
        node = index.defs.get(symbol)
        if node is None:
            continue
        loaded, attrs = _loaded_names(node)
        for name in loaded:
            if name in _BUILTINS:
                continue
            if name in index.defs and name != symbol:
                queue.append((path, name))
                continue
            binding = index.imports.get(name)
            if binding is None:
                for star in index.stars:
                    queue.append((star, name))
                continue
            kind, imported, files, _ = binding
            if kind != "local" or not files:
                stmt = binding[3]
                root = (
                    stmt.names[0].name.split(".", 1)[0]
                    if isinstance(stmt, ast.Import)
                    else (stmt.module or "").split(".", 1)[0]
                )
                if root:
                    external.add(root)
                continue
            targets = [imported] if imported else sorted(attrs.get(name) or {"*"})
            for file in files:
                if "*" in targets:
                    try:
                        target_index = indexes.setdefault(
                            file, _Index(file, file.read_text(encoding="utf-8"), base_dir)
                        )
                    except (OSError, SyntaxError):
                        continue
                    for target in target_index.defs:
                        queue.append((file, target))
                else:
                    for target in targets:
                        if target:
                            queue.append((file, target))
    vfs: dict[str, str] = {}
    real: dict[str, str] = {}
    for path in sorted(needed, key=str):
        try:
            key = "/app/" + path.relative_to(base_dir).as_posix()
        except ValueError:
            continue
        index = indexes.get(path)
        sliced = _slice(index, needed[path]) if index else ""
        if sliced.strip():
            vfs[key] = sliced
            real[key] = str(path)
    return vfs, sorted(external), truncated, real


def _unwrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    return inspect.unwrap(fn)


def _empty(node_id: str) -> NodeHandlerCode:
    return {
        "node_id": node_id,
        "function_name": "",
        "function_module": None,
        "entry_file": "",
        "vfs": {},
        "external_imports": [],
        "is_async": False,
        "extractable": False,
        "truncated": False,
        "tool_names": [],
        "tools_vfs": {},
    }


def _original_callable(fn: Callable[..., Any]) -> tuple[Callable[..., Any], dict[str, Any]]:
    unwrapped = _unwrap(fn)
    if unwrapped is not fn:
        return unwrapped, {
            "type": "wrapped",
            "kind": "wrapped",
            "callable": _provenance(fn),
        }
    try:
        nonlocals = inspect.getclosurevars(fn).nonlocals
    except (TypeError, ValueError):
        return fn, {}
    candidates = [(name, value) for name, value in nonlocals.items() if callable(value)]
    preferred = next((item for item in candidates if item[0] in {"fn", "func", "predicate"}), None)
    if preferred is not None:
        name, original = preferred
        bindings = {
            key: _binding_value(value)[0] for key, value in sorted(nonlocals.items()) if key != name
        }
        return _unwrap(original), {
            "type": "closure_adapter",
            "kind": "closure_adapter",
            "callable": _provenance(fn),
            "original_binding": name,
            "bindings": bindings,
        }
    return fn, {}


def _provenance(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        source_file: str | None = Path(inspect.getfile(fn)).name
    except (OSError, TypeError):
        source_file = None
    return {
        "module": getattr(fn, "__module__", None),
        "name": getattr(fn, "__name__", type(fn).__name__),
        "qualname": getattr(fn, "__qualname__", getattr(fn, "__name__", type(fn).__name__)),
        "is_async": inspect.iscoroutinefunction(fn),
        "source_file": source_file,
    }


def _binding_value(value: Any) -> tuple[Any, bool]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value, True
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}, True
    if isinstance(value, (list, tuple)):
        items = [_binding_value(item) for item in value]
        values = [item[0] for item in items]
        if isinstance(value, tuple):
            return {"type": "tuple", "items": values}, all(item[1] for item in items)
        return values, all(item[1] for item in items)
    if isinstance(value, (set, frozenset)):
        items = [_binding_value(item) for item in value]
        if not all(item[1] for item in items):
            return {"type": type(value).__name__}, False
        values = sorted((item[0] for item in items), key=canonical_json_bytes)
        return {"type": type(value).__name__, "items": values}, True
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        items = {key: _binding_value(value[key]) for key in sorted(value)}
        values = {key: item[0] for key, item in items.items()}
        return values, all(item[1] for item in items.values())
    if callable(value):
        return {"type": "callable", "provenance": _provenance(value)}, False
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}, False


def extract_callable_artifact(
    fn: Callable[..., Any],
    *,
    base_dir: Path | None = None,
) -> tuple[CodeArtifactRef, bytes | None]:
    """Extract one callable into a deterministic bundle and source-free ref."""
    fn = _unwrap(fn)
    provenance = _provenance(fn)
    callable_name = _callable_symbol(fn)
    if callable_name != provenance["name"]:
        provenance["synthetic_name"] = callable_name
    bindings: dict[str, Any] = {}
    supported = True
    try:
        closure = inspect.getclosurevars(fn).nonlocals
    except (TypeError, ValueError):
        closure = {}
    for name in sorted(closure):
        bindings[name], ok = _binding_value(closure[name])
        supported = supported and ok
    try:
        entry_path = Path(inspect.getfile(fn)).resolve()
        root = base_dir or _find_project_root(entry_path)
        vfs, external, truncated, _ = _build_vfs(fn, root)
        entry_file = "/app/" + entry_path.relative_to(root).as_posix()
        if not vfs or entry_file not in vfs:
            raise _ExtractionError("entry source was not found in the extracted closure")
        payload = {
            "schema_version": 1,
            "entry_file": entry_file,
            "vfs": vfs,
            "external_imports": external,
            "runtime_compat": runtime_compatibility(),
            "dependency_lock": dependency_lock(external),
            "callable": {
                "module": provenance["module"],
                "name": callable_name,
                "qualname": provenance["qualname"],
                "is_async": provenance["is_async"],
                "synthetic": callable_name != provenance["name"],
            },
            "bindings": bindings,
        }
        bundle = gzip_json_bundle(payload)
        digest = bundle_sha256(bundle)
        publishable = supported and not truncated
        ref: CodeArtifactRef = {
            "id": f"sha256:{digest}",
            "sha256": digest,
            "media_type": BUNDLE_MEDIA_TYPE,
            "size_bytes": len(bundle),
            "compressed_size": len(bundle),
            "uncompressed_size": len(canonical_json_bytes(payload)),
            "entry_file": entry_file,
            "external_imports": external,
            "extractable": publishable,
            "truncated": truncated,
            "required": True,
            "publishable": publishable,
            "provenance": provenance,
            "bindings": bindings,
            "runtime_compat": payload["runtime_compat"],
            "dependency_lock": payload["dependency_lock"],
        }
        if not supported:
            ref["reason"] = "unsupported_closure_binding"
            ref["recoverability"] = {
                "status": "blocked",
                "action": "replace unsupported closure values with JSON-compatible constants",
            }
        elif truncated:
            ref["reason"] = "vfs_file_limit_exceeded"
            ref["recoverability"] = {
                "status": "blocked",
                "action": "reduce the callable's transitive local dependency closure",
            }
        return ref, bundle
    except (_ExtractionError, OSError, TypeError, ValueError) as exc:
        identity = {
            "provenance": provenance,
            "bindings": bindings,
            "reason": str(exc),
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return {
            "id": f"unavailable:{digest}",
            "sha256": None,
            "media_type": BUNDLE_MEDIA_TYPE,
            "size_bytes": 0,
            "compressed_size": 0,
            "uncompressed_size": 0,
            "entry_file": "",
            "external_imports": [],
            "extractable": False,
            "truncated": False,
            "required": True,
            "publishable": False,
            "reason": str(exc),
            "provenance": provenance,
            "bindings": bindings,
            "runtime_compat": runtime_compatibility(),
            "dependency_lock": dependency_lock([]),
            "recoverability": {
                "status": "blocked",
                "action": "move the required callable to an inspectable Python source file",
            },
        }, None


def _tool_callable(registered: Any) -> tuple[Callable[..., Any] | None, dict[str, Any]]:
    handler = getattr(registered, "handler", None)
    if not callable(handler):
        return None, {}
    original, adapter = _original_callable(handler)
    return original, adapter


def extract_graph_code(
    graph: Any,
    registry: ToolRegistry | Mapping[str, Any] | None = None,
) -> GraphCodeExtraction:
    """Extract and deduplicate all graph-owned callable artifacts."""
    artifacts: dict[tuple[str, str], CodeArtifactRef] = {}
    bundles: dict[str, bytes] = {}
    node_links: dict[str, dict[str, Any]] = {}
    router_links: dict[str, dict[str, Any]] = {}
    edge_links: dict[str, str] = {}
    registered = getattr(registry, "tools", registry)
    registered = registered if isinstance(registered, Mapping) else {}

    def add(fn: Callable[..., Any], owner_role: str) -> str:
        ref, bundle = extract_callable_artifact(fn)
        usage = dict(ref)
        usage["owner_role"] = owner_role
        artifacts.setdefault((ref["id"], owner_role), usage)
        digest = ref.get("sha256")
        if bundle is not None and digest:
            bundles.setdefault(digest, bundle)
        return ref["id"]

    def implementation_ref(fn: Callable[..., Any], owner_role: str) -> dict[str, Any]:
        artifact_ref = add(fn, owner_role)
        artifact = artifacts[(artifact_ref, owner_role)]
        link: dict[str, Any] = {
            "artifact_ref": artifact_ref,
            "owner_role": owner_role,
            "required": True,
            "publishable": artifact["publishable"],
        }
        for field in ("reason", "recoverability"):
            if field in artifact:
                link[field] = artifact[field]
        return link

    def tools_for(names: Any) -> dict[str, Any]:
        links: dict[str, Any] = {}
        for name in sorted(names or ()):
            owner_role = "tool"
            item = registered.get(name)
            original, adapter = _tool_callable(item)
            if original is None:
                links[name] = {
                    "binding": name,
                    "owner_role": owner_role,
                    "required": True,
                    "publishable": False,
                    "reason": "tool_not_registered",
                    "recoverability": {
                        "status": "blocked",
                        "action": "register the declared tool before publishing the manifest",
                    },
                }
                continue
            links[name] = implementation_ref(original, owner_role)
            if adapter:
                links[name]["adapter"] = adapter
        return links

    for node_id, node in sorted(graph.nodes.items()):
        link: dict[str, Any] = {}
        handler = getattr(node, "handler", None)
        if callable(handler):
            original, adapter = _original_callable(handler)
            owner_role = (
                "schema_node"
                if getattr(node, "kind", None) == "schema"
                else "node_handler"
            )
            link.update(implementation_ref(original, owner_role))
            if adapter:
                link["adapter"] = adapter
        declared = tools_for(getattr(node, "tools", ()))
        if declared:
            link["tools"] = declared
        if link:
            node_links[node_id] = link

    for router_id, router in sorted(graph.routers.items()):
        link: dict[str, Any] = {}
        declared = tools_for(getattr(router, "tools", ()))
        if declared:
            link["tools"] = declared
        rules = getattr(router, "rules", None)
        if rules:
            link["predicates"] = [
                add(_unwrap(predicate), "router_predicate")
                for predicate, _ in rules
            ]
        if link:
            router_links[router_id] = link

    occurrences: dict[tuple[str, str, str], int] = {}
    for edge in graph.edges:
        key = (edge.source, edge.target, edge.kind)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        guard = getattr(edge, "guard", None)
        if callable(guard):
            original, _ = _original_callable(guard)
            stable_key = f"{edge.source}->{edge.target}:{edge.kind}:{occurrence}"
            edge_links[stable_key] = add(original, "edge_guard")

    return {
        "code_artifacts": [artifacts[key] for key in sorted(artifacts)],
        "node_implementations": node_links,
        "router_implementations": router_links,
        "edge_guards": edge_links,
        "bundles": bundles,
    }


def _tool_vfs(handler: Callable[..., Any], base: Path | None = None) -> ToolVFS:
    original, _ = _original_callable(handler)
    try:
        path = Path(inspect.getfile(original)).resolve()
        root = base or _find_project_root(path)
        vfs, external, truncated, _ = _build_vfs(original, root)
        return {
            "entry_file": "/app/" + path.relative_to(root).as_posix(),
            "vfs": vfs,
            "external_imports": external,
            "extractable": not truncated,
        }
    except (_ExtractionError, OSError, TypeError, ValueError):
        return {"entry_file": "", "vfs": {}, "external_imports": [], "extractable": False}


def extract_node_code(node: Node, registry: ToolRegistry | None = None) -> NodeHandlerCode:
    """Legacy inline-VFS API. New manifests use :func:`extract_graph_code`."""
    result = _empty(node.id)
    tool_names = list(node.tools or ())
    result["tool_names"] = tool_names
    registered = getattr(registry, "tools", {}) if registry is not None else {}
    result["tools_vfs"] = {
        name: _tool_vfs(registered[name].handler) for name in tool_names if name in registered
    }
    if not callable(node.handler):
        return result
    fn, _ = _original_callable(node.handler)
    result["function_name"] = getattr(fn, "__name__", "")
    result["function_module"] = getattr(fn, "__module__", None)
    result["is_async"] = inspect.iscoroutinefunction(fn)
    try:
        path = Path(inspect.getfile(fn)).resolve()
        root = _find_project_root(path)
        vfs, external, truncated, _ = _build_vfs(fn, root)
        result.update(
            entry_file="/app/" + path.relative_to(root).as_posix(),
            vfs=vfs,
            external_imports=external,
            extractable=bool(vfs) and not truncated,
            truncated=truncated,
        )
    except (_ExtractionError, OSError, TypeError, ValueError):
        pass
    return result


__all__ = [
    "CodeArtifactRef",
    "GraphCodeExtraction",
    "NodeHandlerCode",
    "ToolVFS",
    "_build_vfs",
    "_classify_import",
    "_find_project_root",
    "_unwrap",
    "extract_callable_artifact",
    "extract_graph_code",
    "extract_node_code",
]
