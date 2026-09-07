"""Tests for the Phase-1 VFS extraction pipeline.

Covers:
- VFS builder with function-closure extraction (used helpers/utils only)
- Circular-dependency cycle guard (no infinite loop, no crash)
- stdlib and third-party imports are skipped from the VFS but listed in external_imports
- Tool source extraction via extract_node_code
- Graph manifest round-trip (handler_code key present for handler nodes)
- extractable=False fallback for built-ins / C extensions / no-__file__ objects
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from neosyntropy.remote.extractor import (
    _build_vfs,
    _classify_import,
    _find_project_root,
    extract_node_code,
)

# ---------------------------------------------------------------------------
# Helpers for building an in-memory "project" inside tmp_path
# ---------------------------------------------------------------------------


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _make_node(handler, tools=(), node_id="TestNode") -> Any:
    """Return a minimal Node-like object (no Pydantic overhead in tests)."""
    n = MagicMock()
    n.id = node_id
    n.handler = handler
    n.tools = list(tools)
    n.kind = "handler"
    return n


# ---------------------------------------------------------------------------
# 1. _find_project_root
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    def test_detects_pyproject_toml(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[tool.something]")
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        module_file = pkg / "nodes.py"
        module_file.write_text("x = 1")
        assert _find_project_root(module_file) == tmp_path

    def test_detects_git(self, tmp_path: Path) -> None:
        (tmp_path / ".git").mkdir()
        src = tmp_path / "src" / "app.py"
        src.parent.mkdir()
        src.write_text("x = 1")
        assert _find_project_root(src) == tmp_path

    def test_fallback_to_package_boundary(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        module_file = pkg / "code.py"
        module_file.write_text("x = 1")
        root = _find_project_root(module_file)
        # Should be at or above tmp_path (no marker → fallback to package tree)
        assert root == tmp_path or root == pkg


# ---------------------------------------------------------------------------
# 2. _classify_import
# ---------------------------------------------------------------------------


class TestClassifyImport:
    def test_stdlib_json(self, tmp_path: Path) -> None:
        assert _classify_import("json", tmp_path) == "stdlib"

    def test_stdlib_os_path(self, tmp_path: Path) -> None:
        assert _classify_import("os.path", tmp_path) == "stdlib"

    def test_third_party_requests(self, tmp_path: Path) -> None:
        result = _classify_import("requests", tmp_path)
        assert result in ("third_party", "stdlib")

    def test_local_module(self, tmp_path: Path) -> None:
        # Create a real local module inside tmp_path
        local_pkg = tmp_path / "mylocal"
        local_pkg.mkdir()
        (local_pkg / "__init__.py").write_text("")
        # Add tmp_path to sys.path so find_spec can resolve it
        sys.path.insert(0, str(tmp_path))
        try:
            result = _classify_import("mylocal", tmp_path)
        finally:
            sys.path.remove(str(tmp_path))
        assert result == "local"

    def test_neosyntropy_runtime_never_local(self, tmp_path: Path) -> None:
        """The SDK is host runtime even when the project is this repository."""
        import neosyntropy

        framework_root = Path(neosyntropy.__file__).resolve().parents[1]
        assert _classify_import("neosyntropy", framework_root) == "third_party"
        assert _classify_import("neosyntropy.core.node", framework_root) == "third_party"
        assert _classify_import("neosyntropy", tmp_path) == "third_party"

    def test_venv_installed_pydantic_is_third_party(self, tmp_path: Path) -> None:
        """Packages in .venv/site-packages are not project source."""
        import neosyntropy

        framework_root = Path(neosyntropy.__file__).resolve().parents[1]
        assert _classify_import("pydantic", framework_root) == "third_party"


# ---------------------------------------------------------------------------
# 3. _build_vfs — real local imports
# ---------------------------------------------------------------------------


class TestBuildVfs:
    def test_single_file_no_local_deps(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        src = 'import json\n\ndef handler():\n    return json.dumps({"ok": True})\n'
        src_file = _write(tmp_path / "simple.py", src)

        mod = types.ModuleType("_test_simple")
        mod.__file__ = str(src_file)
        sys.modules["_test_simple"] = mod
        exec(compile(src, str(src_file), "exec"), mod.__dict__)
        fn = mod.handler
        fn.__module__ = "_test_simple"

        try:
            vfs, external, truncated, _real = _build_vfs(fn, tmp_path)
        finally:
            del sys.modules["_test_simple"]

        assert "/app/simple.py" in vfs
        assert "def handler" in vfs["/app/simple.py"]
        assert "json" in external
        assert not truncated

    def test_pydantic_used_by_schema_is_external(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        src = (
            "from pydantic import BaseModel, ConfigDict\n"
            "\n"
            "class Out(BaseModel):\n"
            "    model_config = ConfigDict(extra='forbid')\n"
            "    message: str\n"
            "\n"
            "def handler(ctx):\n"
            "    return Out(message='ok')\n"
        )
        handler_file = _write(tmp_path / "schema_node.py", src)

        mod = types.ModuleType("_test_pydantic_ext")
        mod.__file__ = str(handler_file)
        sys.modules["_test_pydantic_ext"] = mod
        exec(compile(src, str(handler_file), "exec"), mod.__dict__)
        fn = mod.handler
        fn.__module__ = "_test_pydantic_ext"

        try:
            vfs, external, truncated, _real = _build_vfs(fn, tmp_path)
        finally:
            del sys.modules["_test_pydantic_ext"]

        assert "from pydantic import BaseModel, ConfigDict" in vfs["/app/schema_node.py"]
        assert "pydantic" in external
        assert not truncated

    def test_same_file_includes_used_helper_not_unused_sibling(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        src = (
            "def is_positive(amount):\n"
            "    return amount > 0\n"
            "\n"
            "def unused_helper():\n"
            "    return 0\n"
            "\n"
            "def other_handler(ctx):\n"
            "    return unused_helper()\n"
            "\n"
            "def handler(ctx):\n"
            "    return is_positive(ctx['amount'])\n"
        )
        handler_file = _write(tmp_path / "nodes.py", src)

        mod = types.ModuleType("_test_same_file_helpers")
        mod.__file__ = str(handler_file)
        sys.modules["_test_same_file_helpers"] = mod
        exec(compile(src, str(handler_file), "exec"), mod.__dict__)
        fn = mod.handler
        fn.__module__ = "_test_same_file_helpers"

        try:
            vfs, _external, truncated, _real = _build_vfs(fn, tmp_path)
        finally:
            del sys.modules["_test_same_file_helpers"]

        sliced = vfs["/app/nodes.py"]
        assert "def handler" in sliced
        assert "def is_positive" in sliced
        assert "def unused_helper" not in sliced
        assert "def other_handler" not in sliced
        assert not truncated

    def test_used_utils_function_included_unused_omitted(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        utils_src = (
            "def format_amount(amount):\n"
            "    return f'{amount:.2f}'\n"
            "\n"
            "def unused_util():\n"
            "    return 'nope'\n"
        )
        handler_src = (
            "from utils import format_amount, unused_util\n"
            "\n"
            "def handler(ctx):\n"
            "    return format_amount(ctx['amount'])\n"
        )
        _write(tmp_path / "utils.py", utils_src)
        handler_file = _write(tmp_path / "handler.py", handler_src)

        sys.path.insert(0, str(tmp_path))
        mod = types.ModuleType("_test_used_utils")
        mod.__file__ = str(handler_file)
        sys.modules["_test_used_utils"] = mod
        exec(compile(handler_src, str(handler_file), "exec"), mod.__dict__)
        fn = mod.handler
        fn.__module__ = "_test_used_utils"

        try:
            vfs, _external, truncated, _real = _build_vfs(fn, tmp_path)
        finally:
            sys.path.remove(str(tmp_path))
            del sys.modules["_test_used_utils"]
            sys.modules.pop("utils", None)

        assert "/app/handler.py" in vfs
        assert "/app/utils.py" in vfs
        assert "def handler" in vfs["/app/handler.py"]
        assert "format_amount" in vfs["/app/handler.py"]
        assert "unused_util" not in vfs["/app/handler.py"]
        assert "def format_amount" in vfs["/app/utils.py"]
        assert "def unused_util" not in vfs["/app/utils.py"]
        assert not truncated

    def test_module_attr_utils_call(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        utils_src = (
            "def format_amount(amount):\n"
            "    return f'{amount:.2f}'\n"
            "\n"
            "def unused_util():\n"
            "    return 'nope'\n"
        )
        handler_src = (
            "import utils\n\ndef handler(ctx):\n    return utils.format_amount(ctx['amount'])\n"
        )
        _write(tmp_path / "utils.py", utils_src)
        handler_file = _write(tmp_path / "handler.py", handler_src)

        sys.path.insert(0, str(tmp_path))
        mod = types.ModuleType("_test_attr_utils")
        mod.__file__ = str(handler_file)
        sys.modules["_test_attr_utils"] = mod
        exec(compile(handler_src, str(handler_file), "exec"), mod.__dict__)
        fn = mod.handler
        fn.__module__ = "_test_attr_utils"

        try:
            vfs, _external, truncated, _real = _build_vfs(fn, tmp_path)
        finally:
            sys.path.remove(str(tmp_path))
            del sys.modules["_test_attr_utils"]
            sys.modules.pop("utils", None)

        assert "def format_amount" in vfs["/app/utils.py"]
        assert "def unused_util" not in vfs["/app/utils.py"]
        assert not truncated

    def test_neosyntropy_import_not_copied_into_vfs(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        src = "from neosyntropy import Client, FSM, node\n\ndef handler(ctx):\n    return ctx\n"
        handler_file = _write(tmp_path / "cookbook.py", src)

        mod = types.ModuleType("_test_ns_not_in_vfs")
        mod.__file__ = str(handler_file)
        sys.modules["_test_ns_not_in_vfs"] = mod
        code = compile("def handler(ctx):\n    return ctx\n", str(handler_file), "exec")
        exec(code, mod.__dict__)
        fn = mod.handler
        fn.__module__ = "_test_ns_not_in_vfs"

        try:
            vfs, external, truncated, _real = _build_vfs(fn, tmp_path)
        finally:
            del sys.modules["_test_ns_not_in_vfs"]

        assert list(vfs) == ["/app/cookbook.py"]
        assert "def handler" in vfs["/app/cookbook.py"]
        assert not any(key.startswith("/app/neosyntropy/") for key in vfs)
        assert "neosyntropy" not in vfs["/app/cookbook.py"]
        assert "Client" not in vfs["/app/cookbook.py"]
        assert not truncated

    def test_transitive_local_import(self, tmp_path: Path) -> None:
        """Handler imports utils.py which imports helpers.py — all land in VFS."""
        (tmp_path / "pyproject.toml").write_text("")
        helpers_src = "def helper(): return 42\n"
        utils_src = "from helpers import helper\nresult = helper()\n"
        handler_src = "from utils import result\n\ndef handler(): return result\n"

        _write(tmp_path / "helpers.py", helpers_src)
        _write(tmp_path / "utils.py", utils_src)
        handler_file = _write(tmp_path / "handler.py", handler_src)

        sys.path.insert(0, str(tmp_path))
        mod = types.ModuleType("_test_transitive_handler")
        mod.__file__ = str(handler_file)
        sys.modules["_test_transitive_handler"] = mod
        code = compile(handler_src, str(handler_file), "exec")
        exec(code, mod.__dict__)
        fn = mod.handler
        fn.__module__ = "_test_transitive_handler"

        try:
            vfs, external, truncated, _real = _build_vfs(fn, tmp_path)
        finally:
            sys.path.remove(str(tmp_path))
            del sys.modules["_test_transitive_handler"]

        assert "/app/handler.py" in vfs
        assert "/app/utils.py" in vfs
        assert "/app/helpers.py" in vfs
        assert not truncated

    def test_cycle_guard_no_infinite_loop(self, tmp_path: Path) -> None:
        """a.py.handler → b.py.ping → a.py.handler must terminate without error."""
        (tmp_path / "pyproject.toml").write_text("")
        a_src = "from b import ping\n\ndef handler():\n    return ping()\n"
        b_src = "from a import handler as h\n\ndef ping():\n    return h\n"
        _write(tmp_path / "a.py", a_src)
        _write(tmp_path / "b.py", b_src)

        sys.path.insert(0, str(tmp_path))
        mod = types.ModuleType("a")
        mod.__file__ = str(tmp_path / "a.py")
        sys.modules["a"] = mod
        code = compile("def handler():\n    return ping()\n", str(tmp_path / "a.py"), "exec")
        exec(code, mod.__dict__)
        fn = mod.handler
        fn.__module__ = "a"

        try:
            vfs, external, truncated, _real = _build_vfs(fn, tmp_path)
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("a", None)
            sys.modules.pop("b", None)

        assert "/app/a.py" in vfs
        assert "/app/b.py" in vfs
        assert "def handler" in vfs["/app/a.py"]
        assert "def ping" in vfs["/app/b.py"]
        assert not truncated

    def test_stdlib_not_in_vfs(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        src = "import os, json\ndef handler(): return os.getcwd()\n"
        mod_file = _write(tmp_path / "stdlib_test.py", src)

        mod = types.ModuleType("_test_stdlib")
        mod.__file__ = str(mod_file)
        sys.modules["_test_stdlib"] = mod
        exec(compile(src, str(mod_file), "exec"), mod.__dict__)
        fn = mod.handler
        fn.__module__ = "_test_stdlib"

        try:
            vfs, external, truncated, _real = _build_vfs(fn, tmp_path)
        finally:
            del sys.modules["_test_stdlib"]

        # stdlib must NOT appear as a VFS file
        vfs_keys_lower = {k.lower() for k in vfs}
        assert "/app/os.py" not in vfs_keys_lower
        assert "/app/json.py" not in vfs_keys_lower
        # But should appear in external
        assert "os" in external or "json" in external


# ---------------------------------------------------------------------------
# 4. extract_node_code — main API
# ---------------------------------------------------------------------------


class TestExtractNodeCode:
    def test_handler_none_returns_not_extractable(self) -> None:
        node = _make_node(None)
        result = extract_node_code(node)
        assert result["extractable"] is False
        assert result["vfs"] == {}

    def test_reasoning_node_tools_extracted_without_handler(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        tool_src = "def lookup(x): return x\n"
        _write(tmp_path / "tool_only.py", tool_src)

        t_mod = types.ModuleType("_test_tool_only")
        t_mod.__file__ = str(tmp_path / "tool_only.py")
        sys.modules["_test_tool_only"] = t_mod
        exec(compile(tool_src, str(tmp_path / "tool_only.py"), "exec"), t_mod.__dict__)
        t_fn = t_mod.lookup
        t_fn.__module__ = "_test_tool_only"

        registry = MagicMock()
        tool_stub = MagicMock()
        tool_stub.handler = t_fn
        registry.tools = {"lookup": tool_stub}

        try:
            node = _make_node(None, tools=["lookup"], node_id="Reason")
            result = extract_node_code(node, registry)
        finally:
            del sys.modules["_test_tool_only"]

        assert result["extractable"] is False
        assert result["vfs"] == {}
        assert "lookup" in result["tools_vfs"]
        assert "/app/tool_only.py" in result["tools_vfs"]["lookup"]["vfs"]

    def test_builtin_returns_not_extractable(self) -> None:
        node = _make_node(len)  # C built-in; inspect.getfile raises TypeError
        result = extract_node_code(node)
        assert result["extractable"] is False

    def test_real_handler_extractable(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        src = "import json\ndef my_handler(ctx): return json.dumps(ctx)\n"
        handler_file = _write(tmp_path / "real_handler.py", src)

        mod = types.ModuleType("_test_real_handler")
        mod.__file__ = str(handler_file)
        sys.modules["_test_real_handler"] = mod
        exec(compile(src, str(handler_file), "exec"), mod.__dict__)
        fn = mod.my_handler
        fn.__module__ = "_test_real_handler"

        try:
            node = _make_node(fn)
            result = extract_node_code(node)
        finally:
            del sys.modules["_test_real_handler"]

        assert result["extractable"] is True
        assert result["function_name"] == "my_handler"
        assert "/app/real_handler.py" in result["vfs"]
        assert result["entry_file"] == "/app/real_handler.py"
        assert "json" in result["external_imports"]

    def test_wrapped_handler_unwrapped(self, tmp_path: Path) -> None:
        """functools.wraps chain must be unwrapped to find the real source file."""
        import functools

        (tmp_path / "pyproject.toml").write_text("")
        src = "def inner(ctx): return ctx\n"
        handler_file = _write(tmp_path / "inner.py", src)

        mod = types.ModuleType("_test_wrapped")
        mod.__file__ = str(handler_file)
        sys.modules["_test_wrapped"] = mod
        exec(compile(src, str(handler_file), "exec"), mod.__dict__)
        inner_fn = mod.inner
        inner_fn.__module__ = "_test_wrapped"

        @functools.wraps(inner_fn)
        def outer(ctx):
            return inner_fn(ctx)

        outer.__wrapped__ = inner_fn

        try:
            node = _make_node(outer)
            result = extract_node_code(node)
        finally:
            del sys.modules["_test_wrapped"]

        assert result["extractable"] is True
        assert result["function_name"] == "inner"

    def test_tool_vfs_extracted(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("")
        handler_src = "def handler(ctx): pass\n"
        tool_src = "def tool_fn(x): return x\n"
        _write(tmp_path / "handler_m.py", handler_src)
        _write(tmp_path / "tool_m.py", tool_src)

        h_mod = types.ModuleType("_test_tool_handler")
        h_mod.__file__ = str(tmp_path / "handler_m.py")
        sys.modules["_test_tool_handler"] = h_mod
        exec(compile(handler_src, str(tmp_path / "handler_m.py"), "exec"), h_mod.__dict__)
        h_fn = h_mod.handler
        h_fn.__module__ = "_test_tool_handler"

        t_mod = types.ModuleType("_test_tool_fn")
        t_mod.__file__ = str(tmp_path / "tool_m.py")
        sys.modules["_test_tool_fn"] = t_mod
        exec(compile(tool_src, str(tmp_path / "tool_m.py"), "exec"), t_mod.__dict__)
        t_fn = t_mod.tool_fn
        t_fn.__module__ = "_test_tool_fn"

        # Build a minimal tool registry mock
        registry = MagicMock()
        tool_stub = MagicMock()
        tool_stub.handler = t_fn
        registry.tools = {"my_tool": tool_stub}

        try:
            node = _make_node(h_fn, tools=["my_tool"])
            result = extract_node_code(node, registry)
        finally:
            del sys.modules["_test_tool_handler"]
            del sys.modules["_test_tool_fn"]

        assert result["extractable"] is True
        assert "my_tool" in result["tools_vfs"]
        tool_vfs = result["tools_vfs"]["my_tool"]
        assert "/app/tool_m.py" in tool_vfs["vfs"]


# ---------------------------------------------------------------------------
# 5. Graph manifest round-trip
# ---------------------------------------------------------------------------


def _fsm_with_nodes(*nodes: Any) -> MagicMock:
    fsm_mock = MagicMock()
    fsm_mock.entry_id = nodes[0].id
    fsm_mock.input_schema = {"type": "object"}
    fsm_mock.nodes = {n.id: n for n in nodes}
    fsm_mock.routers = {}
    fsm_mock.router_ids = []
    fsm_mock.edges = []
    fsm_mock.groups = {}
    fsm_mock.allow_unlisted_transitions = False
    fsm_mock.function_source = None
    fsm_mock.decorator = None
    return fsm_mock


class TestManifestRoundTrip:
    def test_handler_code_is_replaced_by_source_free_artifact_ref(self, tmp_path: Path) -> None:
        from neosyntropy.remote import graph_manifest_with_bundles, node_manifest_with_bundles

        (tmp_path / "pyproject.toml").write_text("")
        src = "def process(ctx): return ctx\n"
        f = _write(tmp_path / "proc.py", src)

        mod = types.ModuleType("_test_manifest_node")
        mod.__file__ = str(f)
        sys.modules["_test_manifest_node"] = mod
        exec(compile(src, str(f), "exec"), mod.__dict__)
        fn = mod.process
        fn.__module__ = "_test_manifest_node"

        from neosyntropy.core.node.base import Node

        node_obj = Node(
            id="ProcessData",
            kind="handler",
            mode="schema_extraction",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            handler=fn,
        )

        try:
            owned = node_manifest_with_bundles(node_obj)
            packaged = graph_manifest_with_bundles(_fsm_with_nodes(node_obj))
        finally:
            del sys.modules["_test_manifest_node"]

        manifest = packaged.manifest
        assert "handler_code" not in manifest
        graph_node = next(n for n in manifest["nodes"] if n["id"] == "ProcessData")
        assert "handler_code" not in graph_node
        assert "vfs" not in json.dumps(manifest)
        artifact_id = graph_node["implementation"]["artifact_ref"]
        assert artifact_id == owned.manifest["implementation"]["artifact_ref"]
        assert packaged.bundle(artifact_id) == owned.bundle(artifact_id)
        assert manifest["structure_hash"]

    def test_schema_node_without_tools_has_no_handler_code(self) -> None:
        from neosyntropy.core.node.base import Node
        from neosyntropy.monitor.graph.manifest import graph_manifest

        schema_node = Node(
            id="InferResult",
            kind="schema",
            mode="schema_extraction",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        manifest = graph_manifest(_fsm_with_nodes(schema_node))
        assert "handler_code" not in manifest
        graph_node = next(n for n in manifest["nodes"] if n["id"] == "InferResult")
        assert "handler_code" not in graph_node

    def test_reasoning_node_sends_tool_artifact_ref(self, tmp_path: Path) -> None:
        from neosyntropy.core.node.reasoning import ReasoningNode
        from neosyntropy.remote import node_manifest_with_bundles

        (tmp_path / "pyproject.toml").write_text("")
        tool_src = "def lookup_order(order_id): return order_id\n"
        _write(tmp_path / "lookup.py", tool_src)

        t_mod = types.ModuleType("_test_reason_tool")
        t_mod.__file__ = str(tmp_path / "lookup.py")
        sys.modules["_test_reason_tool"] = t_mod
        exec(compile(tool_src, str(tmp_path / "lookup.py"), "exec"), t_mod.__dict__)
        t_fn = t_mod.lookup_order
        t_fn.__module__ = "_test_reason_tool"

        registry = MagicMock()
        tool_stub = MagicMock()
        tool_stub.handler = t_fn
        registry.tools = {"lookup_order": tool_stub}

        try:
            node = ReasoningNode(
                id="DecideRefund",
                input_schema={"type": "object"},
                tools=["lookup_order"],
                prompt="Decide whether to refund.",
            )
            packaged = node_manifest_with_bundles(node, tool_registry=registry)
        finally:
            del sys.modules["_test_reason_tool"]

        payload = packaged.manifest
        assert node.handler is None
        assert "handler_code" not in payload
        tool = payload["implementation"]["tools"]["lookup_order"]
        assert tool["artifact_ref"].startswith("sha256:")
        assert packaged.bundle(tool["artifact_ref"])
        assert "def lookup_order" not in json.dumps(payload)


# ---------------------------------------------------------------------------
# 6. _control_api_graph — handler_code gate
# ---------------------------------------------------------------------------


class TestControlApiGraphGate:
    def _base_manifest(self) -> dict:
        return {
            "schema_version": 1,
            "entry": "A",
            "input_schema": {"type": "object"},
            "nodes": [
                {
                    "id": "A",
                    "kind": "schema",
                    "is_fallback": False,
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "object"},
                    "handler_code": {"vfs": {"/app/a.py": "x=1"}, "extractable": True},
                }
            ],
            "edges": [],
            "groups": [],
            "routers": [],
        }

    def test_handler_code_excluded_by_default(self) -> None:
        from neosyntropy.backend import _control_api_graph

        wire = _control_api_graph(self._base_manifest())
        assert "handler_code" not in wire
        assert "handler_code" not in wire["nodes"][0]

    def test_handler_code_assembled_from_nodes_when_flag_set(self) -> None:
        from neosyntropy.backend import _control_api_graph

        wire = _control_api_graph(self._base_manifest(), include_handler_code=True)
        assert "handler_code" in wire
        assert wire["handler_code"]["A"]["extractable"] is True


def _make_handler_mod(tmp_path: Path, src: str, mod_name: str, filename: str):
    f = _write(tmp_path / filename, src)
    mod = types.ModuleType(mod_name)
    mod.__file__ = str(f)
    sys.modules[mod_name] = mod
    exec(compile(src, str(f), "exec"), mod.__dict__)
    fn = mod.__dict__[src.split("def ")[1].split("(")[0]]
    fn.__module__ = mod_name
    return fn

class TestContentAddressedBundles:
    def test_gzip_bundle_and_digest_are_deterministic(self, tmp_path: Path) -> None:
        import gzip

        from neosyntropy.remote import bundle_sha256, extract_callable_artifact

        (tmp_path / "pyproject.toml").write_text("")
        src = "def handler(ctx):\n    return ctx\n"
        fn = _make_handler_mod(tmp_path, src, "_bundle_determinism", "bundle.py")
        try:
            first_ref, first = extract_callable_artifact(fn)
            second_ref, second = extract_callable_artifact(fn)
        finally:
            del sys.modules["_bundle_determinism"]

        assert first is not None
        assert first == second
        assert first_ref == second_ref
        assert first_ref["sha256"] == bundle_sha256(first)
        decoded = json.loads(gzip.decompress(first))
        assert decoded["vfs"]["/app/bundle.py"] == src
        assert "vfs" not in first_ref
        assert "def handler" not in json.dumps(first_ref)

    def test_manifest_deduplicates_handler_and_tool_bundle(self, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from neosyntropy.remote import graph_manifest_with_bundles

        (tmp_path / "pyproject.toml").write_text("")
        src = "def shared(ctx):\n    return ctx\n"
        fn = _make_handler_mod(tmp_path, src, "_bundle_dedupe", "shared.py")
        from neosyntropy.core.node.base import Node

        node = Node(
            id="Shared",
            handler=fn,
            tools=("shared",),
            mode="reasoning",
            kind="handler",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        registry = MagicMock()
        registered = SimpleNamespace(
            name="shared",
            description="",
            handler=fn,
            json_schema={},
            return_schema=None,
        )
        registry.tools = {"shared": registered}
        graph = _fsm_with_nodes(node)
        try:
            packaged = graph_manifest_with_bundles(graph, registry)
        finally:
            del sys.modules["_bundle_dedupe"]

        node_entry = packaged.manifest["nodes"][0]
        handler_ref = node_entry["implementation"]["artifact_ref"]
        tool_ref = node_entry["implementation"]["tools"]["shared"]["artifact_ref"]
        assert handler_ref == tool_ref
        assert len(packaged.manifest["code_artifacts"]) == 2
        assert {item["owner_role"] for item in packaged.manifest["code_artifacts"]} == {
            "node_handler",
            "tool",
        }
        assert len(packaged.bundles) == 1
        assert "def shared" not in json.dumps(packaged.manifest)

    def test_deterministic_edge_guard_has_stable_ref(self, tmp_path: Path) -> None:
        from neosyntropy.core.edge import edge_deterministic
        from neosyntropy.core.node.base import Node
        from neosyntropy.remote import graph_manifest_with_bundles

        (tmp_path / "pyproject.toml").write_text("")
        src = "def allows(state):\n    return state.get('ok', False)\n"
        guard = _make_handler_mod(tmp_path, src, "_bundle_guard", "guard.py")
        node = Node(
            id="Guarded",
            handler=guard,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        graph = _fsm_with_nodes(node)
        graph.edges = [edge_deterministic("Guarded", "End", guard=guard)]
        try:
            first = graph_manifest_with_bundles(graph)
            second = graph_manifest_with_bundles(graph)
        finally:
            del sys.modules["_bundle_guard"]

        edge = first.manifest["edges"][0]
        assert edge["guard_ref"].startswith("sha256:")
        assert first.manifest["structure_hash"] == second.manifest["structure_hash"]
        assert first.bundles == second.bundles

    def test_functional_node_preserves_original_and_adapter(self, tmp_path: Path) -> None:
        from neosyntropy.remote import graph_manifest_with_bundles

        (tmp_path / "pyproject.toml").write_text("")
        src = (
            "from neosyntropy.core.validation import functional_validation_node\n"
            "\n"
            "@functional_validation_node(id='Check', output_key='accepted')\n"
            "def check(ctx):\n"
            "    return bool(ctx.state.get('accepted'))\n"
        )
        path = _write(tmp_path / "functional.py", src)
        module = types.ModuleType("_functional_artifact")
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(src, str(path), "exec"), module.__dict__)
        graph = _fsm_with_nodes(module.check)
        try:
            packaged = graph_manifest_with_bundles(graph)
        finally:
            del sys.modules[module.__name__]

        implementation = packaged.manifest["nodes"][0]["implementation"]
        assert implementation["artifact_ref"].startswith("sha256:")
        assert implementation["adapter"]["kind"] == "closure_adapter"
        assert implementation["adapter"]["bindings"]["output_key"] == "accepted"
        artifact = next(
            item
            for item in packaged.manifest["code_artifacts"]
            if item["id"] == implementation["artifact_ref"]
        )
        assert artifact["provenance"]["name"] == "check"


class TestManifestV3:
    def test_schema_node_func_provider_and_runtime_metadata(self, tmp_path: Path) -> None:
        from neosyntropy.core.node import SchemaNode
        from neosyntropy.remote import node_manifest_with_bundles

        (tmp_path / "pyproject.toml").write_text("")
        fn = _make_handler_mod(
            tmp_path,
            "def format_result(value: str) -> str:\n    return value.upper()\n",
            "_schema_func",
            "schema_func.py",
        )
        try:
            node = SchemaNode(
                "Format",
                input_schema={"type": "object"},
                prompt="Extract value.",
                func=fn,
                provider="custom/provider",
                prerequisites=("Prepare",),
            )
            packaged = node_manifest_with_bundles(node)
        finally:
            del sys.modules["_schema_func"]

        manifest = packaged.manifest
        assert manifest["schema_version"] == 3
        assert manifest["provider"] == "custom/provider"
        assert manifest["prerequisites"] == ["Prepare"]
        assert manifest["implementation_ref"].startswith("sha256:")
        assert manifest["runtime_compat"]["policy"] == "compatible-runtime"
        assert manifest["dependency_lock"]["policy"] == "compatible-runtime"
        assert manifest["revision"] != manifest["structure_hash"]

    def test_nested_and_lambda_router_rules_get_synthetic_entries(
        self, tmp_path: Path
    ) -> None:
        import gzip

        from neosyntropy.core.node.base import Node
        from neosyntropy.core.routing import DeterministicRouter
        from neosyntropy.remote import graph_manifest_with_bundles

        (tmp_path / "pyproject.toml").write_text("")
        source = (
            "def make_guard(expected):\n"
            "    def matches(ctx):\n"
            "        return ctx.state.get('value') == expected\n"
            "    return matches\n"
            "named_guard = make_guard('yes')\n"
            "lambda_guard = lambda ctx: bool(ctx.state.get('fallback'))\n"
        )
        path = _write(tmp_path / "router_rules.py", source)
        module = types.ModuleType("_router_rules")
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(source, str(path), "exec"), module.__dict__)
        target = Node(
            id="Target",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        router = DeterministicRouter(
            id="Choose",
            input_schema={"type": "object"},
            rules=[
                (module.named_guard, target),
                (module.lambda_guard, target),
            ],
        )
        graph = _fsm_with_nodes(target)
        graph.routers = {router.id: router}
        graph.router_ids = {router.id}
        graph.edges = router.compile()
        try:
            packaged = graph_manifest_with_bundles(graph)
        finally:
            del sys.modules[module.__name__]

        detail = packaged.manifest["routers_detail"][0]
        assert detail["type"] == "deterministic"
        assert [rule["target"] for rule in detail["rules"]] == ["Target", "Target"]
        assert all(rule["predicate_ref"].startswith("sha256:") for rule in detail["rules"])
        lambda_ref = next(
            artifact
            for artifact in packaged.manifest["code_artifacts"]
            if artifact["provenance"]["name"] == "<lambda>"
        )
        payload = json.loads(gzip.decompress(packaged.bundle(lambda_ref["id"])))
        assert payload["callable"]["synthetic"] is True
        assert payload["callable"]["name"].startswith("_neosyntropy_lambda_")
        assert f"def {payload['callable']['name']}" in payload["vfs"]["/app/router_rules.py"]
        named_ref = next(
            artifact
            for artifact in packaged.manifest["code_artifacts"]
            if artifact["provenance"]["name"] == "matches"
        )
        assert named_ref["bindings"]["expected"] == "yes"

    def test_semantic_router_and_group_metadata_are_explicit(self) -> None:
        from neosyntropy.core.group import Group
        from neosyntropy.core.node.base import Node
        from neosyntropy.core.routing import SemanticRouter
        from neosyntropy.monitor.graph.manifest import graph_manifest

        target = Node(
            id="Target",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        group = Group(
            "Work",
            description="Work group",
            metadata={"owner": "ops"},
            entry=target,
            nodes=[target],
            namespace=False,
            parent="Root",
        )
        router = SemanticRouter(
            id="Intent",
            input_schema={"type": "object"},
            routes={"work": group},
            fallback_node=target,
            reasoning="high",
            category="operations",
            provider="custom/router",
        )
        graph = _fsm_with_nodes(target)
        graph.groups = {group.name: group}
        graph.routers = {router.id: router}
        graph.router_ids = {router.router_state_id}
        manifest = graph_manifest(graph)

        assert manifest["allow_unlisted_transitions"] is False
        assert manifest["decorator"] is None
        assert manifest["groups"][0] == {
            "name": "Work",
            "description": "Work group",
            "metadata": {"owner": "ops"},
            "entry": "Target",
            "parent": "Root",
            "namespace": False,
        }
        detail = manifest["routers_detail"][0]
        assert detail["state_id"] == "Intent.Schema"
        assert detail["reasoning"] == "high"
        assert detail["category"] == "operations"
        assert detail["provider"] == "custom/router"
        assert detail["routes"]["work"] == {
            "target": "Work",
            "target_kind": "group",
        }

    def test_required_unextractable_handler_is_not_publishable(self) -> None:
        from neosyntropy import FSM, SchemaNode, edge_deterministic, edge_fallback
        from neosyntropy.core.node.base import Node
        from neosyntropy.remote import graph_manifest_with_bundles, node_manifest_with_bundles

        node = Node(
            id="Builtin",
            handler=len,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        manifest = node_manifest_with_bundles(node).manifest
        implementation = manifest["implementation"]
        artifact = manifest["code_artifacts"][0]
        assert implementation["required"] is True
        assert implementation["publishable"] is False
        assert artifact["extractable"] is False
        assert artifact["recoverability"]["status"] == "blocked"

        fallback = SchemaNode(
            id="Fallback",
            prompt="Fallback.",
            is_fallback=True,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        graph = FSM(
            entry=node,
            nodes=[node, fallback],
            edges=[
                edge_deterministic("Builtin", "End"),
                edge_fallback("Builtin", "Fallback"),
            ],
        )
        graph_payload = graph_manifest_with_bundles(graph).manifest
        assert graph_payload["recoverable"] is False
        assert graph_payload["recovery_issues"]
