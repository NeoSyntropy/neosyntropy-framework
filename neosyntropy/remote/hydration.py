"""Verification and loading for remote content-addressed code bundles."""

from __future__ import annotations

import ast
import gzip
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any


class CodeBundleError(ValueError):
    """A remote code bundle failed integrity or schema validation."""


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def _matches_specifier(version: str, specifier: str) -> bool:
    """Small PEP 440 subset used without adding a runtime dependency."""
    current = _version_tuple(version)
    if not current:
        return False
    for clause in (item.strip() for item in specifier.split(",")):
        if not clause:
            continue
        match = re.fullmatch(r"(~=|==|!=|>=|<=|>|<)?\s*(\d+(?:\.\d+)*)", clause)
        if match is None:
            raise CodeBundleError(f"unsupported runtime version constraint {clause!r}")
        operator = match.group(1) or "=="
        wanted = _version_tuple(match.group(2))
        width = max(len(current), len(wanted))
        left = current + (0,) * (width - len(current))
        right = wanted + (0,) * (width - len(wanted))
        if operator == "~=":
            upper = (
                (wanted[0] + 1,)
                if len(wanted) <= 2
                else (*wanted[:-2], wanted[-2] + 1)
            )
            upper = upper + (0,) * (width - len(upper))
            ok = left >= right and left < upper
        else:
            ok = {
                "==": left == right,
                "!=": left != right,
                ">=": left >= right,
                "<=": left <= right,
                ">": left > right,
                "<": left < right,
            }[operator]
        if not ok:
            return False
    return True


def _constraint(raw: Any) -> str | None:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, Mapping):
        for key in ("specifier", "requires", "version", "constraint"):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        minimum = raw.get("min") or raw.get("minimum")
        maximum = raw.get("max") or raw.get("maximum")
        clauses = []
        if minimum:
            clauses.append(f">={minimum}")
        if maximum:
            clauses.append(f"<={maximum}")
        return ",".join(clauses) or None
    return None


def validate_manifest_compatibility(manifest: Mapping[str, Any]) -> None:
    """Fail before executing code when a v3 manifest cannot run locally."""
    schema_version = manifest.get("schema_version", 1)
    if not isinstance(schema_version, int) or schema_version not in {1, 2, 3}:
        raise CodeBundleError(f"unsupported manifest schema version {schema_version!r}")

    compat = manifest.get("runtime_compat")
    if compat is not None and not isinstance(compat, Mapping):
        raise CodeBundleError("manifest runtime_compat must be an object")
    compat = compat if isinstance(compat, Mapping) else {}
    python_raw = compat.get("python")
    if isinstance(python_raw, Mapping):
        implementation = python_raw.get("implementation")
        running_implementation = platform.python_implementation().lower()
        if (
            isinstance(implementation, str)
            and implementation.strip()
            and implementation.strip().lower() != running_implementation
        ):
            raise CodeBundleError(
                f"manifest requires Python implementation {implementation!r}; "
                f"running {running_implementation!r}"
            )
    python_spec = _constraint(
        python_raw
        or compat.get("python_version")
        or compat.get("requires_python")
    )
    python_version = ".".join(str(item) for item in sys.version_info[:3])
    if python_spec and not _matches_specifier(python_version, python_spec):
        raise CodeBundleError(
            f"manifest requires Python {python_spec}; running {python_version}"
        )

    framework_spec = _constraint(
        compat.get("neosyntropy")
        or compat.get("framework")
        or compat.get("framework_version")
    )
    if framework_spec:
        try:
            framework_version = importlib.metadata.version("neosyntropy")
        except importlib.metadata.PackageNotFoundError:
            framework_version = "0.1.0"
        if not _matches_specifier(framework_version, framework_spec):
            raise CodeBundleError(
                f"manifest requires neosyntropy {framework_spec}; "
                f"running {framework_version}"
            )

    lock = manifest.get("dependency_lock")
    if lock is None:
        return
    if isinstance(lock, Mapping):
        raw_packages = lock.get("packages", lock.get("dependencies", lock))
    else:
        raw_packages = lock
    packages: list[tuple[str, str | None]] = []
    if isinstance(raw_packages, Mapping):
        packages = [
            (str(name), _constraint(requirement))
            for name, requirement in raw_packages.items()
            if name not in {"format", "version", "policy"}
        ]
    elif isinstance(raw_packages, list):
        for item in raw_packages:
            if isinstance(item, str):
                match = re.match(r"([A-Za-z0-9_.-]+)(.*)", item)
                if match:
                    packages.append((match.group(1), match.group(2).strip() or None))
            elif isinstance(item, Mapping) and item.get("name"):
                packages.append(
                    (
                        str(item["name"]),
                        _constraint(
                            item.get("specifier")
                            or item.get("version")
                            or item.get("requires")
                        ),
                    )
                )
    else:
        raise CodeBundleError("manifest dependency_lock must be an object or list")

    for package, specifier in packages:
        if package.lower().replace("_", "-") == "neosyntropy":
            continue
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise CodeBundleError(
                f"manifest dependency {package!r} is not installed"
            ) from exc
        if specifier and not _matches_specifier(installed, specifier):
            raise CodeBundleError(
                f"manifest dependency {package!r} requires {specifier}; "
                f"running {installed}"
            )


def decode_code_bundle(
    bundle: bytes,
    *,
    expected_sha256: str,
    max_compressed_bytes: int = 10 * 1024 * 1024,
    max_uncompressed_bytes: int = 100 * 1024 * 1024,
) -> dict[str, Any]:
    """Verify, decompress, and parse one code bundle."""
    if len(bundle) > max_compressed_bytes:
        raise CodeBundleError("compressed code bundle is too large")
    actual = hashlib.sha256(bundle).hexdigest()
    if actual != expected_sha256:
        raise CodeBundleError(
            f"code bundle checksum mismatch: expected {expected_sha256}, got {actual}"
        )
    try:
        decoded = gzip.decompress(bundle)
    except (OSError, EOFError) as exc:
        raise CodeBundleError("code bundle is not valid gzip") from exc
    if len(decoded) > max_uncompressed_bytes:
        raise CodeBundleError("uncompressed code bundle is too large")
    try:
        payload = json.loads(decoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodeBundleError("code bundle is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise CodeBundleError("unsupported code bundle schema")
    if not isinstance(payload.get("vfs"), dict) or not payload["vfs"]:
        raise CodeBundleError("code bundle has no VFS files")
    return payload


class _EntrypointSanitizer(ast.NodeTransformer):
    def __init__(self, function_name: str) -> None:
        self.function_name = function_name
        self.found = False

    def _sanitize(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        if node.name != self.function_name or self.found:
            return self.generic_visit(node)
        self.found = True
        node.decorator_list = []
        node.returns = None
        for arg in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            arg.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        return node

    visit_FunctionDef = _sanitize
    visit_AsyncFunctionDef = _sanitize


def _sanitized_entry_source(source: str, function_name: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise CodeBundleError(f"entry source is invalid Python: {exc}") from exc
    sanitizer = _EntrypointSanitizer(function_name)
    tree = sanitizer.visit(tree)
    ast.fix_missing_locations(tree)
    if not sanitizer.found:
        raise CodeBundleError(f"entry function {function_name!r} was not found")
    return ast.unparse(tree) + "\n"


def _safe_relative(raw_path: str) -> PurePosixPath:
    relative = PurePosixPath(raw_path.removeprefix("/app/"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise CodeBundleError("code bundle VFS path is unsafe")
    return relative


def _binding_value(raw: Any, callables: Mapping[str, Any]) -> Any:
    if isinstance(raw, list):
        return [_binding_value(item, callables) for item in raw]
    if not isinstance(raw, Mapping):
        return raw
    kind = raw.get("kind") or raw.get("type")
    if kind in {"literal", "constant", "json"} and "value" in raw:
        return _binding_value(raw["value"], callables)
    if kind == "bytes":
        try:
            return bytes.fromhex(str(raw.get("hex") or ""))
        except ValueError as exc:
            raise CodeBundleError("code bundle contains an invalid bytes binding") from exc
    if kind in {"tuple", "set", "frozenset", "list"}:
        values = [_binding_value(item, callables) for item in raw.get("items", ())]
        return {
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
            "list": list,
        }[str(kind)](values)
    if kind in {"dict", "mapping"}:
        values = raw.get("items", raw.get("value", {}))
        if isinstance(values, Mapping):
            return {str(key): _binding_value(value, callables) for key, value in values.items()}
    if kind in {"callable", "callable_ref", "artifact"} or "artifact_ref" in raw:
        reference = raw.get("artifact_ref") or raw.get("ref")
        resolved = callables.get(str(reference))
        if not callable(resolved):
            raise CodeBundleError(f"callable binding {reference!r} is unavailable")
        return resolved
    # v1 bundles stored JSON-compatible closure bindings directly.
    return {str(key): _binding_value(value, callables) for key, value in raw.items()}


def load_bundle_callable(
    payload: Mapping[str, Any],
    *,
    artifact_id: str,
    callables: Mapping[str, Any] | None = None,
) -> tuple[Callable[..., Any], str]:
    """Materialize a VFS and return its undecorated entry callable."""
    vfs = payload.get("vfs")
    callable_meta = payload.get("callable")
    if not isinstance(vfs, Mapping) or not isinstance(callable_meta, Mapping):
        raise CodeBundleError("code bundle is missing callable metadata")
    function_name = str(
        callable_meta.get("entry_name")
        or callable_meta.get("synthetic_name")
        or callable_meta.get("name")
        or ""
    ).strip()
    entry_file = str(payload.get("entry_file") or "").strip()
    source = vfs.get(entry_file)
    if not function_name or not isinstance(source, str):
        raise CodeBundleError("code bundle entrypoint is incomplete")

    root = Path(tempfile.mkdtemp(prefix="neosyntropy-code-"))
    for raw_path, raw_source in vfs.items():
        if not isinstance(raw_path, str) or not isinstance(raw_source, str):
            raise CodeBundleError("code bundle VFS contains invalid entries")
        relative = _safe_relative(raw_path)
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(raw_source, encoding="utf-8")

    entry_relative = _safe_relative(entry_file)
    package = ".".join(entry_relative.parent.parts) or None
    resolved_callables = callables or {}
    bindings = payload.get("bindings")
    bindings = bindings if isinstance(bindings, Mapping) else {}
    namespace: dict[str, Any] = {
        "__file__": str(root.joinpath(*entry_relative.parts)),
        "__name__": f"_neosyntropy_remote_{artifact_id.replace(':', '_')}",
        "__package__": package,
        **{
            str(name): _binding_value(value, resolved_callables)
            for name, value in bindings.items()
        },
    }
    source_name = str(callable_meta.get("name") or function_name)
    entry_source = _sanitized_entry_source(source, function_name)
    if function_name != source_name and source_name not in {"", "<lambda>"}:
        entry_source += f"\n{function_name} = {source_name}\n"
    root_string = str(root)
    if root_string not in sys.path:
        # Keep the materialized root importable for lazy imports inside callbacks.
        sys.path.insert(0, root_string)
    try:
        exec(compile(entry_source, entry_file, "exec"), namespace)
    except Exception as exc:
        raise CodeBundleError(
            f"could not load entry function {function_name!r}: {exc}"
        ) from exc
    loaded = namespace.get(function_name)
    if not callable(loaded) and source_name == "<lambda>":
        candidates = [
            value
            for name, value in namespace.items()
            if not name.startswith("__")
            and callable(value)
            and getattr(value, "__module__", None) == namespace["__name__"]
        ]
        if len(candidates) == 1:
            loaded = candidates[0]
    if not callable(loaded):
        raise CodeBundleError(f"entry function {function_name!r} is not callable")
    return loaded, str(root)


__all__ = [
    "CodeBundleError",
    "decode_code_bundle",
    "load_bundle_callable",
    "validate_manifest_compatibility",
]
