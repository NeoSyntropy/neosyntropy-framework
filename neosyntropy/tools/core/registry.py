"""Pydantic-style tool registration and invocation logging.

Preserves the ``neosyntropy-inference`` ``@neosyntropy`` contract exactly (single pydantic
args model, JSON schema, ``ToolInvocation`` audit log) so native edge
extractors plug in without changes. ``@tool`` is the framework name;
``neosyntropy`` is kept as a drop-in alias.

Enforcement: node handlers receive a :class:`BoundTools` facade that only
permits the tools declared on the node — the tool allow-list is checked at
the call site, fail-closed.
"""
from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ToolInvocation(BaseModel):
    """Structured record of every tool call and output."""

    model_config = ConfigDict(extra="forbid")
    tool: str
    arguments: dict[str, Any]
    result: Any | None = None
    ok: bool = True
    error: str | None = None
    latency_ms: float = Field(ge=0.0)
    timestamp: float


@dataclass
class RegisteredTool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]
    json_schema: dict[str, Any]
    return_schema: dict[str, Any] | None = None


@dataclass
class ToolRegistry:
    tools: dict[str, RegisteredTool] = field(default_factory=dict)
    invocations: list[ToolInvocation] = field(default_factory=list)

    def register(self, tool: RegisteredTool) -> RegisteredTool:
        if tool.name in self.tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self.tools[tool.name] = tool
        return tool

    def get(self, name: str) -> RegisteredTool:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        return self.tools[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self.tools)

    def clear_invocations(self) -> None:
        self.invocations.clear()

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolInvocation:
        tool = self.get(name)
        started = time.perf_counter()
        try:
            validated = tool.args_model.model_validate(arguments)
            result = tool.handler(validated)
            invocation = ToolInvocation(
                tool=name,
                arguments=validated.model_dump(),
                result=result,
                ok=True,
                latency_ms=(time.perf_counter() - started) * 1000,
                timestamp=time.time(),
            )
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            invocation = ToolInvocation(
                tool=name,
                arguments=dict(arguments),
                result=None,
                ok=False,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
                timestamp=time.time(),
            )
        self.invocations.append(invocation)
        return invocation


DEFAULT_REGISTRY = ToolRegistry()


class ToolNotAllowedError(PermissionError):
    """Tool not permitted on this node (allow-list fail-closed)."""


@dataclass
class BoundTools:
    """Node-scoped view of the registry enforcing the node's allow-list."""

    registry: ToolRegistry
    allowed: tuple[str, ...]
    node_id: str

    def names(self) -> tuple[str, ...]:
        return self.allowed

    def __contains__(self, name: str) -> bool:
        return name in self.allowed

    def specs(self) -> tuple[RegisteredTool, ...]:
        return tuple(self.registry.get(name) for name in self.allowed)

    def try_invoke(
        self, name: str, arguments: dict[str, Any] | BaseModel
    ) -> ToolInvocation:
        """Invoke under the allow-list, returning the record instead of raising.

        The allow-list itself still fails closed: a disallowed tool raises
        :class:`ToolNotAllowedError` and never executes. A tool that runs and
        fails comes back as a record with ``ok=False``.
        """
        if name not in self.allowed:
            raise ToolNotAllowedError(
                f"Tool '{name}' is not allowed on node '{self.node_id}'."
            )
        payload = (
            arguments.model_dump() if isinstance(arguments, BaseModel) else dict(arguments)
        )
        return self.registry.invoke(name, payload)

    def invoke(self, name: str, arguments: dict[str, Any] | BaseModel) -> Any:
        invocation = self.try_invoke(name, arguments)
        if not invocation.ok:
            raise ValueError(invocation.error or f"{name} failed")
        return invocation.result


def _is_model(candidate: Any) -> bool:
    return isinstance(candidate, type) and issubclass(candidate, BaseModel)


def _return_schema(
    func: Callable[..., Any],
    *,
    localns: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Best-effort JSON Schema for a tool's return value."""
    try:
        hints = get_type_hints(func, localns=localns, include_extras=True)
    except Exception:
        hints = {}
    annotation = hints.get("return", inspect.signature(func).return_annotation)
    if annotation is inspect.Signature.empty or annotation is None:
        return None
    if _is_model(annotation):
        schema = annotation.model_json_schema()
        schema.setdefault("type", "object")
        schema.setdefault("additionalProperties", False)
        return schema
    origin = get_origin(annotation) or annotation
    if origin is dict:
        args = get_args(annotation)
        value = args[1] if len(args) == 2 else Any
        if _is_model(value):
            return {
                "type": "object",
                "additionalProperties": value.model_json_schema(),
            }
        return {"type": "object"}
    if origin is list:
        args = get_args(annotation)
        item = args[0] if args else Any
        if _is_model(item):
            return {"type": "array", "items": item.model_json_schema()}
        return {"type": "array"}
    if annotation in {str, int, float, bool}:
        return {"type": {str: "string", int: "integer", float: "number", bool: "boolean"}[annotation]}
    name = getattr(annotation, "__name__", None) or str(annotation)
    return {"type": name}


def _resolve_args_model(
    func: Callable[..., Any],
    args_model: type[BaseModel] | None,
    *,
    localns: dict[str, Any] | None = None,
) -> type[BaseModel]:
    if args_model is not None:
        if not _is_model(args_model):
            raise TypeError("args_model must be a pydantic BaseModel subclass")
        return args_model
    params = [
        name for name in inspect.signature(func).parameters if name not in {"self", "cls"}
    ]
    if len(params) != 1:
        raise TypeError(
            f"{func.__name__} must take exactly one args parameter "
            "(or pass args_model= to @tool)"
        )
    param_name = params[0]
    annotation = inspect.signature(func).parameters[param_name].annotation
    if _is_model(annotation):
        return annotation
    namespaces = [localns or {}, getattr(func, "__globals__", {})]
    try:
        from typing import get_type_hints

        hints = get_type_hints(
            func, globalns=getattr(func, "__globals__", None), localns=localns
        )
        candidate = hints.get(param_name)
        if _is_model(candidate):
            return candidate
    except NameError:
        candidate = None
    if isinstance(annotation, str):
        for namespace in namespaces:
            resolved = namespace.get(annotation)
            if _is_model(resolved):
                return resolved
    raise TypeError(f"{func.__name__} args parameter must be annotated with a BaseModel")


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_model: type[BaseModel] | None = None,
    registry: ToolRegistry | None = None,
) -> Any:
    """Register a Python function as a tool with pydantic args::

        class AddToCartArgs(BaseModel):
            product_id: str
            quantity: int

        @tool
        def add_to_cart(args: AddToCartArgs) -> dict:
            \"\"\"Add a quantity of a product to the active cart.\"\"\"
            ...
    """
    import sys

    # Capture caller locals so nested BaseModel classes resolve under
    # `from __future__ import annotations`.
    caller_locals = sys._getframe(1).f_locals

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        model = _resolve_args_model(fn, args_model, localns=caller_locals)
        tool_name = name or fn.__name__
        tool_description = (description or inspect.getdoc(fn) or tool_name).strip()
        target = registry or DEFAULT_REGISTRY

        def handler(validated: BaseModel) -> Any:
            return fn(validated)

        json_schema = model.model_json_schema()
        json_schema.setdefault("additionalProperties", False)
        registered = RegisteredTool(
            name=tool_name,
            description=tool_description,
            args_model=model,
            handler=handler,
            json_schema=json_schema,
            return_schema=_return_schema(fn, localns=caller_locals),
        )
        target.register(registered)

        def wrapped(arguments: dict[str, Any] | BaseModel) -> Any:
            payload = (
                arguments.model_dump()
                if isinstance(arguments, BaseModel)
                else dict(arguments)
            )
            invocation = target.invoke(tool_name, payload)
            if not invocation.ok:
                raise ValueError(invocation.error or f"{tool_name} failed")
            return invocation.result

        wrapped.__neosyntropy_tool__ = registered  # type: ignore[attr-defined]
        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = fn.__doc__
        return wrapped

    if func is not None:
        return decorator(func)
    return decorator


# Drop-in alias matching the neosyntropy-inference decorator name.
neosyntropy = tool


def registered_tools(
    names: Sequence[str] | None = None,
    *,
    registry: ToolRegistry | None = None,
) -> dict[str, RegisteredTool]:
    target = registry or DEFAULT_REGISTRY
    if names is None:
        return dict(target.tools)
    missing = set(names) - set(target.tools)
    if missing:
        raise KeyError(f"Unknown tools: {sorted(missing)}")
    return {name: target.tools[name] for name in names}
