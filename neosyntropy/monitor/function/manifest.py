"""Function manifest generator for UI visualisation and telemetry."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def function_manifest(
    func: Callable[..., Any],
    *,
    fsm: Any = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Return the serialisable description of a decorated function.

    This is emitted at decoration time (when ``@workflow`` or
    ``@function_calling`` is applied) so the console can show the function
    before any run has happened.

    Parameters
    ----------
    func:
        The original Python function that was decorated.
    fsm:
        The compiled ``Workflow`` FSM, if available.  When present, node and
        edge metadata are included.
    project_id:
        Optional project scope for the manifest.
    """
    source_code: str | None = None
    try:
        source_code = inspect.getsource(func)
    except (OSError, TypeError):
        pass

    payload: dict[str, Any] = {
        "schema_version": 1,
        "name": func.__name__,
        "function_name": func.__name__,
        "function_module": getattr(func, "__module__", None),
        "docstring": inspect.getdoc(func),
        "description": inspect.getdoc(func),
        "is_async": inspect.iscoroutinefunction(func),
        "source_code": source_code,
    }

    if fsm is not None:
        payload["node_count"] = len(getattr(fsm, "nodes", {}))
        payload["entry"] = getattr(fsm, "entry_id", None)
        payload["input_schema"] = getattr(fsm, "input_schema", None)
        decorator = getattr(fsm, "decorator", None)
        if decorator:
            payload["decorator"] = decorator

    return payload
