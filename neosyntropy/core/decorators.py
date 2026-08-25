"""Decorators for authoring NeoSyntropy functions and workflows."""

from __future__ import annotations

import asyncio
import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from .graph import Workflow
from .node import ReasoningNode, ReasoningStep, SchemaNode, SchemaStep


def _run_sync(coro: Any) -> Any:
    """Helper to run an async coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(lambda: asyncio.run(coro)).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _prepare_input(args: tuple[Any, ...], kwargs: dict[str, Any], input_schema: Any) -> Any:
    if args and isinstance(args[0], (BaseModel, dict)):
        return args[0]
    if kwargs:
        if isinstance(input_schema, type) and issubclass(input_schema, BaseModel):
            return input_schema(**kwargs)
        return kwargs
    if args:
        return args[0]
    return {}


async def _maybe_log_run(client: Any, project_id: str, input_data: Any, result: Any) -> None:
    try:
        log_fn = getattr(client, "log_run", None)
        if log_fn:
            if inspect.iscoroutinefunction(log_fn):
                await log_fn(project_id=project_id, input=input_data, output=result)
            else:
                log_fn(project_id=project_id, input=input_data, output=result)
    except Exception:
        pass


def _maybe_log_run_sync(client: Any, project_id: str, input_data: Any, result: Any) -> None:
    try:
        log_fn = getattr(client, "log_run", None)
        if log_fn:
            if inspect.iscoroutinefunction(log_fn):
                _run_sync(log_fn(project_id=project_id, input=input_data, output=result))
            else:
                log_fn(project_id=project_id, input=input_data, output=result)
    except Exception:
        pass


def _last_node_output(result: Any) -> Any:
    """Return the decorated function's output from a workflow run."""
    if getattr(result, "rejected", False):
        return result
    steps = getattr(result, "steps", None)
    if steps:
        last = steps[-1]
        outputs = getattr(last, "results", None)
        if outputs:
            return outputs[-1].output
    if hasattr(result, "final_state"):
        return result.final_state
    return result


def function_calling(
    *,
    prompt: str,
    input_schema: type[BaseModel] | dict[str, Any],
    client: Any = None,
    project_id: str | None = None,
    provider: str = "neosyntropy/base",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for simple single-step LLM parameter extraction directly into a function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        node_id = f"{func.__name__}_schema"
        fallback_id = f"{func.__name__}_fallback"

        schema_node = SchemaNode(
            id=node_id,
            input_schema=input_schema,
            prompt=prompt,
            func=func,
            provider=provider,
        )

        fallback_node = SchemaNode(
            id=fallback_id,
            input_schema=input_schema,
            output_schema=schema_node.output_schema,
            prompt="Fallback node",
            is_fallback=True,
            provider=provider,
        )

        fsm = Workflow([schema_node], fallback=fallback_node)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                input_data = _prepare_input(args, kwargs, input_schema)
                result = await fsm.arun(input_data, client=client)
                if client and project_id:
                    await _maybe_log_run(client, project_id, input_data, result)
                return _last_node_output(result)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                input_data = _prepare_input(args, kwargs, input_schema)
                coro = fsm.arun(input_data, client=client)
                result = _run_sync(coro)
                if client and project_id:
                    _maybe_log_run_sync(client, project_id, input_data, result)
                return _last_node_output(result)

            return sync_wrapper

    return decorator


def workflow(
    *,
    input_schema: type[BaseModel] | dict[str, Any],
    steps: Sequence[ReasoningStep | SchemaStep],
    client: Any = None,
    project_id: str | None = None,
    provider: str = "neosyntropy/base",
    tools: Any = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for multi-step reasoning workflows ending with a parameter extraction SchemaStep.

    Pass a :class:`~neosyntropy.tools.core.registry.ToolRegistry` as ``tools``
    so :class:`~neosyntropy.core.node.ReasoningStep` allow-lists can invoke
    registered tools while gathering evidence for parameter extraction.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if not steps or not isinstance(steps[-1], SchemaStep):
            raise ValueError(f"@workflow for '{func.__name__}' must end with a SchemaStep.")

        sequence: list[Any] = []

        for i, step in enumerate(steps[:-1]):
            if not isinstance(step, ReasoningStep):
                raise ValueError("All steps before the final SchemaStep must be ReasoningSteps.")
            step_id = f"{func.__name__}_reasoning_{i}"
            sequence.append(
                ReasoningNode(
                    id=step_id,
                    input_schema=input_schema,
                    prompt=step.instruction,
                    tools=step.tools,
                    provider=provider,
                )
            )

        schema_step = steps[-1]
        schema_node_id = f"{func.__name__}_schema"
        final_prompt = schema_step.instruction or (
            f"Extract the parameters required by the function '{func.__name__}' "
            "based on the gathered context and tool results."
        )
        schema_node = SchemaNode(
            id=schema_node_id,
            input_schema=input_schema,
            prompt=final_prompt,
            func=func,
            provider=provider,
        )
        sequence.append(schema_node)

        fallback_id = f"{func.__name__}_fallback"
        fallback_node = SchemaNode(
            id=fallback_id,
            input_schema=input_schema,
            output_schema=schema_node.output_schema,
            prompt="Fallback node",
            is_fallback=True,
            provider=provider,
        )

        fsm = Workflow(sequence, fallback=fallback_node)

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                input_data = _prepare_input(args, kwargs, input_schema)
                result = await fsm.arun(input_data, client=client, tools=tools)
                if client and project_id:
                    await _maybe_log_run(client, project_id, input_data, result)
                return _last_node_output(result)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                input_data = _prepare_input(args, kwargs, input_schema)
                coro = fsm.arun(input_data, client=client, tools=tools)
                result = _run_sync(coro)
                if client and project_id:
                    _maybe_log_run_sync(client, project_id, input_data, result)
                return _last_node_output(result)

            return sync_wrapper

    return decorator
