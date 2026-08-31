"""Protocols defining common capabilities across FSM graph components."""
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Compilable(Protocol):
    """Protocol for any entity that can be compiled into FSM execution primitives."""

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        """Compile the entity into lower-level representations (nodes, edges, etc.)."""
        ...
