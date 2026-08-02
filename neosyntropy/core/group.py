"""Group: a named collection of nodes.

Groups organize nodes and, when targeted by a semantic edge, scope hybrid
candidate search. The plan validator and executor still decide permission
from expanded edge targets — groups are not a second control path of their
own.
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from .models import RuntimeModel


class Group(RuntimeModel):
    name: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
