"""Group: a named collection of nodes for organization.

Groups affect candidate metadata and developer ergonomics only. The plan
validator and the executor never consult groups — grouping must not create a
second control path.
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from .models import RuntimeModel


class Group(RuntimeModel):
    name: str
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
