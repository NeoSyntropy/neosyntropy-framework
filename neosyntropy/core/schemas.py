"""Common structured input/output models for node declarations."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def strict_model_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the constrained-decoding schema for a Pydantic output model."""
    schema = model.model_json_schema()
    schema.setdefault("type", "object")
    schema["additionalProperties"] = False
    if "properties" in schema:
        schema["required"] = list(schema["properties"])
    return schema


def input_model_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a closed JSON Schema for node/graph input contracts.

    Unlike output schemas (where constrained decoding requires every field),
    optional input fields stay optional: pydantic defaults are honored.
    Unknown keys follow the model's ``extra`` config (forbid by default).
    """
    schema = model.model_json_schema()
    schema.setdefault("type", "object")
    extra = model.model_config.get("extra")
    if extra == "allow":
        schema["additionalProperties"] = True
    else:
        schema["additionalProperties"] = False
    return schema


class EmptyInput(BaseModel):
    """Empty object input: no keys required, unknown keys forbidden."""

    model_config = ConfigDict(extra="forbid")


class OpenInput(BaseModel):
    """Permissive input for nodes that do not constrain workflow state."""

    model_config = ConfigDict(extra="allow")


class EmptyOutput(BaseModel):
    """Empty object schema for nodes that only update state or signal completion."""

    model_config = ConfigDict(extra="forbid")


class TextOutput(BaseModel):
    """Single text payload for simple messaging / fallback nodes."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)


class OpenOutput(BaseModel):
    """Permissive object schema for nodes that return varied dict payloads."""

    model_config = ConfigDict(extra="allow")
