from __future__ import annotations

import json
from typing import Any, Mapping, Sequence
from pydantic import BaseModel, ConfigDict, Field
from ..backend import BackendClient


class NodeTestCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    input_state: dict[str, Any]
    target_node_id: str
    expected_output: dict[str, Any] | None = None
    expected_reasoning: str | None = None
    criteria: list[str] | None = None


class RouterTestCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    input_state: dict[str, Any]
    target_router_id: str
    expected_route: str
    criteria: list[str] | None = None


class FSMTestCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    input_state: dict[str, Any]
    expected_final_state: dict[str, Any] | None = None
    expected_path: list[str] | None = None
    expected_output: dict[str, Any] | None = None
    criteria: list[str] | None = None


class BenchmarkDataset(BaseModel):
    """Collection of benchmark datasets for nodes, routers, and full FSM."""

    model_config = ConfigDict(extra="allow")

    node_cases: list[NodeTestCase] = Field(default_factory=list)
    router_cases: list[RouterTestCase] = Field(default_factory=list)
    fsm_cases: list[FSMTestCase] = Field(default_factory=list)

    @classmethod
    async def from_backend(
        cls, client: BackendClient, project_id: str, node_id: str
    ) -> "BenchmarkDataset":
        """Pull dataset samples from the backend."""
        samples = await client.pull_eval_samples(project_id, node_id)
        
        node_cases = []
        router_cases = []
        fsm_cases = []
        
        for sample in samples:
            # Backend dataset formats may vary. Assuming sample contains 'input' and 'expected'
            # Or if it's a DatasetSampleOut, it might have specific fields.
            payload = sample.get("payload", sample)
            kind = payload.get("kind", "node")
            
            if kind == "node":
                node_cases.append(NodeTestCase(
                    id=sample.get("id"),
                    input_state=payload.get("input_state", payload.get("input", {})),
                    target_node_id=node_id,
                    expected_output=payload.get("expected_output"),
                    criteria=payload.get("criteria"),
                ))
            elif kind == "router":
                router_cases.append(RouterTestCase(
                    id=sample.get("id"),
                    input_state=payload.get("input_state", payload.get("input", {})),
                    target_router_id=node_id,
                    expected_route=payload.get("expected_route", ""),
                    criteria=payload.get("criteria"),
                ))
            elif kind == "fsm":
                fsm_cases.append(FSMTestCase(
                    id=sample.get("id"),
                    input_state=payload.get("input_state", payload.get("input", {})),
                    expected_final_state=payload.get("expected_final_state"),
                    expected_path=payload.get("expected_path"),
                    expected_output=payload.get("expected_output"),
                    criteria=payload.get("criteria"),
                ))
            else:
                # Default to node if kind is unknown
                node_cases.append(NodeTestCase(
                    id=sample.get("id"),
                    input_state=payload.get("input_state", payload.get("input", {})),
                    target_node_id=node_id,
                    expected_output=payload.get("expected_output", payload.get("expected")),
                ))

        return cls(
            node_cases=node_cases,
            router_cases=router_cases,
            fsm_cases=fsm_cases,
        )

    @classmethod
    def from_dataframe(cls, df: Any, kind: str = "node", target_id: str = "") -> "BenchmarkDataset":
        """Load benchmark cases from a pandas DataFrame.
        
        Args:
            df: A pandas DataFrame containing columns that match the test case schema.
            kind: "node", "router", or "fsm"
            target_id: The ID of the target node or router to associate these cases with.
        """
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        node_cases = []
        router_cases = []
        fsm_cases = []

        for idx, row in df.iterrows():
            row_dict = row.dropna().to_dict()
            
            input_state = row_dict.get("input_state", row_dict.get("input", {}))
            if isinstance(input_state, str):
                try:
                    input_state = json.loads(input_state)
                except json.JSONDecodeError:
                    input_state = {"text": input_state}
            
            expected_output = row_dict.get("expected_output", row_dict.get("expected"))
            if isinstance(expected_output, str):
                try:
                    expected_output = json.loads(expected_output)
                except json.JSONDecodeError:
                    expected_output = {"text": expected_output}

            if kind == "node":
                node_cases.append(NodeTestCase(
                    id=str(row_dict.get("id", idx)),
                    input_state=input_state,
                    target_node_id=target_id,
                    expected_output=expected_output,
                    criteria=row_dict.get("criteria"),
                ))
            elif kind == "router":
                router_cases.append(RouterTestCase(
                    id=str(row_dict.get("id", idx)),
                    input_state=input_state,
                    target_router_id=target_id,
                    expected_route=row_dict.get("expected_route", row_dict.get("expected", "")),
                    criteria=row_dict.get("criteria"),
                ))
            elif kind == "fsm":
                fsm_cases.append(FSMTestCase(
                    id=str(row_dict.get("id", idx)),
                    input_state=input_state,
                    expected_final_state=row_dict.get("expected_final_state"),
                    expected_path=row_dict.get("expected_path"),
                    expected_output=expected_output,
                    criteria=row_dict.get("criteria"),
                ))

        return cls(
            node_cases=node_cases,
            router_cases=router_cases,
            fsm_cases=fsm_cases,
        )
