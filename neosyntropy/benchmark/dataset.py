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
    prefill: str | None = None
    expected_output: dict[str, Any] | None = None
    expected_reasoning: str | None = None
    criteria: list[str] | None = None
    critic_json: dict[str, Any] | None = None
    status: str | None = None


class RouterTestCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    input_state: dict[str, Any]
    target_router_id: str
    prefill: str | None = None
    expected_route: str
    criteria: list[str] | None = None
    critic_json: dict[str, Any] | None = None
    status: str | None = None


class FSMTestCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    input_state: dict[str, Any]
    prefill: str | None = None
    expected_final_state: dict[str, Any] | None = None
    expected_path: list[str] | None = None
    expected_output: dict[str, Any] | None = None
    criteria: list[str] | None = None
    critic_json: dict[str, Any] | None = None
    status: str | None = None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sample_input(sample: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(sample.get("payload"))
    raw = sample.get("input_json", payload.get("input_state", payload.get("input")))
    return _as_dict(raw)


def _sample_expected(sample: Mapping[str, Any]) -> Any:
    payload = _as_dict(sample.get("payload"))
    if "ground_truth_json" in sample:
        return sample.get("ground_truth_json")
    return payload.get("expected_output", payload.get("expected", payload.get("expected_route")))


def _sample_is_router(sample: Mapping[str, Any], expected: Any) -> bool:
    payload = _as_dict(sample.get("payload"))
    kind = str(sample.get("kind") or payload.get("kind") or "").lower()
    if kind == "router":
        return True
    if isinstance(expected, dict) and (
        expected.get("chosen_next_node") or expected.get("next_state")
    ):
        return True
    return False


def case_is_labeled(case: NodeTestCase | RouterTestCase | FSMTestCase) -> bool:
    data = case.critic_json
    if not isinstance(data, dict):
        return False
    if data.get("labeled") is True or data.get("good") is True:
        return True
    if data.get("score") in (1, "1"):
        return True
    return data.get("match") is True


def case_is_unlabeled(case: NodeTestCase | RouterTestCase | FSMTestCase) -> bool:
    return case.critic_json is None


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
            if not isinstance(sample, Mapping):
                continue
            payload = _as_dict(sample.get("payload"))
            kind = str(sample.get("kind") or payload.get("kind") or "").lower()
            sample_id = sample.get("id")
            case_id = str(sample_id) if sample_id is not None else None
            input_state = _sample_input(sample)
            expected = _sample_expected(sample)
            critic_json = sample.get("critic_json")
            critic = critic_json if isinstance(critic_json, dict) else None
            status = sample.get("status") if isinstance(sample.get("status"), str) else None
            criteria = payload.get("criteria")

            if kind == "fsm":
                fsm_cases.append(
                    FSMTestCase(
                        id=case_id,
                        input_state=input_state,
                        expected_output=expected if isinstance(expected, dict) else None,
                        expected_final_state=payload.get("expected_final_state"),
                        expected_path=payload.get("expected_path"),
                        criteria=criteria,
                        critic_json=critic,
                        status=status,
                    )
                )
            elif kind == "router" or _sample_is_router(sample, expected):
                route = ""
                if isinstance(expected, dict):
                    raw_route = expected.get("chosen_next_node") or expected.get(
                        "next_state"
                    )
                    route = str(raw_route or payload.get("expected_route") or "")
                elif isinstance(expected, str):
                    route = expected
                router_cases.append(
                    RouterTestCase(
                        id=case_id,
                        input_state=input_state,
                        target_router_id=node_id,
                        expected_route=route or str(payload.get("expected_route") or ""),
                        criteria=criteria,
                        critic_json=critic,
                        status=status,
                    )
                )
            else:
                expected_output = expected if isinstance(expected, dict) else None
                node_cases.append(
                    NodeTestCase(
                        id=case_id,
                        input_state=input_state,
                        target_node_id=node_id,
                        expected_output=expected_output,
                        criteria=criteria,
                        critic_json=critic,
                        status=status,
                    )
                )

        return cls(
            node_cases=node_cases,
            router_cases=router_cases,
            fsm_cases=fsm_cases,
        )

    def labeled_only(self) -> "BenchmarkDataset":
        """Drop unlabeled / discarded cases so benchmark scores gold labels only."""
        return BenchmarkDataset(
            node_cases=[
                case
                for case in self.node_cases
                if case.status != "discarded" and case_is_labeled(case)
            ],
            router_cases=[
                case
                for case in self.router_cases
                if case.status != "discarded" and case_is_labeled(case)
            ],
            fsm_cases=[
                case
                for case in self.fsm_cases
                if case.status != "discarded" and case_is_labeled(case)
            ],
        )

    def unlabeled_ids_by_node(self) -> dict[str, list[str]]:
        by_node: dict[str, list[str]] = {}
        for case in self.node_cases:
            if case.id and case_is_unlabeled(case) and case.status != "discarded":
                by_node.setdefault(case.target_node_id, []).append(case.id)
        for case in self.router_cases:
            if case.id and case_is_unlabeled(case) and case.status != "discarded":
                by_node.setdefault(case.target_router_id, []).append(case.id)
        return by_node

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
