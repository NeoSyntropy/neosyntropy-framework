from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

from neosyntropy import FSM, OpenInput, SchemaNode, TextOutput, edge_deterministic, edge_fallback, node
from neosyntropy.backend import BackendClient, Client
from neosyntropy.benchmark.cookbook_train import CapturedRun, train_captured_runs
from neosyntropy.benchmark.from_run import samples_from_run, trainable_node_ids
from neosyntropy.core.models import (
    AuditRecord,
    ExecutionStepResult,
    NodeResult,
    RunResult,
)
from neosyntropy.core.routing.semantic import SemanticRouter


def _schema_fsm() -> FSM:
    extract = SchemaNode(
        id="ExtractTicket",
        input_schema=OpenInput,
        output_schema=OpenInput,
        prompt="Extract the ticket.",
        provider="gemini-2.5-flash",
    )
    fallback = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="Refuse.",
        provider="gemini-2.5-flash",
    )
    return FSM(
        entry=extract,
        nodes=[extract, fallback],
        edges=[
            edge_deterministic("ExtractTicket", "End"),
            edge_fallback("ExtractTicket", "OutOfScope"),
        ],
    )


def _handler_fsm() -> FSM:
    @node(id="Validate", input_schema=OpenInput, output_schema=OpenInput)
    def validate(ctx: Any) -> Any:
        return ctx.result(output={"ok": True})

    fallback = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="Refuse.",
    )
    return FSM(
        entry=validate,
        nodes=[validate, fallback],
        edges=[
            edge_deterministic("Validate", "End"),
            edge_fallback("Validate", "OutOfScope"),
        ],
    )


def _run_result(*, node_id: str, output: dict[str, Any], request_id: str = "run-1") -> RunResult:
    record = NodeResult(node_id=node_id, status="succeeded", output=output)
    step = ExecutionStepResult(step=0, results=[record])
    audit = AuditRecord(
        request_id=request_id,
        input={"text": "please update billing"},
        initial_state=node_id,
        final_state="End",
        steps=[step],
        committed_transitions=["End"],
        rejected=False,
    )
    return RunResult(
        request_id=request_id,
        steps=[step],
        final_state="End",
        state={},
        completed=True,
        audit=audit,
    )


def test_samples_from_run_maps_trainable_node_output() -> None:
    fsm = _schema_fsm()
    result = _run_result(
        node_id="ExtractTicket",
        output={"text": "billing update"},
    )
    by_node = samples_from_run(result, fsm=fsm, scenario="schema_node_example")
    assert set(by_node) == {"ExtractTicket"}
    sample = by_node["ExtractTicket"][0]
    assert sample["source"] == "real_run"
    assert sample["scenario"] == "schema_node_example"
    assert sample["input_json"] == {"text": "please update billing"}
    assert sample["ground_truth_json"] == {"text": "billing update"}
    assert sample["external_key"] == "run-1:ExtractTicket:0"


def test_handler_nodes_are_not_trainable() -> None:
    fsm = _handler_fsm()
    assert "Validate" not in trainable_node_ids(fsm)
    assert "OutOfScope" in trainable_node_ids(fsm)
    result = _run_result(node_id="Validate", output={"ok": True})
    assert samples_from_run(result, fsm=fsm) == {}


def test_semantic_router_run_labels_chosen_route() -> None:
    @node(id="BillingHelp", input_schema=OpenInput, output_schema=TextOutput)
    def billing(ctx: Any) -> Any:
        return ctx.result(output={"message": "billing"})

    fallback = SchemaNode(
        id="OutOfScope",
        is_fallback=True,
        input_schema=OpenInput,
        output_schema=TextOutput,
        prompt="Refuse.",
    )
    router = SemanticRouter(
        id="SupportIntent",
        input_schema=OpenInput,
        routes={"billing": billing},
        fallback_node=fallback,
        provider="gemini-2.5-flash",
    )
    fsm = FSM(
        entry=router,
        nodes=[billing, fallback],
        routers=[router],
        edges=[
            edge_deterministic("BillingHelp", "End"),
            edge_fallback("SupportIntent", "OutOfScope"),
        ],
    )
    result = _run_result(node_id="BillingHelp", output={"message": "billing"})
    by_node = samples_from_run(result, fsm=fsm, scenario="semantic_router")
    assert "SupportIntent" in trainable_node_ids(fsm)
    assert set(by_node) == {"SupportIntent"}
    assert by_node["SupportIntent"][0]["ground_truth_json"] == {
        "chosen_next_node": "BillingHelp",
        "route": "billing",
    }


def test_create_eval_samples_posts_bulk_payload() -> None:
    calls: list[tuple[str, dict[str, Any], bool]] = []

    async def fake_post(
        self: BackendClient,
        path: str,
        payload: dict[str, Any],
        *,
        allow_list: bool = False,
    ) -> list[dict[str, Any]]:
        calls.append((path, payload, allow_list))
        return [{"id": "s1", **payload["samples"][0]}]

    client = BackendClient("https://api.example.com/api/v1", api_key="nsk_test")
    with patch.object(BackendClient, "post", fake_post):
        created = asyncio.run(
            client.create_eval_samples(
                "proj",
                "ExtractTicket",
                [{"split": "train", "source": "real_run", "input_json": {"text": "a"}}],
            )
        )

    assert created[0]["id"] == "s1"
    assert calls[0][0].endswith("/nodes/ExtractTicket/eval-samples")
    assert calls[0][2] is True


def test_get_tune_status_uses_tune_status_path() -> None:
    seen: list[str] = []

    async def fake_get(self: BackendClient, path: str) -> dict[str, Any]:
        seen.append(path)
        return {
            "project_id": "proj",
            "node_id": "ExtractTicket",
            "sample_count": 1,
            "required": 200,
            "eligible": False,
            "tuned": False,
        }

    client = BackendClient("https://api.example.com/api/v1", api_key="nsk_test")
    with patch.object(BackendClient, "get", fake_get):
        status = asyncio.run(client.get_tune_status("proj", "ExtractTicket"))
    assert status["required"] == 200
    assert seen == [
        "/observability/projects/proj/nodes/ExtractTicket/tune-status"
    ]


def test_train_captured_runs_uploads_criticizes_and_accepts() -> None:
    fsm = _schema_fsm()
    result = _run_result(
        node_id="ExtractTicket",
        output={"text": "billing update"},
    )
    backend = BackendClient("https://api.example.com/api/v1", api_key="nsk_test")
    client = Client(api_key="nsk_test", project_id="proj-1")
    client._backend = backend
    captured = [
        CapturedRun(fsm=fsm, result=result, client=client, project_id="proj-1")
    ]

    posts: list[str] = []
    samples = [
        {
            "id": "s-real",
            "status": "candidate",
            "critic_json": None,
        }
    ]

    async def fake_create(self, project_id, node_id, batch):
        posts.append(f"create:{node_id}:{len(batch)}")
        return [{"id": "s-real"}]

    async def fake_pull(self, project_id, node_id):
        return list(samples)

    async def fake_critic(self, project_id, node_id, sample_ids, **kwargs):
        posts.append(f"critic:{node_id}:{len(sample_ids or [])}")
        samples[0] = {
            "id": "s-real",
            "status": "candidate",
            "critic_json": {"good": True, "labeled": True, "score": 1},
        }
        return {"reviewed": 1, "kept": 1, "deleted": 0}

    async def fake_accept(self, project_id, node_id, sample_id):
        posts.append(f"accept:{sample_id}")
        samples[0]["status"] = "accepted"
        return {"sample": samples[0], "training_sample_id": sample_id, "duplicate": False}

    async def fake_status(self, project_id, node_id):
        return {
            "sample_count": 1,
            "required": 200,
            "eligible": False,
            "tuned": False,
        }

    async def fail_generate(*_args, **_kwargs):
        raise AssertionError("synthetic generation should be skipped when synthetic=0")

    with (
        patch.object(BackendClient, "create_eval_samples", fake_create),
        patch.object(BackendClient, "pull_eval_samples", fake_pull),
        patch.object(BackendClient, "critic_eval_samples", fake_critic),
        patch.object(BackendClient, "accept_eval_sample", fake_accept),
        patch.object(BackendClient, "get_tune_status", fake_status),
        patch(
            "neosyntropy.benchmark.cookbook_train._generate_labeled_pairs",
            fail_generate,
        ),
    ):
        report = asyncio.run(
            train_captured_runs(
                captured,
                cookbook="cookbook/fsm/schema_node_example.py",
                synthetic=0,
                tune=True,
            )
        )

    assert report.project_id == "proj-1"
    assert report.error is None
    node = report.nodes[0]
    assert node.node_id == "ExtractTicket"
    assert node.uploaded == 1
    assert node.accepted == 1
    assert node.eligible is False
    assert "need 200" in (node.skipped or "")
    assert posts == ["create:ExtractTicket:1", "critic:ExtractTicket:1", "accept:s-real"]
