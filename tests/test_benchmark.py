import asyncio

import pandas as pd
import pytest

from neosyntropy.benchmark.dataset import BenchmarkDataset, NodeTestCase, RouterTestCase, FSMTestCase
from neosyntropy.benchmark.critic import ExactMatchCritic, CriticVerdict
from neosyntropy.benchmark.metrics import NodeAccuracyTracker, AccuracyMetrics


def test_dataset_from_dataframe_node():
    data = {
        "input_state": [{"text": "hello"}, '{"text": "world"}'],
        "expected_output": [{"response": "hi"}, '{"response": "earth"}'],
    }
    df = pd.DataFrame(data)
    
    dataset = BenchmarkDataset.from_dataframe(df, kind="node", target_id="TestNode")
    
    assert len(dataset.node_cases) == 2
    
    case1 = dataset.node_cases[0]
    assert case1.input_state == {"text": "hello"}
    assert case1.expected_output == {"response": "hi"}
    assert case1.target_node_id == "TestNode"
    
    case2 = dataset.node_cases[1]
    assert case2.input_state == {"text": "world"}
    assert case2.expected_output == {"response": "earth"}


def test_exact_match_critic_node():
    critic = ExactMatchCritic()
    
    async def _run():
        # Matching
        verdict1 = await critic.evaluate_node(
            target_node_id="TestNode",
            input_state={},
            actual_output={"a": 1},
            expected_output={"a": 1}
        )
        assert verdict1.passed is True
        assert verdict1.score == 1.0
        
        # Non-matching
        verdict2 = await critic.evaluate_node(
            target_node_id="TestNode",
            input_state={},
            actual_output={"a": 1},
            expected_output={"a": 2}
        )
        assert verdict2.passed is False
        assert verdict2.score == 0.0
    
    asyncio.run(_run())


def test_exact_match_critic_router():
    critic = ExactMatchCritic()
    
    async def _run():
        verdict = await critic.evaluate_router(
            target_router_id="TestRouter",
            input_state={},
            actual_route="PathA",
            expected_route="PathA"
        )
        assert verdict.passed is True
        
        verdict_fail = await critic.evaluate_router(
            target_router_id="TestRouter",
            input_state={},
            actual_route="PathA",
            expected_route="PathB"
        )
        assert verdict_fail.passed is False
        
    asyncio.run(_run())


def test_accuracy_metrics():
    tracker = NodeAccuracyTracker(target_node_id="TestNode")
    
    tracker.add_result("case1", CriticVerdict(passed=True, score=1.0, reason="ok"))
    tracker.add_result("case2", CriticVerdict(passed=False, score=0.0, reason="fail"))
    tracker.add_result("case3", CriticVerdict(passed=True, score=0.5, reason="partial"))
    
    assert tracker.metrics.total == 3
    assert tracker.metrics.passed == 2
    assert tracker.metrics.total_score == 1.5
    assert tracker.metrics.accuracy == 2 / 3
    assert tracker.metrics.average_score == 1.5 / 3
    
    assert len(tracker.results) == 3
    assert tracker.results[0]["case_id"] == "case1"
    assert tracker.results[0]["passed"] is True


def test_dataset_from_backend_maps_sample_out_fields():
    class FakeClient:
        async def pull_eval_samples(self, project_id, node_id):
            return [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "input_json": {"text": "refund ORD-1"},
                    "ground_truth_json": {"order_id": "ORD-1", "intent": "refund"},
                    "status": "candidate",
                    "critic_json": {
                        "good": True,
                        "labeled": True,
                        "match": True,
                        "score": 1,
                    },
                },
                {
                    "id": "22222222-2222-2222-2222-222222222222",
                    "input_json": {"intent": "need refund"},
                    "ground_truth_json": {
                        "reasoning": "route to Investigate",
                        "chosen_next_node": "Investigate",
                    },
                    "status": "candidate",
                    "critic_json": None,
                },
            ]

    async def _run():
        dataset = await BenchmarkDataset.from_backend(FakeClient(), "proj", "Extract")
        assert len(dataset.node_cases) == 1
        node = dataset.node_cases[0]
        assert node.input_state == {"text": "refund ORD-1"}
        assert node.expected_output == {"order_id": "ORD-1", "intent": "refund"}
        assert node.critic_json is not None
        assert node.critic_json["labeled"] is True

        assert len(dataset.router_cases) == 1
        router = dataset.router_cases[0]
        assert router.expected_route == "Investigate"
        assert router.critic_json is None

        unlabeled = dataset.unlabeled_ids_by_node()
        assert unlabeled == {"Extract": ["22222222-2222-2222-2222-222222222222"]}
        labeled = dataset.labeled_only()
        assert len(labeled.node_cases) == 1
        assert labeled.router_cases == []

    asyncio.run(_run())


def test_backend_critic_calls_eval_judge():
    from neosyntropy.benchmark.critic import BackendCritic

    calls: list[dict] = []

    class FakeClient:
        async def judge_output(self, project_id, actual_output, ground_truth, *, node_id=None):
            calls.append(
                {
                    "project_id": project_id,
                    "actual_output": actual_output,
                    "ground_truth": ground_truth,
                    "node_id": node_id,
                }
            )
            return {"match": True, "score": 1, "reason": "same intent"}

    async def _run():
        critic = BackendCritic(FakeClient(), "proj-1", model="gemini-2.5-flash")
        verdict = await critic.evaluate_node(
            target_node_id="Extract",
            input_state={"text": "x"},
            actual_output={"order_id": "ORD-1"},
            expected_output={"order_id": "ORD-1"},
        )
        assert verdict.passed is True
        assert verdict.score == 1.0
        assert calls[0]["node_id"] == "Extract"
        assert calls[0]["ground_truth"] == {"order_id": "ORD-1"}

    asyncio.run(_run())


def test_runner_ensure_labels_critics_unlabeled_then_filters():
    from neosyntropy.benchmark.runner import BenchmarkRunner
    from neosyntropy.benchmark.critic import ExactMatchCritic

    critic_calls: list[dict] = []

    class FakeClient:
        async def critic_eval_samples(self, project_id, node_id, sample_ids, **kwargs):
            critic_calls.append(
                {"project_id": project_id, "node_id": node_id, "sample_ids": sample_ids}
            )
            return {"reviewed": 1, "kept": 1, "deleted": 0, "results": []}

        async def pull_eval_samples(self, project_id, node_id):
            return [
                {
                    "id": "aaa",
                    "input_json": {"text": "a"},
                    "ground_truth_json": {"ok": True},
                    "status": "candidate",
                    "critic_json": {
                        "good": True,
                        "labeled": True,
                        "match": True,
                        "score": 1,
                    },
                }
            ]

    async def _run():
        dataset = BenchmarkDataset(
            node_cases=[
                NodeTestCase(
                    id="aaa",
                    input_state={"text": "a"},
                    target_node_id="Extract",
                    expected_output={"ok": True},
                    critic_json=None,
                )
            ]
        )
        runner = BenchmarkRunner(
            dataset=dataset,
            fsm=None,  # type: ignore[arg-type]
            critic=ExactMatchCritic(),
            providers=["gemini-2.5-flash"],
            client=FakeClient(),  # type: ignore[arg-type]
        )
        await runner.ensure_labels("proj-1")
        assert critic_calls == [
            {"project_id": "proj-1", "node_id": "Extract", "sample_ids": ["aaa"]}
        ]
        assert len(runner.dataset.node_cases) == 1
        assert runner.dataset.node_cases[0].critic_json is not None
        assert runner.dataset.node_cases[0].critic_json["labeled"] is True

    asyncio.run(_run())
