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


import asyncio

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
