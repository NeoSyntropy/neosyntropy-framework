from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .critic import CriticVerdict


@dataclass
class AccuracyMetrics:
    total: int = 0
    passed: int = 0
    total_score: float = 0.0

    @property
    def accuracy(self) -> float:
        return (self.passed / self.total) if self.total > 0 else 0.0

    @property
    def average_score(self) -> float:
        return (self.total_score / self.total) if self.total > 0 else 0.0


@dataclass
class NodeAccuracyTracker:
    target_node_id: str
    metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    results: list[dict[str, Any]] = field(default_factory=list)

    def add_result(self, case_id: str | None, verdict: CriticVerdict) -> None:
        self.metrics.total += 1
        if verdict.passed:
            self.metrics.passed += 1
        self.metrics.total_score += verdict.score
        self.results.append({
            "case_id": case_id,
            "passed": verdict.passed,
            "score": verdict.score,
            "reason": verdict.reason,
            "metadata": verdict.metadata,
        })


@dataclass
class RouterAccuracyTracker:
    target_router_id: str
    metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    results: list[dict[str, Any]] = field(default_factory=list)

    def add_result(self, case_id: str | None, verdict: CriticVerdict) -> None:
        self.metrics.total += 1
        if verdict.passed:
            self.metrics.passed += 1
        self.metrics.total_score += verdict.score
        self.results.append({
            "case_id": case_id,
            "passed": verdict.passed,
            "score": verdict.score,
            "reason": verdict.reason,
            "metadata": verdict.metadata,
        })


@dataclass
class FullPathAccuracyTracker:
    metrics: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    results: list[dict[str, Any]] = field(default_factory=list)

    def add_result(self, case_id: str | None, verdict: CriticVerdict) -> None:
        self.metrics.total += 1
        if verdict.passed:
            self.metrics.passed += 1
        self.metrics.total_score += verdict.score
        self.results.append({
            "case_id": case_id,
            "passed": verdict.passed,
            "score": verdict.score,
            "reason": verdict.reason,
            "metadata": verdict.metadata,
        })
