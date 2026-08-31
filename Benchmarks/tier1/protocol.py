"""Shared gold protocol: task accuracy, latency, and money.

Both NeoSyntropy and LangGraph run the same JSONL cases. Accuracy is the
gate; latency and USD are the comparison. Exact route/path match is the
task score so a second LLM judge cannot muddy a latency study.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PHASE_ROUTES: dict[str, str] = {
    "analysis": "AnalystPhase",
    "plan": "PlanStub",
    "solutioning": "SolutioningStub",
    "implementation": "ImplementationStub",
    "core": "CoreStub",
    "help": "HelpStub",
}

ENTRY_ID = "PhaseRouter"
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
CASES_PATH = DATASETS_DIR / "cases.jsonl"

# Customer invoice: NeoSyntropy bills committed hops, not tokens.
# Tuned router (neosyntropy/base) is $0.005; glm-5-maas nodes use the starter
# untuned rate of $0.01. Failed / rejected hops are $0.
TUNED_TRANSITION_USD = 0.005
BASELINE_TRANSITION_USD = 0.01
ROUTER_STATES = {ENTRY_ID}

# LangGraph default: gpt-4.1-mini list price from the NeoSyntropy gateway catalog.
# Override via ModelPrice or LANGGRAPH_MODEL. Units are USD per million tokens.
DEFAULT_LG_MODEL = "gpt-4.1-mini"
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "glm-5-maas": (0.30, 1.20),
}


@dataclass(frozen=True)
class Case:
    id: str
    user_request: str
    expected_route: str
    expected_path: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    expected_output: dict[str, Any] = field(default_factory=dict)
    is_headless: bool = False
    project_workspace: str = "."

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Case:
        path = raw.get("expected_path") or []
        tools = raw.get("required_tools") or []
        output = raw.get("expected_output") or {}
        return cls(
            id=str(raw["id"]),
            user_request=str(raw["user_request"]),
            expected_route=str(raw["expected_route"]),
            expected_path=tuple(str(item) for item in path),
            required_tools=tuple(str(item) for item in tools),
            expected_output=dict(output) if isinstance(output, Mapping) else {},
            is_headless=bool(raw.get("is_headless", False)),
            project_workspace=str(raw.get("project_workspace") or "."),
        )


@dataclass
class Trace:
    """One engine run of one case. Engines fill this; scoring is pure."""

    case_id: str
    system: str
    landing: str = ""
    path: list[str] = field(default_factory=list)
    output: dict[str, Any] = field(default_factory=dict)
    tools_ok: list[str] = field(default_factory=list)
    tools_denied: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    transitions: int = 0
    illegal_hops: int = 0
    usd: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredCase:
    case_id: str
    system: str
    passed: bool
    route_ok: bool
    path_ok: bool
    schema_ok: bool
    tools_ok: bool
    illegal_hops: int
    latency_ms: float
    tokens_in: int
    tokens_out: int
    llm_calls: int
    transitions: int
    usd: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemSummary:
    system: str
    n: int
    passed: int
    accuracy: float
    p50_ms: float
    p95_ms: float
    mean_ms: float
    tokens_in: int
    tokens_out: int
    llm_calls: int
    transitions: int
    usd: float
    usd_per_1k: float
    illegal_hops: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cases(path: Path | None = None) -> list[Case]:
    target = path or CASES_PATH
    cases: list[Case] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cases.append(Case.from_mapping(json.loads(line)))
    return cases


def map_phase_label(raw: str) -> tuple[str, int]:
    """Map a free-text / structured phase label to a landing node.

    Returns ``(landing_id, illegal_hops)``. Unknown labels are coerced to
    ``OutOfScope`` and counted as an illegal hop — LangGraph has no graph
    gate, so this is the analogue of a rejected semantic edge.
    """
    label = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "out_of_scope": "out_of_scope",
        "outofscope": "out_of_scope",
        "fallback": "out_of_scope",
        "oos": "out_of_scope",
        "unknown": "out_of_scope",
    }
    label = aliases.get(label, label)
    if label in PHASE_ROUTES:
        return PHASE_ROUTES[label], 0
    if label == "out_of_scope":
        return "OutOfScope", 0
    return "OutOfScope", 1


def path_from_committed(
    committed: Sequence[str], *, entry: str = ENTRY_ID
) -> list[str]:
    """Rebuild a node path from ``audit.committed_transitions``."""
    nodes: list[str] = []
    for hop in committed:
        text = str(hop)
        if "->" in text:
            src, dst = text.split("->", 1)
            src, dst = src.strip(), dst.strip()
            if not nodes:
                nodes.append(src)
            elif nodes[-1] != src:
                nodes.append(src)
            if dst:
                nodes.append(dst)
        elif text and (not nodes or nodes[-1] != text):
            nodes.append(text)
    if nodes and nodes[0] != entry:
        nodes.insert(0, entry)
    elif not nodes:
        nodes = [entry]
    return nodes


def landing_from_path(path: Sequence[str], *, entry: str = ENTRY_ID) -> str:
    for node in path:
        if node not in {entry, "End"}:
            return node
    return "End"


def is_subsequence(expected: Sequence[str], actual: Sequence[str]) -> bool:
    if not expected:
        return True
    index = 0
    for node in actual:
        if index < len(expected) and node == expected[index]:
            index += 1
    return index == len(expected)


def token_usd(
    tokens_in: int,
    tokens_out: int,
    *,
    model: str = DEFAULT_LG_MODEL,
) -> float:
    input_rate, output_rate = MODEL_PRICES.get(
        model, MODEL_PRICES[DEFAULT_LG_MODEL]
    )
    return (tokens_in * input_rate + tokens_out * output_rate) / 1_000_000


def neosyntropy_usd(committed: Sequence[str]) -> float:
    """Invoice USD for committed hops. Rejected steps must not be passed in."""
    total = 0.0
    for hop in committed:
        text = str(hop)
        source = text.split("->", 1)[0].strip() if "->" in text else text.strip()
        if not source or source == "End":
            continue
        if source in ROUTER_STATES:
            total += TUNED_TRANSITION_USD
        else:
            total += BASELINE_TRANSITION_USD
    return round(total, 6)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def score_trace(case: Case, trace: Trace) -> ScoredCase:
    if trace.error:
        return ScoredCase(
            case_id=case.id,
            system=trace.system,
            passed=False,
            route_ok=False,
            path_ok=False,
            schema_ok=False,
            tools_ok=False,
            illegal_hops=trace.illegal_hops,
            latency_ms=trace.latency_ms,
            tokens_in=trace.tokens_in,
            tokens_out=trace.tokens_out,
            llm_calls=trace.llm_calls,
            transitions=trace.transitions,
            usd=trace.usd,
            reason=trace.error,
        )

    route_ok = trace.landing == case.expected_route
    path_ok = is_subsequence(case.expected_path, trace.path)
    tools_ok = set(case.required_tools) <= set(trace.tools_ok) and not trace.tools_denied

    schema_ok = True
    reasons: list[str] = []
    for key, expected in case.expected_output.items():
        actual = trace.output.get(key)
        if key in {"analysis_report", "text", "value"}:
            if not str(actual or "").strip():
                schema_ok = False
                reasons.append(f"{key} empty")
        elif actual != expected:
            schema_ok = False
            reasons.append(f"{key}={actual!r} != {expected!r}")

    passed = (
        route_ok
        and path_ok
        and schema_ok
        and tools_ok
        and trace.illegal_hops == 0
    )
    if not route_ok:
        reasons.insert(0, f"route {trace.landing!r} != {case.expected_route!r}")
    if not path_ok:
        reasons.append(f"path {trace.path} missing {list(case.expected_path)}")
    if not tools_ok:
        reasons.append(
            f"tools {trace.tools_ok} missing {list(case.required_tools)}"
            + (f" denied={trace.tools_denied}" if trace.tools_denied else "")
        )
    if trace.illegal_hops:
        reasons.append(f"illegal_hops={trace.illegal_hops}")

    return ScoredCase(
        case_id=case.id,
        system=trace.system,
        passed=passed,
        route_ok=route_ok,
        path_ok=path_ok,
        schema_ok=schema_ok,
        tools_ok=tools_ok,
        illegal_hops=trace.illegal_hops,
        latency_ms=trace.latency_ms,
        tokens_in=trace.tokens_in,
        tokens_out=trace.tokens_out,
        llm_calls=trace.llm_calls,
        transitions=trace.transitions,
        usd=trace.usd,
        reason="; ".join(reasons),
    )


def summarize(system: str, rows: Sequence[ScoredCase]) -> SystemSummary:
    n = len(rows)
    passed = sum(1 for row in rows if row.passed)
    latencies = [row.latency_ms for row in rows]
    usd = sum(row.usd for row in rows)
    return SystemSummary(
        system=system,
        n=n,
        passed=passed,
        accuracy=(passed / n) if n else 0.0,
        p50_ms=_percentile(latencies, 0.50),
        p95_ms=_percentile(latencies, 0.95),
        mean_ms=float(statistics.fmean(latencies)) if latencies else 0.0,
        tokens_in=sum(row.tokens_in for row in rows),
        tokens_out=sum(row.tokens_out for row in rows),
        llm_calls=sum(row.llm_calls for row in rows),
        transitions=sum(row.transitions for row in rows),
        usd=round(usd, 6),
        usd_per_1k=round((usd / n) * 1000, 4) if n else 0.0,
        illegal_hops=sum(row.illegal_hops for row in rows),
    )


def format_table(summaries: Iterable[SystemSummary]) -> str:
    rows = list(summaries)
    header = (
        f"{'system':<14} {'n':>4} {'acc':>7} {'p50 ms':>8} {'p95 ms':>8} "
        f"{'calls':>6} {'tok in':>8} {'tok out':>8} {'tx':>4} "
        f"{'USD':>8} {'$/1k':>8} {'illegal':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.system:<14} {row.n:>4} {row.accuracy:>6.1%} {row.p50_ms:>8.0f} "
            f"{row.p95_ms:>8.0f} {row.llm_calls:>6} {row.tokens_in:>8} "
            f"{row.tokens_out:>8} {row.transitions:>4} {row.usd:>8.4f} "
            f"{row.usd_per_1k:>8.3f} {row.illegal_hops:>8}"
        )
    return "\n".join(lines)
