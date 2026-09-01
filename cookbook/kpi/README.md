# KPI cookbook

KPI nodes **score** a run path.  They always proceed to the next step — they
never fail the run.  If a score threshold must gate execution, place a
`functional_validation_node` *after* the KPI node and branch on
`state["valid"]`.

KPI can be inserted at **three levels** of the FSM hierarchy, each using the
same `SemanticXxx` / `functional_xxx` factory convention.  All levels produce
nodes whose `output_schema` is `KpiResult {"name": str, "score": float, "reason": str}`.

| Level | When to use | Writes state |
|---|---|---|
| **Node** | Score a single node's output mid-path | `functional_kpi_node` yes · `SemanticKpiNode` no (output only) |
| **Group** | Score the outcome of a whole group's internal path | same as above |
| **FSM** | Score the entire run before `End` | `functional_fsm_path_kpi` yes · `SemanticFSMPathKpi` no |

## Examples

- `node_kpi_example.py` — `functional_kpi_node` scores a schema extraction
  result (headline + bullet quality) and writes `state["summary_quality"]`
- `group_kpi_example.py` — `functional_group_path_kpi` auto-registers into
  a triage group and scores urgency + team + email completeness
- `fsm_path_kpi_example.py` — `functional_fsm_path_kpi` uses
  `extract_fsm_path(ctx)` to inspect the full execution history and compute a
  composite quality score before `End`

## Run

```bash
python cookbook/kpi/node_kpi_example.py
python cookbook/kpi/group_kpi_example.py
python cookbook/kpi/fsm_path_kpi_example.py
```

Credentials are loaded from `tests/.env`.  Set `NEOSYNTROPY_PROVIDER` to
override the inference provider (default: `gemini-2.5-flash`).

## Quick reference

```python
from neosyntropy.core.kpi import (
    # Node level
    SemanticKpiNode,
    functional_kpi_node,
    # Group level
    SemanticGroupPathKpi,
    functional_group_path_kpi,
    # FSM level
    SemanticFSMPathKpi,
    functional_fsm_path_kpi,
    # Shared path helpers (re-exported from validation)
    FSMPathInfo,
    extract_fsm_path,
)
```

### Node level

```python
# LLM scorer — output: {"name": str, "score": float, "reason": str}
quality = SemanticKpiNode(
    "answer_quality",
    input_schema=QueryInput,
    prompt="Score the answer quality on a scale of 0–1. name='answer_quality'.",
)

# Python handler — also writes state["completeness"] and state["kpis"]
@functional_kpi_node(id="completeness", input_schema=State, output_key="completeness")
def completeness(ctx: NodeContext) -> KpiResult:
    ran = set(ctx.state.get("steps_ran", []))
    required = {"StepA", "StepB", "StepC"}
    score = len(required & ran) / len(required)
    return KpiResult(name="completeness", score=score, reason=f"ran={sorted(ran & required)}")
```

### Group level

```python
billing_group = Group(name="billing")
# ... add nodes and internal edges ...

# Auto-registers into the group; wires ProcessPayment → billing_quality
@functional_group_path_kpi(
    group=billing_group,
    after="ProcessPayment",
    input_schema=BillingState,
    output_key="billing_quality",
)
def billing_quality(ctx: NodeContext) -> KpiResult:
    confirmed = ctx.state.get("payment_confirmed", False)
    return KpiResult(
        name="billing_quality",
        score=1.0 if confirmed else 0.0,
        reason=f"confirmed={confirmed}",
    )
```

### FSM level

```python
# Placed as the last node before End; inspect the full execution history
@functional_fsm_path_kpi(id="PathScore", input_schema=FinalState)
def path_score(ctx: NodeContext) -> KpiResult:
    path = extract_fsm_path(ctx)        # FSMPathInfo
    required = {"ParseQuery", "GenerateAnswer"}
    coverage = len(required & set(path.nodes_executed)) / len(required)
    return KpiResult(name="path_quality", score=coverage, reason=f"path={path.nodes_executed}")

# Wire straight to End — no guard needed
edges=[
    edge_deterministic("LastStep", "PathScore"),
    edge_deterministic("PathScore", "End"),    # always continues
]
```

### Combining KPI with validation (threshold gate)

```python
# 1. Score the path
@functional_fsm_path_kpi(id="PathScore", input_schema=State, output_key="path_score")
def path_score(ctx: NodeContext) -> KpiResult: ...

# 2. Gate on the score
@functional_validation_node(id="ScoreThreshold", input_schema=State)
def score_threshold(ctx: NodeContext) -> bool:
    return float(ctx.state.get("path_score", 0.0)) >= 0.7

# Edges
edge_deterministic("LastStep", "PathScore"),
edge_deterministic("PathScore", "ScoreThreshold"),
edge_deterministic("ScoreThreshold", "End",       guard=lambda s: bool(s.get("valid"))),
edge_deterministic("ScoreThreshold", "Fallback",  guard=lambda s: not bool(s.get("valid", True))),
```
