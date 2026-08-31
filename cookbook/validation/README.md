# Validation cookbook

Validation can be inserted at **three levels** of the FSM hierarchy.  Each
level uses the same `SemanticXxx` / `functional_xxx` factory convention and
always produces a node whose `output_schema` is `ValidationResult`
`{"valid": bool, "reason": str}`.

| Level | When to use | Writes `state["valid"]` |
|---|---|---|
| **Node** | Gate a single node's output mid-path | `functional_validation_node` yes · `SemanticValidationNode` no (output only) |
| **Group** | Gate the outcome of a whole group's internal path | same as above |
| **FSM** | Gate the entire run before `End` | `functional_fsm_path_validator` yes · `SemanticFSMPathValidator` no (output only) |

## Examples

- `node_validation_example.py` — `functional_validation_node` checks a
  schema extraction result; branches on `state["valid"]` to `End` or fallback
- `group_path_validation_example.py` — `SemanticGroupPathValidator` (LLM
  judge) runs as the terminal node inside a triage group; output available in
  `NodeResult.output`
- `fsm_path_validation_example.py` — `functional_fsm_path_validator` uses
  `extract_fsm_path(ctx)` to inspect the full execution history and enforce
  that all required steps ran before the workflow completes

## Run

```bash
python cookbook/validation/node_validation_example.py
python cookbook/validation/group_path_validation_example.py
python cookbook/validation/fsm_path_validation_example.py
```

Credentials are loaded from `tests/.env`.  Set `NEOSYNTROPY_PROVIDER` to
override the inference provider (default: `gemini-2.5-flash`).

## Quick reference

```python
from neosyntropy.core.validation import (
    # Node level
    SemanticValidationNode,
    functional_validation_node,
    # Group level
    SemanticGroupPathValidator,
    functional_group_path_validator,
    # FSM level
    SemanticFSMPathValidator,
    functional_fsm_path_validator,
    FSMPathInfo,
    extract_fsm_path,
)
```

### Node level

```python
# LLM judge — output: {"valid": bool, "reason": str}
guard = SemanticValidationNode(
    "sql_safety_check",
    input_schema=QueryInput,
    prompt="Return valid=false if the SQL contains DROP or DELETE without WHERE.",
)

# Python handler — also writes state["valid"] and state["valid_reason"]
@functional_validation_node(id="length_check", input_schema=SummaryOutput)
def length_check(ctx: NodeContext) -> bool:
    return len(ctx.state.get("bullets", [])) >= 2
```

### Group level

```python
billing_group = Group(name="billing")
# ... add nodes and internal edges ...

# Auto-registers into the group; wires ClassifyUrgency → ValidateTriage
SemanticGroupPathValidator(
    "ValidateTriage",
    group=billing_group,
    after="ClassifyUrgency",
    input_schema=TriageResult,
    prompt="Verify the triage result is coherent.",
)

# Or the functional variant (writes state)
@functional_group_path_validator(group=billing_group, after="ClassifyUrgency")
def validate_triage(ctx: NodeContext) -> bool:
    return ctx.state.get("urgency") in ("low", "normal", "high")
```

### FSM level

```python
# Placed as the last node before End; inspect the full execution history
@functional_fsm_path_validator(id="AuditPath", input_schema=FinalState)
def audit_path(ctx: NodeContext) -> ValidationResult:
    path = extract_fsm_path(ctx)          # FSMPathInfo
    missing = {"StepA", "StepB"} - set(path.nodes_executed)
    if missing:
        return ValidationResult(valid=False, reason=f"missing: {sorted(missing)}")
    return ValidationResult(valid=True)
```
