# NeoSyntropy concepts explained

This guide explains each core concept: **what it is**, **when to use it**, and a
**minimal example**. For methodology and wire contracts, see
[`concepts.md`](concepts.md).

**Models propose. The graph permits.**
A finite-state machine defines what is allowed; routers and nodes only propose
the next step. Nothing commits unless every gate passes.

```text
intent + state
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│ candidates  │ ──▶ │ router plan  │ ──▶ │ validate   │
└─────────────┘     └──────────────┘     └─────┬──────┘
                                               │
                    ┌────────────┐     ┌───────▼──────┐
                    │  commit    │ ◀── │   execute    │
                    │  + audit   │     │   nodes      │
                    └────────────┘     └──────────────┘
```

---

## Mental model

| Idea | Meaning |
|---|---|
| Proposal ≠ permission | Nodes and routers suggest; edges + validators decide |
| One current state | Exactly one FSM position per run; at most one commit per step |
| Fail-closed | Missing edge, failed guard, bad input → reject, no commit |
| Nodes ≠ positions | A node is a capability; the graph state is where you are |
| Tools ≠ vertices | Tools hang off nodes; they never appear as graph states |
| Fallback is isolated | Exactly one dedicated fallback; never mixed with actionable work |

Determinism here is **architectural** (graphs + gates), not `temperature=0`.

---

## Which primitive should I use?

| Decision type | Primitive |
|---|---|
| Must be exact (auth, money, policy) | `DeterministicRouter` or edge `guard` |
| Soft intent among known lanes | `SemanticRouter` |
| Gather evidence with tools | `ReasoningNode` |
| Emit typed structure | `SchemaNode` |
| Both gather + emit | `CombineNode` |
| Side effects in your code | `@node` handler |
| Anything else | dedicated `fallback` node |

---

## 1. `FSM` / `Workflow` — the permission surface

The graph is the single source of truth: nodes, edges, routers, groups, and an
optional `input_schema` for entry at `Start`.

- **`FSM(...)`** — full authoring (nodes, edges, routers, groups, entry).
- **`Workflow(nodes, fallback=...)`** — thin helper for simple linear graphs.

```python
graph = FSM(
    nodes=[...],
    routers=[auth],
    entry=auth,
    input_schema=RefundRequest,
    edges=[...],
)
```

`input_schema` is checked only when the run starts at `Start`. Unknown keys are
refused; mid-workflow resumes are not re-checked (that state is what the
workflow itself produced).

**Use when:** you are defining the whole workflow and its entry contract.

---

## 2. `Node` / `@node` — executable capability

A **capability**, not a workflow position. Either:

- a Python handler (`@node`), or
- a provider-backed constructor (`SchemaNode` / `ReasoningNode`).

Returns a **proposal**: `output`, `state_updates`, optional `next_state`.
Nothing commits until gates pass.

```python
@node(id="VerifyIdentity", prerequisites=(), output_schema=EmptyOutput)
def verify_identity(ctx):
    return ctx.result(output={}, state_updates={"verified": True})
```

**Use when:** you own the logic in code (auth side-effects, payouts, API calls).

---

## 3. `SchemaNode` — constrained JSON (no tools)

Provider-backed extraction. The model must return JSON matching `output_schema`.
It has no tools.

```python
SchemaNode(
    id="Ticket",
    input_schema=OpenInput,
    output_schema=SupportTicket,
    prompt="Extract a support ticket as JSON.",
)
```

**Use when:** the step’s only job is typed structure (ticket, classification,
summary).

---

## 4. `ReasoningNode` — tools + plain-text notes

Provider-backed reasoning. May call allow-listed tools; writes notes, not
free-form business state.

```python
ReasoningNode(
    id="Investigate",
    input_schema=OpenInput,
    prompt="Look up the order and note eligibility.",
    tools=("lookup_order",),
)
```

Tool loop (fail-closed):

1. Model emits `<TOOL:name>` (no args)
2. Extractor fills JSON from the tool’s pydantic schema
3. Validate → allow-list → execute → reinject
4. Ungranted / invalid tools never run; every attempt is audited in
   `NodeResult.tool_calls`

**Use when:** evidence gathering or short deliberation before a hard decision
or schema step.

---

## 5. `CombineNode` — reasoning then schema

Authoring sugar that expands to two FSM states:

```text
{id}  (ReasoningNode + tools)  →  {id}.Schema  (SchemaNode)
```

External edges **enter** `{id}` and **leave** `{id}.Schema`.

```python
CombineNode(
    id="Clearance",
    input_schema=OpenInput,
    tools=("lookup_order",),
    output_schema=SupportTicket,
    prompt="Gather evidence, then extract a ticket.",
)
```

**Use when:** one business step needs both tool use and a typed artifact.

---

## 6. `Edge` — one permitted movement

Kinds:

| Kind | Role |
|---|---|
| `deterministic` | Hard path / rule outcome |
| `semantic` | Model may propose this hop (still validated) |
| `fallback` | Safe stop when nothing else applies |

Optional **guards** (`callable(state) -> bool`) fail closed — a guard that
raises denies the transition.

```python
Edge(
    source="CalculateRefund",
    target="IssueRefund",
    kind="deterministic",
    guard=lambda s: s.get("refund_amount", 0) > 0,
)
```

You usually author routers instead of hand-writing every edge; routers compile
into edges. The transition table remains the single source of permission: a
node may propose `next_state`, but only a listed edge (or explicit
`allow_unlisted_transitions=True`) permits the move.

---

## 7. `DeterministicRouter` — hard rules

First matching `(predicate, target)` wins. Compiles to deterministic edges.
No model call.

```python
auth = DeterministicRouter(
    id="CheckAuth",
    rules=[
        (lambda ctx: ctx.state.get("token_valid") is True, intent),
        (lambda ctx: ctx.state.get("token_valid") is False, login),
    ],
)
```

**Use when:** compliance, auth, eligibility, policy — anything that must not
be “creative”.

Targets may be nodes or other routers. Group targets belong on
`SemanticRouter` (with an optional group `entry`).

When running against the backend control API, the SDK resolves unique
deterministic hops locally so the backend receives a concrete current state.

---

## 8. `SemanticRouter` — labeled intent routes

A model picks among **labeled** targets. The proposal is still validated
against the graph.

```python
intent = SemanticRouter(
    id="CustomerIntent",
    routes={
        "refund": refunds_group,
        "status": order_status_node,
    },
    fallback_node=out_of_scope,
)
```

**Use when:** soft intent branching (“refund vs status vs chat”).

**Do not use when:** unbounded agent tool loops — that belongs on a
`ReasoningNode`.

Offline / no backend (`PreferredPathRouter`): a unique semantic target may be
taken automatically; ambiguous intents fall to the fallback.

---

## 9. `Group` — named subgraph (optional)

Organization plus an optional authored subgraph that merges into the parent
FSM:

- `@group.node(...)` to attach nodes
- `group.routers = [...]`
- `group.entry = "..."` — a semantic hop to the group lands here
- `group.add_edge(source, router_or_node)`

```python
billing = Group(name="billing")

@billing.node(id="ValidateCard", input_schema=OpenInput, output_schema=EmptyOutput)
def validate(ctx):
    return ctx.result(output={}, state_updates={"card_valid": True})

logic = DeterministicRouter(
    id="BillingLogic",
    rules=[
        (lambda ctx: ctx.state.get("card_valid") is True, "ProcessPayment"),
        (lambda ctx: ctx.state.get("card_valid") is False, "RejectCard"),
    ],
)
billing.routers = [logic]
billing.entry = "ValidateCard"
billing.add_edge("ValidateCard", "BillingLogic")
```

At runtime there is still **one** control engine — groups are not nested
runtimes. Compiled edges feed the same `ControlManager` pipeline.

---

## 10. `tool` / `ToolRegistry` — capabilities on nodes

```python
from pydantic import BaseModel
from neosyntropy import tool

class LookupOrderArgs(BaseModel):
    order_id: str

@tool
def lookup_order(args: LookupOrderArgs) -> dict:
    """Look up an order by id."""
    ...

@node(id="Verify", tools=("lookup_order",), output_schema=EmptyOutput)
def verify(ctx):
    order = ctx.tools.invoke("lookup_order", {"order_id": "ord_1"})
    return ctx.result(output={}, state_updates={"amount": order["amount"]})
```

- Declared on the node (`tools=(...)`)
- Invoked only through `ctx.tools` (allow-list enforced fail-closed)
- Never a graph vertex

For provider-backed nodes, only the node’s own tools appear in its prompt.
The default parameter extractor uses the same provider; plug a trained edge
extractor in with `ControlManager(graph, extractor=...)`.

---

## 11. `ControlManager` — one control cycle

```text
input schema → candidates → router proposal → validate
           → execute → guards / transitions → commit → AuditRecord
```

Guarantees:

- **Proposal is not permission** — validators and the transition table decide
- **One current state** — at most one atomic commit per plan step
- **Fail-closed before commit** — high confidence + successful execution still
  cannot advance illegally
- **Fallback isolation** — exactly one dedicated fallback; never mixed with
  actionable nodes
- **Auditability by construction** — every cycle emits an `AuditRecord`

A rejection is a normal outcome: `result.rejected` is set, nothing was
committed for the offending step, and the audit explains why.

```python
manager = ControlManager(graph, tools=registry)
result = manager.run(RunRequest(intent="refund my order", state={...}))
print(result.final_state)
print(result.audit.committed_transitions)
```

With backend credentials configured, the **backend owns** candidate selection,
routing, plan validation, and commits. The client defines the graph, runs local
handlers, and submits results. Responses never include topology, candidates,
execution plans, providers, or model names.

Set `NEOSYNTROPY_API_URL` with `NEOSYNTROPY_API_KEY` + `NEOSYNTROPY_PROJECT_ID`
(or `NEOSYNTROPY_ACCESS_TOKEN`). `ControlManager(graph)` discovers them
automatically.

---

## 12. Observability

When a backend client is configured, `ControlManager` reports lifecycle events
to the telemetry API. Telemetry is bounded and best-effort: an unavailable or
slow observer never changes execution, validation, commits, or raised errors.

- **Default:** capture run/step payloads so the console can replay the FSM
- **`capture_payloads=False`:** lifecycle + visualization manifest only
  (schemas, node ids, prompts, edges — not handlers, state, or intent)
- **`graph_manifest(graph)`:** inspect the manifest payload
- Custom observers: `ControlManager(graph, observer=...)`

See [`examples/observability.py`](../examples/observability.py).

---

## How the concepts compose

```text
                    ┌─ DeterministicRouter ─┐
Start ──entry──▶   │   (hard rules)        │
                    └──────────┬────────────┘
                               │
                    ┌──────────▼────────────┐
                    │   SemanticRouter      │  model picks a label
                    │   routes={...}        │
                    └──────────┬────────────┘
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
            Group           SchemaNode      Fallback
         (entry node)      (JSON out)     (OutOfScope)
               │
        ReasoningNode / CombineNode
               │
        DeterministicRouter   ← eligibility / payout rules
               │
          @node handlers      ← IssueRefund / DenyRefund
```

Example shape for a support desk:

1. `CheckAuth` (`DeterministicRouter`) — token valid or force login
2. `CustomerIntent` (`SemanticRouter`) — refund / status / fallback
3. `refunds` group entry → `InvestigateRefund` (`ReasoningNode` + tools)
4. `RefundLogic` (`DeterministicRouter`) — eligible → issue, else deny
5. `OutOfScope` — dedicated fallback when nothing matches

---

## Related docs

- [`concepts.md`](concepts.md) — methodology, fail-closed gates, SLM wire contracts
- Site concepts: [nodes](https://docs.neosyntropy.com/concepts/nodes) ·
  [model-backed nodes](https://docs.neosyntropy.com/concepts/model-nodes) ·
  [routers](https://docs.neosyntropy.com/concepts/routers) ·
  [edges](https://docs.neosyntropy.com/concepts/edges) ·
  [groups](https://docs.neosyntropy.com/concepts/groups) ·
  [control manager](https://docs.neosyntropy.com/concepts/control-manager)
- [`examples/refund_workflow.py`](../examples/refund_workflow.py)
- [`examples/model_tool_calling.py`](../examples/model_tool_calling.py)
