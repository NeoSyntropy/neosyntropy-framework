# NeoSyntropy concepts explained

This guide explains each core concept: **what it is**, **when to use it**, and a
**minimal example**. For methodology and wire contracts, see
[`concepts.md`](concepts.md).

**Models propose. The graph permits.**
A finite-state machine defines what is allowed; routers and nodes only propose
the next step. Nothing commits unless every gate passes.

```text
input + state
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
| Ground a step in stored documents | `Knowledge` + `retrieval_node` |
| Persist / query structured data | database adapter (`VectorDb`, Postgres, Mongo, …) |
| Anything else | dedicated `fallback` node |

---

## 1. `FSM` / `Workflow` — the permission surface

The graph is the single source of truth: nodes, edges, routers, groups, and a
required `entry` (node or router) whose `input_schema` is the workflow contract
for the run `input` dict.

- **`FSM(...)`** — full authoring (nodes, edges, routers, groups, entry).
- **`Workflow(nodes, fallback=...)`** — thin helper for simple linear graphs
  (first node becomes `entry`).

```python
auth = DeterministicRouter(
    id="CheckAuth",
    input_schema=RefundRequest,
    rules=[...],
)
graph = FSM(
    nodes=[...],
    routers=[auth],
    entry=auth,
    edges=[...],
)
```

`entry` is required. Its `input_schema` becomes `FSM.input_schema` and is checked
against the run `input` when the run starts at `entry.id`. Unknown keys are
refused; mid-workflow resumes are not re-checked (that state is what the
workflow itself produced). There is no synthetic `Start` state. Mutable
workflow `state` is separate from `input`.

**Use when:** you are defining the whole workflow and its entry contract.

### `fsm.run()` — argument reference

```python
result = fsm.run(
    EntryModel(...),   # positional — the run INPUT, validated against entry.input_schema
    state={...},       # keyword — the mutable workflow STATE bag
    client=client,     # keyword — NeoSyntropy Client (owns routing, inference, commits)
    tools=registry,    # keyword — ToolRegistry for nodes that declare tools=("...",)
)
```

| Argument | What it is | Inside a node |
|---|---|---|
| `request` (positional) | Typed run input — validated against `entry.input_schema` at run start. Immutable for the whole run. | `ctx.input` |
| `state=` | Mutable workflow bag — accumulates `state_updates` from every committed node. Pre-populate with auth context, ids, flags. | `ctx.state` |
| `client=` | `Client(api_key=..., project_id=...)` — the backend owns candidate selection, routing, plan validation, and commits. Omit to route locally. | — |
| `tools=` | `ToolRegistry` — nodes can only call tools declared in their `tools=(...)` **and** present here. Fail-closed. | `ctx.tools.invoke(...)` |

> [!NOTE]
> `input` is evidence, not authority. A request carrying `refund_approved=True` does not
> approve a refund — only a committed `state_update` from a node that passed every gate does.

### `Workflow` — linear graph helper

`Workflow` is a convenience constructor for creating simple linear graphs where nodes execute one after another in sequence. It automatically:
- Sets `entry` to the first node in the sequence.
- Wires each node to the next in the sequence using deterministic edges (`sequence[0] -> sequence[1] -> ... -> End`).
- Wires the required `fallback` node as a fallback edge from the first node.

```python
from neosyntropy import Workflow, node, OpenInput, EmptyOutput, TextOutput

@node(id="Step1", input_schema=OpenInput, output_schema=EmptyOutput)
def step1(ctx):
    return ctx.result(output={}, state_updates={"step1_done": True})

@node(id="Step2", input_schema=OpenInput, output_schema=EmptyOutput)
def step2(ctx):
    return ctx.result(output={}, state_updates={"step2_done": True}, next_state="End")

@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "Out of scope."})

# Automatically wires: Step1 -> Step2 -> End
# Fallback: Step1 -> OutOfScope
fsm = Workflow(
    sequence=[step1, step2],
    fallback=out_of_scope,
)
```

**Use when:** you have a simple, straight-line chain of execution and do not need branching routers, semantic routing, or manual edge definition.

---

## 2. `Node` / `@node` — executable capability

A **capability**, not a workflow position. Either:

- a Python handler (`@node` decorator), or
- a provider-backed constructor (`SchemaNode` / `ReasoningNode`).

The node returns a **proposal** — `output`, `state_updates`, optional `next_state`.
Nothing commits until every gate passes.

`input_schema` and `output_schema` are **required**. Use `OpenInput` when the node
does not need to constrain the workflow state.

```python
@node(
    id="VerifyIdentity",
    input_schema=OpenInput,    # required — what workflow state the node expects
    output_schema=EmptyOutput, # required — what the node produces
    prerequisites=(),          # optional — node ids that must have already run
)
def verify_identity(ctx):
    """Verify the caller owns the order."""
    return ctx.result(output={}, state_updates={"verified": True})
```

**Use when:** you own the logic in Python (auth checks, DB writes, API calls, anything
deterministic that the graph should execute as a side effect).

### Full FSM example

```python
from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client, FSM, OpenInput, EmptyOutput, TextOutput,
    node, edge_deterministic, edge_fallback,
)

client = Client(api_key="...", project_id="...")


class OrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    customer_id: str


@node(id="VerifyOwnership", input_schema=OrderInput, output_schema=EmptyOutput)
def verify_ownership(ctx):
    """Check the customer owns the order before doing any work."""
    verified = ctx.input["order_id"].startswith("ord_")
    return ctx.result(output={}, state_updates={"verified": verified})


@node(id="ApproveOrder", input_schema=OpenInput, output_schema=EmptyOutput)
def approve_order(ctx):
    return ctx.result(output={}, state_updates={"approved": True}, next_state="End")


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "Cannot process this request."})


fsm = FSM(
    entry=verify_ownership,
    nodes=[verify_ownership, approve_order, out_of_scope],
    edges=[
        edge_deterministic("VerifyOwnership", "ApproveOrder"),
        edge_deterministic("ApproveOrder", "End"),
        edge_fallback("VerifyOwnership", "OutOfScope"),
    ],
)

result = fsm.run(
    OrderInput(order_id="ord_123", customer_id="cust_42"),
    state={},
    client=client,
)
print(result.final_state)                      # "End"
print(result.audit.committed_transitions)
```

---

## 3. `SchemaNode` — constrained JSON (no tools)

Provider-backed extraction. The model must return JSON matching `output_schema`.
It has no tools.

All constructor arguments are **keyword-only**. `input_schema`, `output_schema`,
and `prompt` are required.

```python
SchemaNode(
    id="ExtractTicket",
    input_schema=OpenInput,        # required — what state the node reads
    output_schema=SupportTicket,   # required — constrained JSON contract
    prompt="Extract a support ticket as JSON from the customer message.",
)
```

**Use when:** the step's only job is to produce a typed structure (ticket,
classification, summary) from prior conversation or state.

### Full FSM example

```python
from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client, FSM, OpenInput, TextOutput,
    SchemaNode, node, edge_deterministic, edge_fallback,
)

client = Client(api_key="...", project_id="...")


class CustomerMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class SupportTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str
    issue: str
    priority: str


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "I can't help with that."})


extract_ticket = SchemaNode(
    id="ExtractTicket",
    input_schema=CustomerMessage,
    output_schema=SupportTicket,
    prompt=(
        "Extract a structured support ticket from the customer message. "
        "Set priority to 'high' if the customer mentions urgency or damage."
    ),
)

fsm = FSM(
    entry=extract_ticket,
    nodes=[extract_ticket, out_of_scope],
    edges=[
        edge_deterministic("ExtractTicket", "End"),
        edge_fallback("ExtractTicket", "OutOfScope"),
    ],
)

result = fsm.run(
    CustomerMessage(text="My order ord_99 arrived broken, need help urgently!"),
    state={},
    client=client,
)
print(result.final_state)
for step in result.steps:
    for item in step.results:
        # {"order_id": "ord_99", "issue": "arrived broken", "priority": "high"}
        print(item.output)
```

---

## 4. `ReasoningNode` — tools + plain-text notes

Provider-backed reasoning. The model calls allow-listed tools and writes
**plain-text notes** (not structured JSON). Its output feeds a later
`SchemaNode` or `CombineNode` that converts the notes into a typed artifact.

`input_schema`, `prompt`, and `tools` are **required**. Declaring an empty
`tools=()` raises `ValueError` — if you don't need tools, use `SchemaNode`.

```python
ReasoningNode(
    id="InvestigateRefund",
    input_schema=RefundInput,           # required
    prompt="Investigate the refund claim using tools. Reason step-by-step.",
    tools=(                             # required — at least one tool
        "lookup_customer_account",
        "fetch_transaction_history",
    ),
)
```

Tool loop (fail-closed):

1. Model emits `<TOOL:name>` — **no arguments**, just the trigger
2. Extractor fills a JSON object constrained by the tool's pydantic args schema
3. Validate → allow-list check → execute → reinject result as a `tool` message
4. Reasoning continues until the model stops emitting tool triggers
5. Ungranted / invalid tools **never run**; every attempt (ok, denied, failed)
   is recorded in `NodeResult.tool_calls`

**Use when:** a step needs to gather evidence (look up orders, check policies,
call APIs) before a later schema step produces the typed decision.

### Full FSM example

```python
from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client, FSM, OpenInput, TextOutput, ToolRegistry,
    ReasoningNode, SchemaNode, node,
    edge_deterministic, edge_fallback,
    tool,
)

client = Client(api_key="...", project_id="...")
registry = ToolRegistry()


# ── Input schema ─────────────────────────────────────────────────────────────
class RefundInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str   # customer's raw request; order/customer ids go into state={}


# ── Tool definitions ──────────────────────────────────────────────────────────
class LookupCustomerArgs(BaseModel):
    customer_id: str

@tool(registry=registry)
def lookup_customer_account(args: LookupCustomerArgs) -> dict:
    """Look up customer loyalty tier and account status."""
    return {"vip_tier": "Gold", "account_status": "active",
            "extended_return_window_days": 60}


class FetchTransactionArgs(BaseModel):
    order_id: str

@tool(registry=registry)
def fetch_transaction_history(args: FetchTransactionArgs) -> dict:
    """Fetch order details including amount and reported condition."""
    return {"amount": 1299.99, "purchase_days_ago": 35,
            "item_condition_reported": "defective"}


# ── Output schema (for the summarise step) ────────────────────────────────────
class RefundDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_valid: bool
    decision: str    # "approve" | "deny" | "escalate"
    rationale: str


# ── Nodes ─────────────────────────────────────────────────────────────────────
investigate = ReasoningNode(
    id="Investigate",
    input_schema=RefundInput,
    prompt=(
        "Investigate the refund claim by calling tools. "
        "Look up the customer account and fetch the transaction. "
        "Reason step-by-step about eligibility."
    ),
    tools=("lookup_customer_account", "fetch_transaction_history"),
)

summarise = SchemaNode(
    id="Summarise",
    input_schema=OpenInput,
    output_schema=RefundDecision,
    prompt=(
        "Using the prior reasoning and tool findings, produce a refund "
        "decision as JSON. Set claim_valid and decision with a rationale."
    ),
)

@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "I can't help with that."})


# ── Graph ─────────────────────────────────────────────────────────────────────
fsm = FSM(
    entry=investigate,
    nodes=[investigate, summarise, out_of_scope],
    edges=[
        edge_deterministic("Investigate", "Summarise"),
        edge_deterministic("Summarise", "End"),
        edge_fallback("Investigate", "OutOfScope"),
    ],
)

result = fsm.run(
    RefundInput(intent="Refund ord_98765 — laptop arrived defective"),
    state={"customer_id": "cust_12345", "order_id": "ord_98765"},
    client=client,
    tools=registry,
)
print(f"Final state : {result.final_state}")
print(f"Rejected    : {result.rejected}")

for step in result.steps:
    for item in step.results:
        print(f"\n=== {item.node_id} output ===")
        print(item.output)
        for record in item.tool_calls:
            verdict = "ok" if record.ok else ("denied" if record.denied else "failed")
            print(f"  [{verdict}] {record.tool} {record.arguments}")
```

---

## 5. `CombineNode` — reasoning then schema

Authoring shortcut that compiles into **two FSM states** at construction time:

```text
{id}           ← reasoning half (ReasoningNode + tools)
    │
    │  deterministic edge (auto-generated)
    ▼
{id}.Schema    ← schema half (SchemaNode → constrained JSON)
```

External edges **enter** `{id}` and **leave** `{id}.Schema`. The internal
linking edge is injected automatically — you never write it.

All constructor arguments are keyword-only. `id`, `input_schema`, `tools`,
`output_schema`, and `prompt` are **required**.

```python
CombineNode(
    id="InvestigateAndSummarise",
    input_schema=RefundInput,            # required
    tools=("lookup_customer_account",),  # required — at least one tool
    output_schema=RefundDecision,        # required — constrained JSON for {id}.Schema
    prompt="Investigate the claim with tools, then summarise.",  # required
    schema_prompt=None,       # optional — overrides the extraction prompt on {id}.Schema
                              # defaults to "Extract structured JSON from prior reasoning."
)
```

`CombineNode` **cannot be a fallback** — use a `SchemaNode(is_fallback=True)` instead.

**Use when:** one business step needs both tool-calling evidence gathering
and a typed JSON artifact, without manually wiring two nodes.

### Full FSM example

```python
from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client, FSM, OpenInput, TextOutput, ToolRegistry,
    CombineNode, node,
    edge_deterministic, edge_fallback,
    tool,
)

client   = Client(api_key="...", project_id="...")
registry = ToolRegistry()


# ── Schemas ───────────────────────────────────────────────────────────────────
class ClaimInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: str


class ClaimSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id:    str
    claim_valid: bool
    decision:    str     # "approve" | "deny" | "escalate"
    rationale:   str


# ── Tools ─────────────────────────────────────────────────────────────────────
class FetchOrderArgs(BaseModel):
    order_id: str

@tool(registry=registry)
def fetch_order(args: FetchOrderArgs) -> dict:
    """Fetch purchase date and item condition for the given order."""
    return {"purchase_days_ago": 20, "item_condition": "defective", "amount": 499.0}


# ── CombineNode expands to: InvestigateClaim  →  InvestigateClaim.Schema ──────
investigate_and_summarise = CombineNode(
    id="InvestigateClaim",
    input_schema=ClaimInput,
    tools=("fetch_order",),
    output_schema=ClaimSummary,
    prompt=(
        "Use the fetch_order tool to look up the order. "
        "Reason about whether the refund claim is valid."
    ),
    # schema_prompt is optional; default: "Extract structured JSON from prior reasoning."
)


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "I can't help with that."})


# ── Graph — enter at InvestigateClaim, leave from InvestigateClaim.Schema ─────
fsm = FSM(
    entry=investigate_and_summarise,         # targets the reasoning half
    nodes=[investigate_and_summarise, out_of_scope],
    edges=[
        # Exit from the schema half ("{id}.Schema") to End
        edge_deterministic("InvestigateClaim.Schema", "End"),
        edge_fallback("InvestigateClaim", "OutOfScope"),
    ],
)

result = fsm.run(
    ClaimInput(intent="Refund order ord_77 — item was defective"),
    state={"order_id": "ord_77", "customer_id": "cust_55"},
    client=client,
    tools=registry,
)
print(f"Final state : {result.final_state}")
for step in result.steps:
    for item in step.results:
        print(f"\n=== {item.node_id} ===")
        print(item.output)
```

---

## 5a. `State` — the mutable workflow bag

Every run has a **`state` dict** that persists and accumulates across every
node execution. It is the shared memory of the workflow.

- **Starts as** the `state={}` you pass to `fsm.run()`
- **Grows** as each committed node adds its `state_updates`
- **Is read** inside any node handler via `ctx.state`
- **Is never committed** until every gate passes — a rejected plan leaves
  state unchanged for that step

```text
fsm.run(input, state={"order_id": "ord_1"})
         │
         ▼
  VerifyOwnership  →  state_updates={"verified": True}
         │                     ↓ committed on success
         ▼
  state = {"order_id": "ord_1", "verified": True}
         │
         ▼
  CalculateRefund  →  state_updates={"refund_amount": 49.0}
         │                     ↓ committed on success
         ▼
  state = {"order_id": "ord_1", "verified": True, "refund_amount": 49.0}
```

**Every node type** — `@node` handlers, `SchemaNode`, `ReasoningNode`,
`CombineNode` — can propose state changes via `state_updates` in their
`NodeResult`. The control layer, not the node, decides when to commit them.

### Reading and writing state

```python
@node(id="CalculateRefund", input_schema=OpenInput, output_schema=EmptyOutput)
def calculate_refund(ctx):
    # READ from state (set by a prior node)
    amount = ctx.state.get("order_amount", 0.0)

    # PROPOSE a state change — not committed yet
    return ctx.result(
        output={},
        state_updates={"refund_amount": amount * 0.9},
    )
```

### State vs input

| | `input` | `state` |
|---|---|---|
| Set by | caller via `fsm.run(EntryModel(...), ...)` | caller pre-population + node `state_updates` |
| Validated | yes — against `entry.input_schema` at run start | no schema check |
| Mutated during run | never | yes — accumulates per committed step |
| Authority | none — evidence only | none — gates decide, not values |
| Accessed in node | `ctx.input["key"]` | `ctx.state["key"]` |

> [!NOTE]
> `input` is immutable evidence. Passing `{"refund_approved": True}` in input
> does not approve a refund. Only a node's committed `state_update`, after
> every gate passes, changes the workflow state.

### Full FSM example

```python
from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client, FSM, OpenInput, EmptyOutput, TextOutput,
    DeterministicRouter, node, edge_deterministic, edge_fallback,
)

client = Client(api_key="...", project_id="...")


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str


@node(id="LookupOrder", input_schema=OrderRequest, output_schema=EmptyOutput)
def lookup_order(ctx):
    """Fetch the order amount and write it into state."""
    # Simulate a DB call
    amount = 149.0
    return ctx.result(output={}, state_updates={"order_amount": amount, "eligible": amount > 0})


@node(id="IssueRefund", input_schema=OpenInput, output_schema=EmptyOutput)
def issue_refund(ctx):
    """Use refund_amount from state — set by CalculateRefund before this runs."""
    refund = ctx.state.get("refund_amount", 0)
    print(f"Issuing refund of ${refund:.2f}")
    return ctx.result(output={}, state_updates={"refund_issued": True}, next_state="End")


@node(id="DenyRefund", input_schema=OpenInput, output_schema=TextOutput)
def deny_refund(ctx):
    return ctx.result(output={"message": "Refund denied."}, next_state="End")


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "Cannot process request."})


# DeterministicRouter reads state set by LookupOrder
eligibility = DeterministicRouter(
    id="EligibilityGate",
    rules=[
        (lambda ctx: ctx.state.get("eligible") is True,  "IssueRefund"),
        (lambda ctx: ctx.state.get("eligible") is False, "DenyRefund"),
    ],
)

fsm = FSM(
    entry=lookup_order,
    nodes=[lookup_order, issue_refund, deny_refund, out_of_scope],
    routers=[eligibility],
    edges=[
        edge_deterministic("LookupOrder", "EligibilityGate"),
        edge_deterministic("EligibilityGate", "IssueRefund"),
        edge_deterministic("EligibilityGate", "DenyRefund"),
        edge_deterministic("IssueRefund", "End"),
        edge_deterministic("DenyRefund", "End"),
        edge_fallback("LookupOrder", "OutOfScope"),
    ],
)

result = fsm.run(
    OrderRequest(order_id="ord_42"),
    state={},          # starts empty — LookupOrder populates it
    client=client,
)
print(f"Final state : {result.final_state}")
print(f"Workflow state : {result.state}")
# result.state → {"order_amount": 149.0, "eligible": True, "refund_issued": True}
```

---

## 6. `Edge` — one permitted movement

The transition table is the **single source of permission**. A node result may
propose `next_state`; only a listed edge (or explicit
`allow_unlisted_transitions=True`) permits the move.

Three edge kinds — that's all. There are no labels, no priority tiers:

| Kind | Role | Helper |
|---|---|---|
| `deterministic` | Auto-commits when its guard passes. No router involved. | `edge_deterministic(src, tgt, guard=...)` |
| `semantic` | Scopes the semantic router to a node or group target. | `edge_semantic(src, tgt, target_kind=...)` |
| `fallback` | Safe stop when neither deterministic nor semantic yields a route. | `edge_fallback(src, tgt)` |

**You rarely construct `Edge` directly.** Author `DeterministicRouter` /
`SemanticRouter` objects instead — they compile into edges at `FSM` build time.
Hand-write edges only when you need a one-off guard or a bare `End` hop.

### Guards (optional)

An optional **guard** is a `callable(state: dict) -> bool` that gates the
transition at runtime. Guards are **fail-closed**: if the guard raises an
exception the transition is denied, never fail-open.

```python
from neosyntropy import Edge, edge_deterministic

# Bare Edge constructor (rarely needed):
Edge(
    source="CalculateRefund",
    target="IssueRefund",
    kind="deterministic",
    guard=lambda s: s.get("refund_amount", 0) > 0,  # guard is optional
)

# Equivalent helper (preferred):
edge_deterministic(
    "CalculateRefund",
    "IssueRefund",
    guard=lambda s: s.get("refund_amount", 0) > 0,
)
```

### Constraints

- `deterministic` and `fallback` edges must target a **node** (not a group)
- `semantic` edges may target a node **or** a group
- `fallback` edges are selected only when both deterministic and semantic
  routing miss — never mixed with actionable nodes in the same plan step

### Full FSM example

```python
from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client, FSM, OpenInput, EmptyOutput, TextOutput,
    node, edge_deterministic, edge_fallback,
)

client = Client(api_key="...", project_id="...")


class RefundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str


@node(id="CalculateRefund", input_schema=RefundRequest, output_schema=EmptyOutput)
def calculate_refund(ctx):
    amount = 49.0   # stub
    return ctx.result(output={}, state_updates={"refund_amount": amount})


@node(id="IssueRefund", input_schema=OpenInput, output_schema=EmptyOutput)
def issue_refund(ctx):
    return ctx.result(output={}, state_updates={"refund_issued": True}, next_state="End")


@node(id="DenyRefund", input_schema=OpenInput, output_schema=TextOutput)
def deny_refund(ctx):
    return ctx.result(output={"message": "Refund amount is zero — denied."}, next_state="End")


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "Cannot process request."})


fsm = FSM(
    entry=calculate_refund,
    nodes=[calculate_refund, issue_refund, deny_refund, out_of_scope],
    edges=[
        # Guard: only proceed to IssueRefund if refund_amount > 0
        edge_deterministic(
            "CalculateRefund", "IssueRefund",
            guard=lambda s: s.get("refund_amount", 0) > 0,
        ),
        # Guard: deny when amount is 0
        edge_deterministic(
            "CalculateRefund", "DenyRefund",
            guard=lambda s: s.get("refund_amount", 0) <= 0,
        ),
        edge_deterministic("IssueRefund", "End"),
        edge_deterministic("DenyRefund",  "End"),
        # Fallback: selected only when no deterministic edge matches
        edge_fallback("CalculateRefund", "OutOfScope"),
    ],
)

result = fsm.run(
    RefundRequest(order_id="ord_42"),
    state={},
    client=client,
)
print(result.final_state)
print(result.audit.committed_transitions)
```

---

## 7. `DeterministicRouter` — hard rules

First matching `(predicate, target)` rule wins. Compiles to deterministic edges
at `FSM` build time. **No model call — ever.**

Predicates receive a `ctx` object with `ctx.state` (the current workflow state
dict). They may also accept a raw `state` dict for compatibility.

```python
auth = DeterministicRouter(
    id="CheckAuth",
    input_schema=CustomerRequest,   # required when this router is the FSM entry
    rules=[
        (lambda ctx: ctx.state.get("token_valid") is True,  intent_router),
        (lambda ctx: ctx.state.get("token_valid") is False, login_node),
    ],
)
```

### Constraints

- **Targets** may be nodes, node ids, or other routers
- **Group targets are forbidden** — passing a `Group` as a target raises
  `ValueError` at construction time. Use `SemanticRouter` for group routes
- `input_schema` is optional unless the router is the FSM `entry` — then it
  is required (becomes the workflow entry contract)
- When running against the backend API, the SDK resolves unique deterministic
  hops **locally** so the backend always receives a concrete current state

**Use when:** compliance, auth, eligibility, policy — anything that must not be
"creative" and must be auditable without a model call.

### Full FSM example

```python
from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client, FSM, OpenInput, EmptyOutput, TextOutput,
    DeterministicRouter, node, edge_deterministic, edge_fallback,
)

client = Client(api_key="...", project_id="...")


class PolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    account_age_days: int


@node(id="CheckPolicy", input_schema=PolicyRequest, output_schema=EmptyOutput)
def check_policy(ctx):
    """Evaluate eligibility and write the result to state."""
    eligible = ctx.input["account_age_days"] >= 30
    return ctx.result(output={}, state_updates={"eligible": eligible})


@node(id="ApproveRequest", input_schema=OpenInput, output_schema=EmptyOutput)
def approve_request(ctx):
    return ctx.result(output={}, state_updates={"approved": True}, next_state="End")


@node(id="DenyRequest", input_schema=OpenInput, output_schema=TextOutput)
def deny_request(ctx):
    return ctx.result(output={"message": "Account too new — request denied."}, next_state="End")


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "Cannot process request."})


# Hard rule: reads ctx.state set by CheckPolicy
eligibility = DeterministicRouter(
    id="EligibilityGate",
    rules=[
        (lambda ctx: ctx.state.get("eligible") is True,  "ApproveRequest"),
        (lambda ctx: ctx.state.get("eligible") is False, "DenyRequest"),
    ],
)

fsm = FSM(
    entry=check_policy,
    nodes=[check_policy, approve_request, deny_request, out_of_scope],
    routers=[eligibility],
    edges=[
        edge_deterministic("CheckPolicy",     "EligibilityGate"),
        edge_deterministic("EligibilityGate", "ApproveRequest"),
        edge_deterministic("EligibilityGate", "DenyRequest"),
        edge_deterministic("ApproveRequest",  "End"),
        edge_deterministic("DenyRequest",     "End"),
        edge_fallback("CheckPolicy", "OutOfScope"),
    ],
)

result = fsm.run(
    PolicyRequest(text="I'd like a refund", account_age_days=45),
    state={},
    client=client,
)
print(result.final_state)                      # "End"
print(result.audit.committed_transitions)      # ["CheckPolicy", "EligibilityGate", "ApproveRequest"]
```

---

## 8. `SemanticRouter` — labeled intent routes

A model picks among **labeled** targets (`routes={label: node_or_group}`).
The proposal is validated against the graph before anything executes.

```python
intent = SemanticRouter(
    id="CustomerIntent",
    routes={
        "refund":  refunds_group,       # label → group (scoped to entry)
        "status":  order_status_node,   # label → node
        "billing": billing_group,
    },
    fallback_node=out_of_scope,         # cannot be a group
)
```

### Route limit (v1)

> [!IMPORTANT]
> In this version the router supports **up to 9 labeled routes** plus 1 fallback
> slot — 10 candidate slots total (`[0]..[9]` on the wire, index 9 reserved for
> the fallback / `UNSUPPORTED_OR_OUT_OF_SCOPE_INTENT`). Unlimited routes are
> planned for a future version.

### Execution topology

The backend router does not just pick one node — it returns a **plan** with a
topology that the `ControlManager` executes:

| Topology | What it means |
|---|---|
| `sequential` | Nodes run one after another; each sees the previous node's state updates |
| `parallel` | Nodes run in the same step; their `state_updates` must not conflict |
| `fallback` | Only the fallback node runs |

The model may propose `sequential` or `parallel` inside a single intent route.
For example, a "refund" route could run `ValidateEligibility` then `CalculateRefund`
sequentially, or run `LookupAccount` and `FetchOrder` in parallel before a
decision node.

```text
SemanticRouter  →  "refund"  →  [ValidateEligibility → CalculateRefund]  (sequential)
                →  "status"  →  [LookupOrder]                            (sequential)
                →  fallback  →  [OutOfScope]                             (fallback)
```

### Constraints

- `fallback_node` **cannot be a `Group`** — it must be a node or node id
- Group targets in `routes` are allowed; the hop lands on `group.entry` when set
- Targets may be nodes, groups, or other routers; duplicate route targets are
  caught at compile time

**Use when:** soft intent branching ("refund vs status vs chat").

**Do not use when:** unbounded agent tool loops — use `ReasoningNode` for that.

Offline / no backend (`PreferredPathRouter`): a unique semantic target is taken
automatically; ambiguous intents fall to the fallback.

### Full FSM example

```python
from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client, FSM, OpenInput, EmptyOutput, TextOutput,
    DeterministicRouter, SemanticRouter,
    node, edge_deterministic, edge_fallback,
)

client = Client(api_key="...", project_id="...")


class CustomerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    token: str


# ── Handler nodes ─────────────────────────────────────────────────────────────
@node(id="CheckAuth", input_schema=CustomerRequest, output_schema=EmptyOutput)
def check_auth(ctx):
    valid = ctx.input["token"].startswith("tok_")
    return ctx.result(output={}, state_updates={"token_valid": valid})


@node(id="ProcessRefund", input_schema=OpenInput, output_schema=EmptyOutput)
def process_refund(ctx):
    return ctx.result(output={}, state_updates={"refund_issued": True}, next_state="End")


@node(id="LookupOrderStatus", input_schema=OpenInput, output_schema=EmptyOutput)
def lookup_order_status(ctx):
    return ctx.result(output={}, state_updates={"status_fetched": True}, next_state="End")


@node(id="RequireLogin", input_schema=OpenInput, output_schema=TextOutput)
def require_login(ctx):
    return ctx.result(output={"message": "Please log in first."}, next_state="End")


@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "I can't help with that."})


# ── Routers ───────────────────────────────────────────────────────────────────
# DeterministicRouter: hard auth gate first
auth_gate = DeterministicRouter(
    id="AuthGate",
    input_schema=CustomerRequest,
    rules=[
        (lambda ctx: ctx.state.get("token_valid") is True,  "CustomerIntent"),
        (lambda ctx: ctx.state.get("token_valid") is False, "RequireLogin"),
    ],
)

# SemanticRouter: model picks the intent lane (max 9 routes in v1)
intent_router = SemanticRouter(
    id="CustomerIntent",
    routes={
        "refund":  process_refund,       # sequential: just one node here
        "status":  lookup_order_status,  # sequential: just one node here
    },
    fallback_node=out_of_scope,          # cannot be a group
)

# ── Graph ─────────────────────────────────────────────────────────────────────
fsm = FSM(
    entry=check_auth,
    nodes=[check_auth, process_refund, lookup_order_status, require_login, out_of_scope],
    routers=[auth_gate, intent_router],
    edges=[
        edge_deterministic("CheckAuth", "AuthGate"),
        edge_deterministic("AuthGate", "CustomerIntent"),
        edge_deterministic("AuthGate", "RequireLogin"),
        edge_deterministic("ProcessRefund", "End"),
        edge_deterministic("LookupOrderStatus", "End"),
        edge_fallback("CustomerIntent", "OutOfScope"),
    ],
)

result = fsm.run(
    CustomerRequest(text="I want a refund for order ord_42", token="tok_abc123"),
    state={},
    client=client,
)
print(result.final_state)
print(result.audit.committed_transitions)
```

---

## 9. `Group` — named subgraph (optional)

Organization plus an optional authored subgraph that merges into the parent
FSM:

- `@group.node(...)` to attach nodes
- `group.routers = [...]`
- `group.entry = "..."` — a semantic hop to the group lands here
- `group.add_edge(source, router_or_node)`  # deterministic router  

```python
billing = Group(name="billing")

@billing.node(id="ValidateCard", output_schema=EmptyOutput)
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
result = manager.run(RunRequest(input={"text": "refund my order"}, state={...}))
# or: result = fsm.run(EntryModel(...), state={...}, client=client)
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
  (schemas, node ids, prompts, edges — not handlers, state, or run input)
- **`graph_manifest(graph)`:** inspect the manifest payload
- Custom observers: `ControlManager(graph, observer=...)`

See [`examples/observability.py`](../examples/observability.py).

---

## 13. Databases — storage adapters

A **database** is a storage backend the graph can call. It is not a graph
vertex. Nodes, `Knowledge`, and `retrieval_node` use adapters; the FSM still
decides when those calls are allowed.

The package groups adapters by kind:

| Kind | Package | Typical use |
|---|---|---|
| Vector | [`neosyntropy.databases.vector`](../neosyntropy/databases/vector) | Semantic search (`VectorDb.search`) — Chroma, Qdrant, Pinecone, pgvector, … |
| Graph | [`neosyntropy.databases.graph`](../neosyntropy/databases/graph) | Relationship queries (Neo4j) |
| Relational | [`neosyntropy.databases.relational`](../neosyntropy/databases/relational) | SQL reads (Postgres) |
| Document | [`neosyntropy.databases.document`](../neosyntropy/databases/document) | Document stores (Mongo) |
| Object / blob | [`neosyntropy.databases.storage`](../neosyntropy/databases/storage) | Load files from S3, GCS, Azure Blob |

Vector stores share [`VectorDb`](../neosyntropy/databases/vector/base.py):
`insert` / `upsert` / `search`. That is the interface `Knowledge` and
`retrieval_node` call.

```python
from neosyntropy.databases.vector.chroma import ChromaDb
from neosyntropy.knowledge.document import Document

chroma = ChromaDb(collection="support_docs", persistent_client=True, path="./chroma")
chroma.upsert("policy", [Document(content="Renewals bill on the first of the month.")])
hits = chroma.search("when are renewals billed?", limit=5)
```

**Use when:** you need a concrete store. Prefer wrapping it in `Knowledge`
when several stores, loaders, or transforms share one corpus.

---

## 14. Knowledge — corpus and ETL

[`Knowledge`](../neosyntropy/knowledge/knowledge.py) is a named corpus over
one or more databases: vector stores, SQL/NoSQL readers, embedders, rerankers,
and an optional transform pipeline.

It implements three protocols:

- **`KnowledgeProtocol`** — `insert` / `delete` / `get` against registered stores
- **`KnowledgeTransformProtocol`** — `load` → `transform` (optional FSM) → `store`
- **`KnowledgeRetrievalProtocol`** — `search` / `asearch` (and optional retrieval FSM)

[`FileSystemKnowledge`](../neosyntropy/knowledge/filesystem.py) is the local
directory variant (grep / list / read). Loaders pull remote files into a
corpus: S3, GCS, Azure Blob, SharePoint, GitHub.

```python
from neosyntropy.knowledge import Knowledge
from neosyntropy.knowledge.document import Document
from neosyntropy.databases.vector.chroma import ChromaDb

kb = Knowledge(
    name="support_kb",
    vector_db=ChromaDb(collection="support_docs"),
)
kb.insert([Document(content="Grace period: 7 days after the due date.")])
docs = kb.search("grace period", limit=5)
```

Transform one corpus into another (chunk, summarize, re-embed):

```python
from neosyntropy.knowledge import Knowledge

summaries = Knowledge(transform=summarize_documents, name="summaries")
summaries.transform(source=source_kb, destination=dest_kb)
```

Cookbook: [`retrieval`](../cookbook/knowledge/retrieval_example.py) ·
[`transform`](../cookbook/knowledge/transform_example.py)

**Use when:** nodes should read a shared, versioned corpus rather than
embedding store credentials and query logic in every handler.

---

## 15. `retrieval_node` — knowledge into state

[`retrieval_node`](../neosyntropy/core/node/retrieval.py) is an FSM node that
reads a query from workflow state, searches a `Knowledge` base or `VectorDb`,
and writes the hits back into state. Retrieval is a capability on the graph,
not an agent choosing tools.

```python
from neosyntropy import retrieval_node, node, OpenInput, EmptyOutput
from neosyntropy.knowledge.filesystem import FileSystemKnowledge

kb = FileSystemKnowledge(base_dir="./policies")

fetch_context = retrieval_node(
    id="FetchContext",
    knowledge=kb,          # or vector_db=chroma
    query_key="query",
    output_key="context",
    limit=5,
)

@node(id="Answer", prerequisites=("FetchContext",), input_schema=OpenInput, output_schema=EmptyOutput)
def answer(ctx):
    context = ctx.state["context"]   # list of {content, meta_data}
    return ctx.result(output={}, state_updates={"answered": True}, next_state="End")
```

Pass `format_as_string=True` when a later `ReasoningNode` or `SchemaNode`
should receive one concatenated block instead of a list of dicts.

**Use when:** a graph step must ground itself in stored documents before a
model or handler runs. Keep store credentials on `Knowledge`; keep permission
on the FSM.

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
        retrieval_node        ← Knowledge / VectorDb → state
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
3. `refunds` group entry → `FetchPolicy` (`retrieval_node` + `Knowledge`) → `InvestigateRefund` (`ReasoningNode` + tools)
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
- [`cookbook/knowledge`](../cookbook/knowledge) — `FileSystemKnowledge` search and transform
- [`neosyntropy/databases`](../neosyntropy/databases) — vector, graph, relational, document, and object-store adapters
- [`retrieval_node`](../neosyntropy/core/node/retrieval.py) — inject search hits into FSM state
