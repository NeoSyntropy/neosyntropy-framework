# NeoSyntropy Concepts

NeoSyntropy separates probabilistic decisions from deterministic control.
Models can propose what should happen next; a finite-state graph defines what
is allowed to happen. This document distills the methodology the framework
enforces.

## The mental model

Agent frameworks are black boxes: decisions emerge from prompts, the action
space is unbounded, and compliance becomes forensics after the fact.
NeoSyntropy inverts this. You decompose business logic into **states, edges,
and executable nodes**, and the engine enforces that map on every
cycle. Invalid transitions are rejected before side effects; compliance holds
by construction because every accepted step is a logged state change.

Determinism here is architectural, not `temperature=0`. Explicit graphs,
transition tables, plan validation, and edge/transition gates constrain what AI may
choose *before* anything executes. Routing by a model is allowed — but only
inside a machine-checkable plan over a precompiled graph. Determinism is
bounded nondeterminism under gates, not "same tokens every time".

## The primitives

The model-backed constructors are how developers put tiny AI models into
application code. Each node gets a small, scoped prompt and a fixed contract;
the control layer — not the model — owns transitions and commits.

### SchemaNode — constrained JSON

`SchemaNode` is provider-backed schema extraction: the model must return JSON
that matches `output_schema`. It has no tools. Use it when the step's job is
to produce a typed structure (ticket, summary, classification) from state or
prior notes.

```python
SchemaNode(
    id="Ticket",
    input_schema=OpenInput,
    output_schema=SupportTicket,
    prompt="Extract a support ticket as JSON.",
)
```

### ReasoningNode — tools + notes

`ReasoningNode` is provider-backed reasoning: the model may call allow-listed
tools and writes plain-text notes (not free-form business state). Use it when
the step needs evidence gathering or short deliberation before a later schema
step.

```python
ReasoningNode(
    id="Support",
    input_schema=OpenInput,
    prompt="Help the customer with their order.",
    tools=("lookup_order",),
)
```

Provider-backed nodes that declare tools run a split reasoning/extraction
loop, because guessing arguments and choosing actions are different problems:

1. **Trigger** — the reasoner emits `<TOOL:tool_name>` with no arguments.
2. **Extract** — a parameter extractor fills a JSON object constrained by the
   tool's pydantic schema.
3. **Validate** — arguments are checked against the args model *before* the
   tool runs.
4. **Execute** — the call passes through the node's allow-list.
5. **Reinject** — the outcome returns as a `tool` message and reasoning
   continues.

Here too, proposal is not permission: a model asking for a tool the node does
not declare is denied and told so, and the tool never executes. Deterministic
code calling an undeclared tool is a different thing — a programming error —
and rejects the cycle outright. Every attempt (denied, failed, or successful)
is recorded in `NodeResult.tool_calls`, so the audit trail stays complete.

### CombineNode — reasoning then schema

`CombineNode` is an authoring unit that expands into two FSM states: entry
`{id}` (reasoning + tools) then `{id}.Schema` (constrained JSON). External
edges should target `{id}` and leave from `{id}.Schema`.

```python
CombineNode(
    id="Clearance",
    input_schema=OpenInput,
    tools=("lookup_order",),
    output_schema=SupportTicket,
    prompt="Gather evidence, then extract a ticket.",
)
```

### Node — executable capability

A node packages a capability: a Python handler (`@node`) or a provider-backed
prompt (`SchemaNode` / `ReasoningNode`), declared tools, prerequisites, and an
optional group. The router may select one or several nodes per cycle.
**Nodes are not workflow positions**: running three nodes in parallel does not
create three states.

Nodes return a `NodeResult` — output, `state_updates`, and an optional
`next_state`. That result is a proposal. Nothing a node returns commits
anything.

Tools are capabilities *on* a node (`tools=("lookup_order",)`), never graph
vertices. Handlers reach tools through a bound facade that enforces the
allow-list fail-closed: calling an undeclared tool is denied fail-closed, not
an error to retry around.

### Edge — one permitted movement

The transition table is the single source of permission. A node result may
propose `next_state`; only a listed edge (or an explicit
`allow_unlisted_transitions=True`) permits the move. Missing edges do not
skip selection — search finds relevant nodes, and validation fail-closes.

Edges carry labels with a fixed priority order
(`load < first < next < inferred-next < complete < return < route < conditional`)
that drives the deterministic preferred-path router, and optional **guards**:
callables over the state that gate the edge at runtime. Guards fail closed —
a guard that raises denies the transition.


### Group — organization (and optional authored subgraph)

Groups name collections of nodes for organization and candidate metadata.
A semantic edge may target a group to scope routing to that group's nodes.
When a group declares ``entry``, entering the group lands on that entry
state instead of offering every member as a candidate.

Groups can also author an internal subgraph that compiles into the parent
FSM — nodes via ``@group.node``, internal ``DeterministicRouter`` /
``SemanticRouter`` units, an ``entry``, and ``add_edge`` links from nodes
to routers. At runtime there is still one control path: the compiled edges
feed the same ControlManager pipeline; the group is not a second engine.

```python
billing = Group(name="billing")

@billing.node(id="ValidateCard", input_schema=OpenInput, output_schema=EmptyOutput)
def validate(ctx):
    return ctx.result(output={}, state_updates={"card_valid": True})

@billing.node(id="ProcessPayment", input_schema=OpenInput, output_schema=EmptyOutput)
def pay(ctx):
    return ctx.result(output={})

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

### DeterministicRouter — hard rules

`DeterministicRouter` encodes hard business rules: the first matching
`(predicate, target)` wins. Predicates see run context/state; targets may be
nodes, groups, or other routers. At compile time the unit becomes
deterministic edges the control cycle can follow without calling a model.

```python
auth = DeterministicRouter(
    id="CheckAuth",
    rules=[
        (lambda ctx: ctx.state.get("token_valid") is True, intent_router),
        (lambda ctx: ctx.state.get("token_valid") is False, login_node),
    ],
)
```

Guards stay local: when running against the backend control API, the SDK
resolves unique deterministic hops before the remote cycle so the backend
receives a concrete current state.

### SemanticRouter — labeled intent routes

`SemanticRouter` lets a model choose among labeled targets
(`routes={label: node_or_group}`) with an optional `fallback_node`. The model
proposes a label; validators and the transition table still decide whether
that hop is legal. Use it for intent branching, not for unconstrained
tool-using agents.

```python
intent = SemanticRouter(
    id="CustomerIntent",
    routes={
        "wants_to_pay": billing_group,
        "needs_support": support_group,
    },
    fallback_node=general_chat,
)
```

### ControlManager — the pipeline as one object

```text
input -> candidate selection -> router proposal -> plan validation
      -> execution -> guards / transition checks -> one state commit
      -> audit record
```

Rules the manager guarantees:

- **Proposal is not permission.** The router (deterministic or SLM) only
  proposes `{reasoning, topology, execution_plan}`. The validator and the
  transition table decide.
- **Exactly one current state** per workflow instance; at most one state
  commit per plan step, applied atomically. Parallel nodes may not propose
  conflicting next states or conflicting values for the same key.
- **Fail-closed gates before commit.** High router confidence plus successful
  execution still cannot advance illegally.
- **Fallback isolation.** Every graph has exactly one dedicated fallback
  node — a safe stop when nothing actionable applies. Fallback plans select
  only the fallback; it never mixes with actionable nodes.
- **Input is evidence, not authority.** A `RunRequest` carrying
  `refund.requested` does not approve a refund; adapters normalize and
  authenticate, they never choose nodes or mutate state.
- **Auditability by construction.** Every cycle emits an `AuditRecord` (plan,
  candidates, gate checks, committed transitions, rejection reason), so a
  review checks a graph path, not a transcript.

## The SLM wire contracts

The framework preserves the trained model contracts as stable wire formats:

- **Router**: `### Instruction:` / `### Response:` prompt wrap; instruction
  with category, current FSM state, conversation history, prior graph
  actions, intent, and exactly 10 candidates `[0]..[9]` where index 9 is the
  reserved `UNSUPPORTED_OR_OUT_OF_SCOPE_INTENT` slot; plan JSON
  `{reasoning, topology, execution_plan}` constrained to
  `parallel | sequential | fallback`. Hybrid plans are encoded as
  `sequential` with parallel inner steps; the adapter maps that shape to the
  internal `HYBRID` topology.
- **Tools**: `@tool` registers a function with a single pydantic args model
  and a JSON schema (`additionalProperties: false`); every invocation is
  logged as a `ToolInvocation`. The decorator is a drop-in for the original
  `@neosyntropy` contract, so trained edge extractors plug in unchanged.

## Economics

The billable and auditable unit is the **successful state transition** — a
verified state change that passed every gate. Failed plans, rejected
transitions, and hallucinated loops never count as successful transitions. Because each
node runs with a small, scoped prompt inside its own step (no mega-prompts),
cheap small language models can do the work, and the control layer — not the
model — carries the guarantees.
