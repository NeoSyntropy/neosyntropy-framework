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

### Node — executable capability

A node packages a capability: a Python handler or a provider-backed prompt,
declared tools, prerequisites, and an optional group. The router may select
one or several nodes per cycle. **Nodes are not workflow positions**: running
three nodes in parallel does not create three states.

Nodes return a `NodeResult` — output, `state_updates`, and an optional
`next_state`. That result is a proposal. Nothing a node returns commits
anything.

Tools are capabilities *on* a node (`tools=("lookup_order",)`), never graph
vertices. Handlers reach tools through a bound facade that enforces the
allow-list fail-closed: calling an undeclared tool is denied fail-closed, not
an error to retry around.

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


### Group — organization, not control

Groups name collections of nodes for organization and candidate metadata.
The validator and the executor never consult groups. Grouping must not
create a second control path — one pipeline owns the sequence end to end.

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
