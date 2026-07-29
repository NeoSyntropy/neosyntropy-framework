# NeoSyntropy Framework

A deterministic control layer for AI workflows. Models propose what should
happen next; a finite-state graph defines what is allowed to happen.

Five primitives span the problem space:

| Primitive | Role |
|---|---|
| `Node` | Executable capability (handler or provider-backed), never a workflow position |
| `Edge` | One permitted movement between states, with labels and fail-closed guards |
| `Axiom` | An invariant enforced before execution and before commit — a broken axiom is a rejected step |
| `Group` | Organization for nodes; never a second control path |
| `ControlManager` | The whole cycle: select → route → validate → gate → execute → gate → commit → audit |

## Install

```bash
pip install -e .          # from this directory
pip install -e .[dev]     # with pytest + ruff
```

## Quickstart

```python
from neosyntropy import BackendClient, ControlManager, Edge, Graph, axiom, node

@node(id="VerifyIdentity")
def verify_identity(ctx):
    """Verify the requester owns the order."""
    return ctx.result(state_updates={"verified": True})

@node(id="IssueRefund", prerequisites=("VerifyIdentity",))
def issue_refund(ctx):
    return ctx.result(state_updates={"refund_issued": True}, next_state="End")

@node(id="OutOfScope", is_fallback=True)
def out_of_scope(ctx):
    return ctx.result(output="Out of scope for this workflow.")

@axiom(name="MaxRefund", error_message="Refund exceeds the cap.")
def max_refund(ctx, proposal):
    return proposal.state.get("refund_amount", 0) <= 200

graph = Graph(
    nodes=[verify_identity, issue_refund, out_of_scope],
    edges=[
        Edge(source="Start", target="VerifyIdentity", label="first"),
        Edge(source="VerifyIdentity", target="IssueRefund", label="next"),
        Edge(source="IssueRefund", target="End", label="complete"),
    ],
    axioms=[max_refund],
)

backend = BackendClient(
    "https://api.neosyntropy.com",
    api_key="your-api-key",
    project_id="your-project-id",
)
manager = ControlManager(graph, backend=backend)
result = manager.run({"intent": "refund my order", "current_state": "Start"})

print(result.final_state)                    # one committed transition
print(result.audit.committed_transitions)    # ["Start->VerifyIdentity"]
```

Every cycle returns a `RunResult` with a full `AuditRecord`: the proposed
plan, the candidates, every axiom check, and the committed transitions. A
rejection (illegal plan, broken axiom, illegal transition, failed guard) is a
normal outcome — `result.rejected` is set, nothing was committed for the
offending step, and the audit explains why.

## Routing and control

When backend credentials are configured, **the backend owns the control
cycle**: candidate selection, routing, plan validation, and state commits.
The client only defines the graph and executes local handlers. Responses never
include topology, candidates, execution plans, providers, or model names.

- **Backend control** (default with credentials): `POST /control/runs` +
  `POST /control/runs/{id}/results`. The SDK loops: receive opaque execute
  steps → run local handlers → submit results → accept commits/rejections.
- **`DeterministicRouter`** (offline fallback, no backend): walks outgoing
  edges by label priority; proposes the dedicated fallback when nothing is
  legal.
- Node generation for handler-less nodes may still use `/framework/slm`;
  selection/routing stay behind the control API.

Set `NEOSYNTROPY_API_URL` with `NEOSYNTROPY_API_KEY` + `NEOSYNTROPY_PROJECT_ID`
(or `NEOSYNTROPY_ACCESS_TOKEN`). `ControlManager(graph)` discovers them
automatically.

## Observability

When a backend client is configured, `ControlManager` reports the control
lifecycle to:

- `POST /api/v1/telemetry/runs`
- `POST /api/v1/telemetry/runs/{id}/events`
- `POST /api/v1/telemetry/runs/{id}/finish`

Telemetry is bounded and best-effort: an unavailable or slow observer never
changes execution, validation, state commits, returned results, or raised
execution errors. Events cover plan proposals, step start/completion,
committed transitions, rejection/failure, and finish.

Only a visualization manifest leaves the process. It contains node IDs,
display names, groups, fallback markers, and labeled edges. Prompts, handlers,
guards, tools, providers, descriptions, metadata, axioms, request intent,
history, state, outputs, and errors are excluded.

Use `graph_manifest(graph)` to inspect that payload or provide a custom
`RunObserver` with `ControlManager(graph, observer=...)`. See
[`examples/observability.py`](examples/observability.py).

## Tools

```python
from pydantic import BaseModel
from neosyntropy import tool

class AddToCartArgs(BaseModel):
    product_id: str
    quantity: int

@tool
def add_to_cart(args: AddToCartArgs) -> dict:
    """Add a quantity of a product to the active cart."""
    ...
```

Tools are capabilities on nodes (`@node(..., tools=("add_to_cart",))`), never
graph vertices. Handlers call them through `ctx.tools.invoke(...)`, which
enforces the node's allow-list fail-closed and logs every invocation.

### Model-driven tool calling

A node with no Python handler runs through the NeoSyntropy backend. If it
declares tools, the framework runs the split reasoning/extraction loop:

```python
Node(
    id="Support",
    prompt="Help the customer with their order.",
    tools=("lookup_order",),
)
```

The model reasons, emits `<TOOL:lookup_order>` with no arguments, a parameter
extractor fills a JSON object constrained by the tool's pydantic schema,
arguments are validated before the tool runs, and the outcome is reinjected so
reasoning continues. Only the node's own tools appear in its prompt.

The default extractor uses the same provider; plug a trained edge extractor in
with `ControlManager(graph, extractor=my_extractor)` — anything implementing
`async extract(messages, tool) -> ToolCall` works, which is the same contract
the trained extractors already speak.

Guarantees in this loop: a proposed tool that the node does not declare is
denied and never executed (the refusal is fed back so the model can recover),
invalid arguments never reach the tool, duplicate calls are rejected, and the
loop is bounded by `max_tool_calls`. Every attempt lands in
`NodeResult.tool_calls`, so an axiom can gate on tool usage:

```python
@axiom(name="NoFailedTools")
def no_failed_tools(ctx, proposal):
    if proposal.result is None:
        return True
    return all(record.ok for record in proposal.result.tool_calls)
```

## Example and docs

- [`examples/refund_workflow.py`](examples/refund_workflow.py) — end-to-end
  workflow with guards, axioms, tools, a rejection, and a fallback.
- [`docs/concepts.md`](docs/concepts.md) — the distilled methodology: nodes
  vs states, proposal vs permission, fail-closed axioms, wire contracts.

## Tests

```bash
pytest
ruff check .
```
