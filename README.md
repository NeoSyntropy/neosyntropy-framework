<p align="center">
  <img src="docs/assets/neosyntropy-mark.png" alt="NeoSyntropy" width="96" />
</p>

<h1 align="center">NeoSyntropy Framework</h1>

<p align="center">
  <strong>Stop paying for LLM hallucinations. Start paying for deterministic transitions.</strong>
</p>

Agent frameworks leave three problems unsolved:

1. **Hallucinations** — the model invents steps, invents facts, and invents transitions you never approved.
2. **Unit cost** — you pay for tokens, retries, and “thinking,” not for a successful business outcome.
3. **Uncontrolled token spend** — every step re-reads history, self-correction loops bill you again, and finance only learns the number when the invoice arrives.

NeoSyntropy is a **deterministic control layer** for AI workflows. Models propose what should happen next; a finite-state graph defines what is allowed to happen. Every action follows a path you defined: **0% deviation** from your business logic, enforced at the engine level — not begged for in a prompt.

Replace opaque token bills with one measurable unit: the **successful state transition** (rates as low as **$0.002**). Failed or hallucinated steps are not billable events.

---

Core primitives — start with the model-backed nodes. That is the point of the
framework: you drop tiny, scoped AI models into ordinary code paths
(`SchemaNode` / `ReasoningNode`), while routers and `ControlManager` keep every
proposal inside a fail-closed graph.

| Primitive | Role |
|---|---|
| [`SchemaNode`](docs/concepts-explained.md#3-schemanode--constrained-json-no-tools) | Provider-backed extraction: a small model returns constrained JSON for `output_schema`, no tools |
| [`ReasoningNode`](docs/concepts-explained.md#4-reasoningnode--tools--plain-text-notes) | Provider-backed reasoning: a small model may call allow-listed tools and write plain-text notes |
| [`CombineNode`](docs/concepts-explained.md#5-combinenode--reasoning-then-schema) | Authoring unit that expands to reasoning → schema FSM states |
| [`Node`](docs/concepts-explained.md#2-node--node--executable-capability) | Executable capability (Python handler or provider-backed), never a workflow position |
| [`DeterministicRouter`](docs/concepts-explained.md#7-deterministicrouter--hard-rules) | First matching `(predicate, target)` rule wins; compiles to deterministic edges |
| [`SemanticRouter`](docs/concepts-explained.md#8-semanticrouter--labeled-intent-routes) | Model picks among labeled targets (`routes={label: node_or_group}`); still validated against the graph |
| [`Edge`](docs/concepts-explained.md#6-edge--one-permitted-movement) | One permitted movement: `deterministic`, `semantic`, or `fallback` |
| [`Group`](docs/concepts-explained.md#9-group--named-subgraph-optional) | Named node collection; optional `entry`, internal routers, and `add_edge` that compile into the FSM |
| [`ControlManager`](docs/concepts-explained.md#11-controlmanager--one-control-cycle) | The whole cycle: deterministic → semantic router → fallback → validate → execute → commit |

Each concept explained (what / when / example):
[`docs/concepts-explained.md`](docs/concepts-explained.md)

Site docs:
[Nodes](https://docs.neosyntropy.com/concepts/nodes) ·
[Model-backed nodes](https://docs.neosyntropy.com/concepts/model-nodes) ·
[Routers](https://docs.neosyntropy.com/concepts/routers) ·
[Edges](https://docs.neosyntropy.com/concepts/edges) ·
[Groups](https://docs.neosyntropy.com/concepts/groups) ·
[Control manager](https://docs.neosyntropy.com/concepts/control-manager)

## Sign up

Create an account at [neosyntropy.com](https://neosyntropy.com) to use the
control plane. New accounts include **100 free node passes** so you can run
workflows before buying credits. Paid usage starts at **$0.005 per node**
(minimum); the starter package is **$10 for 1,000 state transitions**. After
sign-in, create a project and copy your API key + project id for the client
below (or use the **neo-code** CLI: `neo-code login`).

## Install

```bash
pip install neosyntropy
```

From a local checkout:

```bash
pip install -e .          # editable install
pip install -e ".[dev]"   # with pytest + ruff + build tools
```

## CLI login and projects

Auth and project management live in the **neo-code** CLI (separate package), not
in this framework install. The Python import API (`from neosyntropy import …`)
is unchanged. Sign in at [neosyntropy.com](https://neosyntropy.com) first, then:

```bash
# from the neo-code package / BMAD-METHOD-NEOSYNTROPY checkout
neo-code login
neo-code project create "Support automation" --use
neo-code project list
```

```bash
neo-code --api-url http://localhost:8000 --profile development login
neo-code --profile development project use <project-id>
neo-code logout
```

## Quickstart

```python
from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    Client,
    EmptyOutput,
    OpenInput,
    SchemaNode,
    TextOutput,
    Workflow,
    node,
)

# Connect with a workspace API key from Settings. Pass project_id=... if you
# already have a project, or create/reuse one by slug:
client = Client(api_key="your-api-key")
client.create_project(name="Support Bot", slug="support-bot")


class RefundTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str
    amount: float
    reason: str


@node(id="VerifyIdentity", input_schema=OpenInput, output_schema=EmptyOutput)
def verify_identity(ctx):
    """Verify the requester owns the order."""
    return ctx.result(output={}, state_updates={"verified": True})

# Provider-backed: model must return JSON matching RefundTicket.
extract_ticket = SchemaNode(
    id="ExtractTicket",
    input_schema=OpenInput,
    output_schema=RefundTicket,
    prompt="Extract a refund ticket as JSON from the customer request.",
)

@node(id="IssueRefund", prerequisites=("ExtractTicket",), input_schema=OpenInput, output_schema=EmptyOutput)
def issue_refund(ctx):
    return ctx.result(output={}, state_updates={"refund_issued": True}, next_state="End")

@node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
def out_of_scope(ctx):
    return ctx.result(output={"message": "Out of scope for this workflow."})

fsm = Workflow(
    [verify_identity, extract_ticket, issue_refund],
    fallback=out_of_scope,
)

result = fsm.run(
    {"text": "refund order ord_123 for 40 dollars — item arrived damaged"},
    client=client,
)

print(result.final_state)
print(result.audit.committed_transitions)
```

Every cycle returns a `RunResult` with a full `AuditRecord`. A rejection is a
normal outcome — `result.rejected` is set and the audit explains why.

More detail: [`docs/concepts-explained.md`](docs/concepts-explained.md) ·
[`examples/refund_workflow.py`](examples/refund_workflow.py)

Decorators that extract parameters into a Python function:
[`cookbook/decorators`](cookbook/decorators) (`@function_calling`, `@workflow`).

## License

NeoSyntropy Framework is licensed under the
[GNU Affero General Public License v3.0 or later](LICENSE) (AGPL-3.0-or-later).

You may use and modify it, including commercially, but if you distribute a
modified version — or run a modified version as a network service — you must
make the corresponding source available under the same license.
