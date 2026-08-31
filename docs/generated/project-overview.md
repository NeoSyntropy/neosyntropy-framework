# NeoSyntropy Framework: Project Overview

## Purpose

The NeoSyntropy Framework is a **deterministic control layer for AI workflows**. Its primary goal is to address common challenges in AI agent frameworks, specifically:

1.  **Hallucinations**: By enforcing a finite-state graph, the framework ensures that models cannot invent steps, facts, or transitions not explicitly defined, leading to 0% deviation from business logic.
2.  **Unit Cost**: It shifts the billing model from opaque token usage to measurable **successful state transitions**, providing predictable costs.
3.  **Uncontrolled Token Spend**: The framework prevents wasteful self-correction loops and re-reading of history by strictly adhering to defined paths.

In essence, NeoSyntropy ensures that "Models propose. The graph permits."

## Core Primitives

### 1. FSM / Workflow

The **Finite State Machine (FSM)**, represented by `FSM` or the `Workflow` helper, is the central orchestrator. It defines the entire graph of nodes, edges, and routers, acting as the single source of truth for permitted actions and transitions.

*   **`FSM(...)`**: Used for full authoring of complex graphs, including nodes, edges, routers, groups, and a required `entry` point.
*   **`Workflow(...)`**: A convenience constructor for simple, linear graphs where nodes execute sequentially. It automatically wires nodes and a fallback.

The `entry` point's `input_schema` defines the contract for the workflow's initial input.

### 2. Nodes

Nodes are the executable capabilities within the workflow. They are not workflow positions but rather actions that can be performed. All nodes require `input_schema` and `output_schema`.

*   **`@node` (Python Handler)**: A decorator for Python functions that perform deterministic logic, such as authentication checks, database writes, or API calls. They return a `NodeResult` with `output` and `state_updates`.
*   **`SchemaNode`**: A provider-backed node designed for extracting constrained JSON output. It has no tool-calling capabilities.
*   **`ReasoningNode`**: A provider-backed node that calls allow-listed tools and generates plain-text notes. Its output is typically consumed by a subsequent `SchemaNode` or `CombineNode` for structured extraction.
*   **`CombineNode`**: An authoring shortcut that internally expands into a `ReasoningNode` (for tool-calling and reasoning) followed by a `SchemaNode` (for structured JSON output). It simplifies workflows requiring both evidence gathering and typed decisions.

Nodes return *proposals* for `output`, `state_updates`, and `next_state`. These proposals are only committed if they pass all validation gates defined by the graph.

### 3. State

Every workflow run maintains a mutable `state` dictionary.

*   It starts with the `state={}` provided to `fsm.run()`.
*   It accumulates `state_updates` proposed by committed nodes.
*   Nodes access it via `ctx.state`.
*   `input` (from `fsm.run()`) is immutable evidence, validated once at the start. `state` is mutable and accumulates changes throughout the workflow.

### 4. Edges

Edges define the permitted movements between nodes or routers in the FSM. The transition table is the single source of permission.

*   **`deterministic`**: Auto-commits when its optional `guard` (a `callable(state: dict) -> bool`) passes. No router involved.
*   **`semantic`**: Scopes the `SemanticRouter` to a node or group target.
*   **`fallback`**: A safe stop, selected only when neither deterministic nor semantic routing yields a route.

Routers compile into edges at `FSM` build time; direct `Edge` construction is rare.

### 5. Routers

Routers are responsible for directing the flow of the workflow.

*   **`DeterministicRouter`**: Implements hard, rule-based branching. It uses a list of `(predicate, target)` rules. Predicates evaluate `ctx.state`. The first matching rule determines the next step. **No model calls are involved**, making it suitable for compliance, authentication, or policy enforcement. Targets can be nodes, node IDs, or other routers, but not groups.
*   **`SemanticRouter`**: A model picks among labeled targets (`routes={label: node_or_group}`). The model's proposal is validated against the graph. It supports up to 9 labeled routes plus 1 fallback. The backend returns a "plan" with a topology (`sequential`, `parallel`, `fallback`) that the `ControlManager` executes. Used for soft intent branching.

### 6. Group

A `Group` is a named subgraph that provides organizational structure. It can contain its own nodes, routers, and edges, and defines an optional `entry` point. At runtime, groups are merged into the parent FSM, and there is still a single control engine.

### 7. Tools / ToolRegistry

Tools are capabilities that can be invoked by `ReasoningNode`s.

*   Defined using the `@tool` decorator and registered in a `ToolRegistry`.
*   Declared on a node via `tools=(...)`.
*   Invoked only through `ctx.tools` within a node, with allow-listing enforced.
*   Tools are never graph vertices themselves.

## How a Run Works (Control Loop)

The `ControlManager` orchestrates each step of a workflow run, ensuring determinism and adherence to the defined graph. The cycle involves:

1.  **Input Schema Validation**: The initial `input` to `fsm.run()` is validated against the `entry` node's `input_schema`.
2.  **Candidate Generation**: Based on the current state and available edges/routers, potential next steps are identified.
3.  **Router Proposal**: If a router is involved, it proposes a next step (e.g., `DeterministicRouter` finds a matching rule, `SemanticRouter` uses a model to pick an intent).
4.  **Validation**: The proposed plan (which might include multiple nodes in `sequential` or `parallel` topology) is rigorously validated against the FSM's defined edges and guards.
5.  **Execution**: If validation passes, the node(s) in the plan are executed.
6.  **Guards / Transitions**: Any guards on outgoing edges are evaluated.
7.  **Commit**: If all checks pass, the `output` and `state_updates` from the executed node(s) are committed to the workflow's `state`.
8.  **AuditRecord**: Every step, whether successful or rejected, is recorded in an `AuditRecord`.

**Guarantees of the Control Loop:**

*   **Proposal is not permission**: Model or router proposals are always subject to validation against the graph.
*   **One current state**: At most one atomic commit occurs per plan step.
*   **Fail-closed before commit**: Even if a model has high confidence, an illegal transition or failed guard will result in rejection, and the workflow state remains unchanged for that step.
