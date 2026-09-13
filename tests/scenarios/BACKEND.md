# Backend APIs to validate for cookbook scenarios

Cookbook examples under `cookbook/` are now copied into `tests/scenarios/cookbook_*`.
The copies run **offline** so CI does not need a live API.

When the same cookbook is run live (`python cookbook/...` with `tests/.env`),
the backend should honor the endpoints below. Paths are relative to `/api/v1`.

## How to read `when`

| `when` | Enable with | Meaning |
|---|---|---|
| `cookbook_main` | `Client(api_key=...)` passed to `FSM.run` / decorator | Happens on a normal cookbook script |
| `monitor` | `NEOSYNTROPY_MONITOR=TRUE` (implied by remote) | Graph structure + run telemetry |
| `remote_execution` | `NEO_REMOTE_EXECUTION=TRUE` | Publish recoverable graphs and drive `/control/runs` |

`NEO_REMOTE_EXECUTION=TRUE` implies monitoring.

## Shared API groups

### Project bootstrap — every Client-backed cookbook

| Method | Path | Validate |
|---|---|---|
| `GET` | `/observability/projects` | `ensure_project` lists the key's projects |
| `POST` | `/observability/projects` | creates `{name, slug, description}`; `409` reuses by slug or case-insensitive name |

### Inference — SchemaNode, ReasoningNode, LLM validators/KPI

| Method | Path | Validate |
|---|---|---|
| `POST` | `/framework/inference` | `purpose`, optional `node` + `context` + `tools`, `schema`; response `text` and optional `tool_calls` |

Provider ids such as `gemini-2.5-flash` are **not** resolved in the SDK. The
client falls back to `neosyntropy/base` (`BackendProvider`) and the backend
must route the node's `provider` itself.

### Decorator concept registration — `@function_calling` / `@workflow`

| Method | Path | Validate |
|---|---|---|
| `POST` | `/observability/projects/{project_id}/functions` | fire-and-forget function manifest at decoration time |

### Monitor (`NEOSYNTROPY_MONITOR=TRUE`)

| Method | Path | Validate |
|---|---|---|
| `POST` | `/observability/projects/{project_id}/graphs` | source-free graph structure |
| `POST` | `/telemetry/runs` | run opened with graph manifest + input |
| `POST` | `/telemetry/runs/{run_id}/events` | `step_started`, `plan_proposed`, `step_completed`; reasoning cookbooks also `tools_executed` |
| `POST` | `/telemetry/runs/{run_id}/finish` | `status` + `final_state` |

### Remote execution (`NEO_REMOTE_EXECUTION=TRUE`)

| Method | Path | Validate |
|---|---|---|
| `POST` | `/observability/projects/{project_id}/code-artifacts/missing` | `{hashes}` → `{missing}` |
| `PUT` | `/observability/projects/{project_id}/code-artifacts/{sha256}` | gzip bundle upload |
| `PUT` | `/observability/projects/{project_id}/graphs/{graph_id}/recovery` | executable recovery revision |
| `GET` | `/observability/projects/{project_id}/graphs/{graph_id}` | graph after publish |
| `GET` | `/observability/projects/{project_id}/graphs/{graph_id}/artifacts` | artifact list |
| `GET` | `/observability/projects/{project_id}/graphs/{graph_id}/artifacts/{artifact_id}/download` | bundle bytes |
| `POST` | `/control/runs` | backend-owned control cycle (`graph` + `request`) |
| `POST` | `/control/runs/{run_id}/results` | node results or `client_rejection` until `End` |

Semantic-router cookbooks additionally:

| Method | Path | Validate |
|---|---|---|
| `POST` | `/framework/router` | legacy router; prefer routing **inside** `/control/runs` |

Without remote execution, `PreferredPathRouter` cannot choose among multiple
semantic labels and falls back. Live semantic cookbooks must be run with
`NEO_REMOTE_EXECUTION=TRUE`.

## Per-cookbook matrix

| Scenario | Cookbook | Backend APIs to validate |
|---|---|---|
| `cookbook_python_node` | `fsm/python_node_example.py` | project + monitor + remote **control** (no inference; Python handlers) |
| `cookbook_schema_node` | `fsm/schema_node_example.py` | project + **inference** + monitor + remote control |
| `cookbook_reasoning_prompt_tools` | `fsm/reasoning_node_prompt_tools_example.py` | inference with `tools` + `tools_executed` telemetry + remote control |
| `cookbook_reasoning_steps` | `fsm/reasoning_node_steps_example.py` | multi-step inference + tools + `tools_executed` |
| `cookbook_semantic_router_parallel` | `fsm/semantic_router_parallel_example.py` | remote **control/runs** (routing) + optional `/framework/router` |
| `cookbook_semantic_router_sequential` | `fsm/semantic_router_sequential_example.py` | same as parallel (chained follow-up Python nodes) |
| `cookbook_node_kpi` | `kpi/node_kpi_example.py` | inference for `SummarizeText`; KPI node is local Python |
| `cookbook_group_kpi` | `kpi/group_kpi_example.py` | inference for extract/classify; group KPI is local Python |
| `cookbook_fsm_path_kpi` | `kpi/fsm_path_kpi_example.py` | inference for parse/answer; path KPI is local Python |
| `cookbook_node_validation` | `validation/node_validation_example.py` | inference + **guarded deterministic edges** on `state["valid"]` via `/control/runs` |
| `cookbook_group_path_validation` | `validation/group_path_validation_example.py` | inference for extract/classify **and** `SemanticGroupPathValidator` |
| `cookbook_fsm_path_validation` | `validation/fsm_path_validation_example.py` | inference + path validator (Python) + guarded edges |
| `cookbook_function_calling` | `decorators/function_calling_example.py` | **functions** concept POST + inference for two graphs on one project |
| `cookbook_workflow_reasoning` | `decorators/workflow_reasoning_example.py` | functions concept + inference + tool loop + `tools_executed` |
| `cookbook_filesystem` | `tools/filesystem_example.py` | **none** — local `CodingTools` only |
| `cookbook_web_search` | `tools/web_search_example.py` | **none** — local DuckDuckGo / Trafilatura |
| `cookbook_email` | `tools/email_example.py` | **none** — local SMTP (`EmailTools`) |
| `cookbook_knowledge_retrieval` | `knowledge/retrieval_example.py` | **none** — local `FileSystemKnowledge.search` |
| `cookbook_knowledge_transform` | `knowledge/transform_example.py` | **none** — local `Knowledge.transform` |

The original production scenarios (`order_refund`, `billing_payment`,
`policy_gate`, `knowledge_ingest`, `knowledge_transform`,
`knowledge_retrieval`) are local SQLite/vector graphs. They do not call the
backend unless you attach a `Client`.

## Not exercised by cookbooks

These SDK methods exist but no cookbook hits them today. Do not treat a
cookbook run as coverage for:

| Method | Path |
|---|---|
| `GET` | `/observability/graphs/{graph_id}` (cross-project fetch) |
| `GET` | `/observability/projects/{project_id}/nodes/{node_id}/eval-samples` |
| `POST` | `/observability/projects/{project_id}/nodes/{node_id}/eval-samples/critic` |
| `POST` | `/eval/judge` |
| `POST` | `/observability/projects/{project_id}/nodes/{node_id}/tune` |
| `GET` | `/observability/projects/{project_id}/nodes/{node_id}/tune/{job_id}` |
| `POST` | `/observability/projects/{project_id}/knowledge` |
| `POST` | `/telemetry/concepts/{concept_id}/events` |

## Suggested backend test order

1. `cookbook_python_node` — project + `/control/runs` linear deterministic chain (no LLM).
2. `cookbook_schema_node` — add `/framework/inference` constrained JSON.
3. `cookbook_reasoning_prompt_tools` — inference `tool_calls` + `tools_executed`.
4. `cookbook_semantic_router_parallel` — `/control/runs` labeled routing.
5. `cookbook_function_calling` — two graphs, one project, `/functions` concept.
6. `cookbook_node_validation` — `/control/runs` must not mix the fallback node with `End` after a validation gate. Two guarded deterministic edges (`valid` → End, `not valid` → OutOfScope) have to be resolved as a unique next step (client-side guards, then a single candidate). The offline copy currently stops after `ValidateSummary` because mixed fallback/actionable candidates are rejected; the live backend must do better.
