# Scenario map

This directory is the **scenario map**: production-style graphs that leave
unit tests and run against related stores (relational tables, vector indexes,
filesystem corpora), plus **copies of every cookbook example**.

| Path | Role |
|---|---|
| [`GRAPH.md`](GRAPH.md) (this file) | Map of every scenario and its graph |
| [`BACKEND.md`](BACKEND.md) | Which backend APIs each cookbook should validate live |
| `<scenario>/scenario.py` | Story **and** executable code in one file |
| [`cookbook_support.py`](cookbook_support.py) | Loader, offline provider, and API catalog for cookbook copies |
| [`deliveries/`](deliveries/) | Per-scenario tests that the run really landed (DB, vector, files) |

```text
tests/scenarios/
├── GRAPH.md
├── BACKEND.md
├── stores.py
├── cookbook_support.py
├── order_refund/scenario.py
├── billing_payment/scenario.py
├── policy_gate/scenario.py
├── knowledge_ingest/scenario.py
├── knowledge_transform/scenario.py
├── knowledge_retrieval/scenario.py
├── cookbook_python_node/scenario.py
├── cookbook_schema_node/scenario.py
├── cookbook_reasoning_prompt_tools/scenario.py
├── cookbook_reasoning_steps/scenario.py
├── cookbook_semantic_router_parallel/scenario.py
├── cookbook_semantic_router_sequential/scenario.py
├── cookbook_node_kpi/scenario.py
├── cookbook_group_kpi/scenario.py
├── cookbook_fsm_path_kpi/scenario.py
├── cookbook_node_validation/scenario.py
├── cookbook_group_path_validation/scenario.py
├── cookbook_fsm_path_validation/scenario.py
├── cookbook_function_calling/scenario.py
├── cookbook_workflow_reasoning/scenario.py
├── cookbook_filesystem/scenario.py
├── cookbook_web_search/scenario.py
├── cookbook_email/scenario.py
├── cookbook_knowledge_retrieval/scenario.py
├── cookbook_knowledge_transform/scenario.py
└── deliveries/
```

## How the map is wired

```mermaid
flowchart TB
    subgraph map["tests/scenarios"]
        GRAPH["GRAPH.md"]
        STORE["stores.py<br/>SQLite + InMemoryVectorDb"]
    end

    subgraph fsm["FSM / control graphs"]
        OR["order_refund"]
        BP["billing_payment"]
        PG["policy_gate"]
    end

    subgraph knowledge["Knowledge production"]
        KI["knowledge_ingest"]
        KT["knowledge_transform"]
        KR["knowledge_retrieval"]
        KI --> KT
        KT --> KR
    end

    GRAPH --> OR
    GRAPH --> BP
    GRAPH --> PG
    GRAPH --> KI
    GRAPH --> KT
    GRAPH --> KR

    STORE --> OR
    STORE --> BP
    STORE --> PG
    STORE --> KI
    STORE --> KT
    STORE --> KR

    subgraph cookbooks["cookbook copies"]
        CFSM["cookbook_python_node / cookbook_schema_node / cookbook_reasoning_* / cookbook_semantic_router_*"]
        CKPI["cookbook_node_kpi / cookbook_group_kpi / cookbook_fsm_path_kpi"]
        CVAL["cookbook_node_validation / cookbook_group_path_validation / cookbook_fsm_path_validation"]
        CDEC["cookbook_function_calling / cookbook_workflow_reasoning"]
        CTOOL["cookbook_filesystem / cookbook_web_search / cookbook_email"]
        CKNOW["cookbook_knowledge_retrieval / cookbook_knowledge_transform"]
    end

    GRAPH --> CFSM
    GRAPH --> CKPI
    GRAPH --> CVAL
    GRAPH --> CDEC
    GRAPH --> CTOOL
    GRAPH --> CKNOW

    subgraph deliveries["deliveries/"]
        D1["test_order_refund<br/>orders + refunds tables"]
        D2["test_billing_payment<br/>payments table"]
        D3["test_policy_gate<br/>decisions table"]
        D4["test_knowledge_ingest<br/>vector docs + contents"]
        D5["test_knowledge_transform<br/>summary vector docs"]
        D6["test_knowledge_retrieval<br/>hits + retrieval_log"]
        DC["test_cookbook_*<br/>cookbook_runs + backend_apis"]
    end

    OR --> D1
    BP --> D2
    PG --> D3
    KI --> D4
    KT --> D5
    KR --> D6
    CFSM --> DC
    CKPI --> DC
    CVAL --> DC
    CDEC --> DC
    CTOOL --> DC
    CKNOW --> DC
```

## Scenario graphs

### order_refund

Validate an existing order, persist a refund row, then mark the order refunded.

```mermaid
flowchart LR
    ValidateOrder --> WriteRefund --> ConfirmRefund --> End
    ValidateOrder -.-> ErrorFallback
    WriteRefund -.-> ErrorFallback
    ConfirmRefund -.-> ErrorFallback
```

### billing_payment

Group-scoped card check, then a deterministic router to capture or reject.

```mermaid
flowchart LR
    ValidateCard --> BillingLogic
    BillingLogic -->|card_valid| ProcessPayment --> End
    BillingLogic -->|not valid| RejectCard --> End
    ValidateCard -.-> OutOfScope
```

### policy_gate

Hard eligibility rule (account age) written to a decisions table.

```mermaid
flowchart LR
    CheckPolicy --> EligibilityGate
    EligibilityGate -->|eligible| ApproveRequest --> End
    EligibilityGate -->|not eligible| DenyRequest --> End
    CheckPolicy -.-> OutOfScope
```

### knowledge_ingest

Python workflow loads documents into `Knowledge` (vector DB) and a contents table.

```mermaid
flowchart LR
    LoadCorpus --> PersistKnowledge --> End
    LoadCorpus -.-> ErrorFallback
    PersistKnowledge -.-> ErrorFallback
```

### knowledge_transform

Filesystem corpus → transform summaries → destination `Knowledge` vector store.

```mermaid
flowchart LR
    SourceFS["FileSystemKnowledge"] --> Transform
    Transform --> DestVdb["destination vector DB"]
    Transform --> Jobs["transform_jobs table"]
```

### knowledge_retrieval

Search ingested Knowledge, then log hits so deliveries can query `retrieval_log`.

```mermaid
flowchart LR
    BindQuery --> SearchKnowledge --> LogHits --> End
    BindQuery -.-> ErrorFallback
    SearchKnowledge -.-> ErrorFallback
    LogHits -.-> ErrorFallback
```

## Cookbook copies

Each `cookbook_*` scenario loads the matching file under `cookbook/` and
runs it offline. Live backend endpoints for the same graphs are listed in
[`BACKEND.md`](BACKEND.md).

### cookbook_python_node

Copy of `cookbook/fsm/python_node_example.py`.

```mermaid
flowchart LR
    ValidateOrder --> FormatResponse --> End
    ValidateOrder -.-> ErrorFallback
    FormatResponse -.-> ErrorFallback
```

### cookbook_schema_node

Copy of `cookbook/fsm/schema_node_example.py`.

```mermaid
flowchart LR
    ExtractTicket --> End
    ExtractTicket -.-> OutOfScope
```

### cookbook_reasoning_prompt_tools

Copy of `cookbook/fsm/reasoning_node_prompt_tools_example.py`.

```mermaid
flowchart LR
    RouteIntent --> End
    RouteIntent -.-> OutOfScope
```

### cookbook_reasoning_steps

Copy of `cookbook/fsm/reasoning_node_steps_example.py`.

```mermaid
flowchart LR
    SupportDecision_step_0 --> SupportDecision_step_1 --> SupportDecision_step_2 --> End
```

### cookbook_semantic_router_parallel

Copy of `cookbook/fsm/semantic_router_parallel_example.py`.

```mermaid
flowchart LR
    CaptureRequest --> SupportIntent
    SupportIntent --> BillingHelp --> End
    SupportIntent --> ShippingHelp --> End
```

### cookbook_semantic_router_sequential

Copy of `cookbook/fsm/semantic_router_sequential_example.py`.

```mermaid
flowchart LR
    CaptureRequest --> SupportIntent
    SupportIntent --> InvestigateBilling --> ResolveRequest --> End
    SupportIntent --> InvestigateShipping --> ResolveRequest
```

### cookbook_node_kpi

Copy of `cookbook/kpi/node_kpi_example.py`.

```mermaid
flowchart LR
    SummarizeText --> ScoreSummary --> End
```

### cookbook_group_kpi

Copy of `cookbook/kpi/group_kpi_example.py`.

```mermaid
flowchart LR
    ExtractTicket --> ClassifyUrgency --> ScoreTriage --> End
```

### cookbook_fsm_path_kpi

Copy of `cookbook/kpi/fsm_path_kpi_example.py`.

```mermaid
flowchart LR
    ParseQuery --> GenerateAnswer --> PathScore --> End
```

### cookbook_node_validation

Copy of `cookbook/validation/node_validation_example.py`.

```mermaid
flowchart LR
    SummarizeText --> ValidateSummary --> End
    ValidateSummary --> OutOfScope
```

### cookbook_group_path_validation

Copy of `cookbook/validation/group_path_validation_example.py`.

```mermaid
flowchart LR
    ExtractTicket --> ClassifyUrgency --> ValidateTriage --> End
```

### cookbook_fsm_path_validation

Copy of `cookbook/validation/fsm_path_validation_example.py`.

```mermaid
flowchart LR
    ParseQuery --> GenerateAnswer --> AuditPath --> End
```

### cookbook_function_calling

Copy of `cookbook/decorators/function_calling_example.py`.

```mermaid
flowchart LR
    ExtractParams --> greet --> End
    ExtractParams2 --> summarize --> End
```

### cookbook_workflow_reasoning

Copy of `cookbook/decorators/workflow_reasoning_example.py`.

```mermaid
flowchart LR
    lookup_sku --> check_stock --> ExtractParams --> place_order
```

### cookbook_filesystem

Copy of `cookbook/tools/filesystem_example.py`. Local toolkit only.

### cookbook_web_search

Copy of `cookbook/tools/web_search_example.py`. Local toolkit only.

### cookbook_email

Copy of `cookbook/tools/email_example.py`. Local toolkit only (intended message recorded; SMTP not sent).

### cookbook_knowledge_retrieval

Copy of `cookbook/knowledge/retrieval_example.py` (`FileSystemKnowledge.search`).

### cookbook_knowledge_transform

Copy of `cookbook/knowledge/transform_example.py` (`Knowledge.transform`).
