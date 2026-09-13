# Scenario map

This directory is the **scenario map**: production-style graphs that leave
unit tests and run against related stores (relational tables, vector indexes,
filesystem corpora).

| Path | Role |
|---|---|
| [`GRAPH.md`](GRAPH.md) (this file) | Map of every scenario and its graph |
| `<scenario>/scenario.py` | Story **and** executable code in one file |
| [`deliveries/`](deliveries/) | Per-scenario tests that the run really landed (DB, vector, files) |

```text
tests/scenarios/
├── GRAPH.md
├── stores.py                  shared in-memory SQLite + vector adapters
├── order_refund/scenario.py
├── billing_payment/scenario.py
├── policy_gate/scenario.py
├── knowledge_ingest/scenario.py
├── knowledge_transform/scenario.py
├── knowledge_retrieval/scenario.py
└── deliveries/
    ├── test_order_refund.py
    ├── test_billing_payment.py
    ├── test_policy_gate.py
    ├── test_knowledge_ingest.py
    ├── test_knowledge_transform.py
    ├── test_knowledge_retrieval.py
    └── test_scenario_map.py
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

    subgraph deliveries["deliveries/"]
        D1["test_order_refund<br/>orders + refunds tables"]
        D2["test_billing_payment<br/>payments table"]
        D3["test_policy_gate<br/>decisions table"]
        D4["test_knowledge_ingest<br/>vector docs + contents"]
        D5["test_knowledge_transform<br/>summary vector docs"]
        D6["test_knowledge_retrieval<br/>hits + retrieval_log"]
    end

    OR --> D1
    BP --> D2
    PG --> D3
    KI --> D4
    KT --> D5
    KR --> D6
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
