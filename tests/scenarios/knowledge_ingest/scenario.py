"""Scenario: knowledge ingest.

Load a small corpus into ``Knowledge``: documents land in the vector index
and a relational ``knowledge_contents`` table. Deliveries check both stores.

```mermaid
flowchart LR
    LoadCorpus --> PersistKnowledge --> End
    LoadCorpus -.-> ErrorFallback
    PersistKnowledge -.-> ErrorFallback
```
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    FSM,
    NodeContext,
    OpenInput,
    TextOutput,
    edge_deterministic,
    edge_fallback,
    node,
)
from neosyntropy.knowledge.document import Document
from tests.scenarios.stores import (
    InMemoryVectorDb,
    ScenarioKnowledge,
    SqliteDatabase,
    run_fsm,
)

SCENARIO_ID = "knowledge_ingest"
SCENARIO_TITLE = "Knowledge ingest"


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus_id: str


class LoadOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_count: int


class PersistOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stored: int


CORPUS = (
    Document(
        name="renewal_policy.txt",
        content="Renewals are billed on the first of the month. Grace period is 7 days.",
        meta_data={"topic": "billing"},
    ),
    Document(
        name="refund_policy.txt",
        content="Refunds under $100 are auto-approved when the order is paid.",
        meta_data={"topic": "refunds"},
    ),
)


@dataclass
class KnowledgeIngestScenario:
    fsm: FSM
    db: SqliteDatabase
    vector_db: InMemoryVectorDb
    knowledge: ScenarioKnowledge

    def run(self, payload: IngestRequest):
        return run_fsm(self.fsm, payload)


def _schema(db: SqliteDatabase) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS knowledge_contents ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "corpus_id TEXT NOT NULL, "
        "name TEXT NOT NULL, "
        "content TEXT NOT NULL)"
    )


def build_scenario() -> KnowledgeIngestScenario:
    db = SqliteDatabase(name="knowledge_db")
    vector_db = InMemoryVectorDb(name="policies")
    knowledge = ScenarioKnowledge(vector_db=vector_db, database=db, name="policy_corpus")
    _schema(db)

    @node(id="LoadCorpus", input_schema=IngestRequest, output_schema=LoadOutput)
    def load_corpus(ctx: NodeContext) -> object:
        docs = [
            {"name": doc.name, "content": doc.content, "topic": doc.meta_data.get("topic")}
            for doc in CORPUS
        ]
        return ctx.result(
            output={"document_count": len(docs)},
            state_updates={
                "corpus_id": ctx.input["corpus_id"],
                "documents": docs,
            },
        )

    @node(id="PersistKnowledge", input_schema=OpenInput, output_schema=PersistOutput)
    def persist_knowledge(ctx: NodeContext) -> object:
        corpus_id = str(ctx.state["corpus_id"])
        documents = [
            Document(
                name=item["name"],
                content=item["content"],
                meta_data={"topic": item["topic"], "corpus_id": corpus_id},
            )
            for item in ctx.state["documents"]
        ]
        knowledge.insert(documents)
        for doc in documents:
            db.insert_row(
                "knowledge_contents",
                {
                    "corpus_id": corpus_id,
                    "name": doc.name,
                    "content": doc.content,
                },
            )
        return ctx.result(
            output={"stored": len(documents)},
            state_updates={"stored": len(documents)},
        )

    @node(
        id="ErrorFallback",
        input_schema=OpenInput,
        output_schema=TextOutput,
        is_fallback=True,
    )
    def error_fallback(ctx: NodeContext) -> object:
        return ctx.result(output={"message": "ingest failed"})

    fsm = FSM(
        entry=load_corpus,
        nodes=[load_corpus, persist_knowledge, error_fallback],
        edges=[
            edge_deterministic("LoadCorpus", "PersistKnowledge"),
            edge_deterministic("PersistKnowledge", "End"),
            edge_fallback("LoadCorpus", "ErrorFallback"),
            edge_fallback("PersistKnowledge", "ErrorFallback"),
        ],
    )
    return KnowledgeIngestScenario(fsm=fsm, db=db, vector_db=vector_db, knowledge=knowledge)
