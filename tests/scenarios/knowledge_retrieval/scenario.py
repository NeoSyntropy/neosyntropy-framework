"""Scenario: knowledge retrieval.

Search an ingested ``Knowledge`` corpus from an FSM node, then log every
hit into ``retrieval_log``. Deliveries check the vector hits and the log table.

```mermaid
flowchart LR
    BindQuery --> SearchKnowledge --> LogHits --> End
    BindQuery -.-> ErrorFallback
    SearchKnowledge -.-> ErrorFallback
    LogHits -.-> ErrorFallback
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

SCENARIO_ID = "knowledge_retrieval"
SCENARIO_TITLE = "Knowledge retrieval"


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    request_id: str


class BindOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str


class SearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hit_count: int


class LogOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    logged: int


SEEDED = (
    Document(
        name="renewal_policy.txt",
        content="Renewals are billed on the first of the month. Grace period is 7 days.",
        meta_data={"topic": "billing"},
    ),
    Document(
        name="shipping_update.txt",
        content="The parcel is in transit. Expected delivery is tomorrow afternoon.",
        meta_data={"topic": "shipping"},
    ),
)


@dataclass
class KnowledgeRetrievalScenario:
    fsm: FSM
    db: SqliteDatabase
    vector_db: InMemoryVectorDb
    knowledge: ScenarioKnowledge

    def run(self, payload: SearchRequest):
        return run_fsm(self.fsm, payload)


def _schema(db: SqliteDatabase) -> None:
    db.execute(
        "CREATE TABLE retrieval_log ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "request_id TEXT NOT NULL, "
        "query TEXT NOT NULL, "
        "doc_name TEXT NOT NULL, "
        "content TEXT NOT NULL)"
    )


def build_scenario() -> KnowledgeRetrievalScenario:
    db = SqliteDatabase(name="retrieval_db")
    vector_db = InMemoryVectorDb(name="policies")
    knowledge = ScenarioKnowledge(vector_db=vector_db, name="support_corpus")
    knowledge.insert(list(SEEDED))
    _schema(db)

    @node(id="BindQuery", input_schema=SearchRequest, output_schema=BindOutput)
    def bind_query(ctx: NodeContext) -> object:
        query = str(ctx.input["query"]).strip()
        if not query:
            return ctx.result(
                output={"query": ""},
                state_updates={"query": "", "reason": "empty query"},
            )
        return ctx.result(
            output={"query": query},
            state_updates={"query": query, "request_id": ctx.input["request_id"]},
        )

    @node(id="SearchKnowledge", input_schema=OpenInput, output_schema=SearchOutput)
    def search_knowledge(ctx: NodeContext) -> object:
        query = str(ctx.state.get("query", ""))
        hits = knowledge.search(query, limit=5)
        serialized = [{"name": doc.name, "content": doc.content} for doc in hits]
        return ctx.result(
            output={"hit_count": len(serialized)},
            state_updates={"hits": serialized, "hit_count": len(serialized)},
        )

    @node(id="LogHits", input_schema=OpenInput, output_schema=LogOutput)
    def log_hits(ctx: NodeContext) -> object:
        request_id = str(ctx.state.get("request_id", ""))
        query = str(ctx.state.get("query", ""))
        hits = list(ctx.state.get("hits") or [])
        for hit in hits:
            db.insert_row(
                "retrieval_log",
                {
                    "request_id": request_id,
                    "query": query,
                    "doc_name": str(hit["name"]),
                    "content": str(hit["content"]),
                },
            )
        return ctx.result(
            output={"logged": len(hits)},
            state_updates={"logged": len(hits)},
        )

    @node(
        id="ErrorFallback",
        input_schema=OpenInput,
        output_schema=TextOutput,
        is_fallback=True,
    )
    def error_fallback(ctx: NodeContext) -> object:
        return ctx.result(output={"message": "retrieval failed"})

    fsm = FSM(
        entry=bind_query,
        nodes=[bind_query, search_knowledge, log_hits, error_fallback],
        edges=[
            edge_deterministic("BindQuery", "SearchKnowledge"),
            edge_deterministic("SearchKnowledge", "LogHits"),
            edge_deterministic("LogHits", "End"),
            edge_fallback("BindQuery", "ErrorFallback"),
            edge_fallback("SearchKnowledge", "ErrorFallback"),
            edge_fallback("LogHits", "ErrorFallback"),
        ],
    )
    return KnowledgeRetrievalScenario(fsm=fsm, db=db, vector_db=vector_db, knowledge=knowledge)
