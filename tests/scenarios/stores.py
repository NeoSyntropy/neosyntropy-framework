"""In-memory stores used by scenario graphs and their delivery tests.

Scenarios persist side effects here so deliveries can query the related
database (and vector index) after a run — the same checks you would make
against Postgres or a vector DB in production.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from typing import Any

from neosyntropy.knowledge.document import Document


class InMemoryVectorDb:
    """Minimal vector-store stand-in with insert / upsert / search."""

    def __init__(self, name: str = "vdb") -> None:
        self.name = name
        self.docs: list[Document] = []
        self.upsert_called = False

    def upsert_available(self) -> bool:
        return True

    def upsert(self, content_hash: str, documents: list[Document], filters: Any = None) -> None:
        self.upsert_called = True
        self.docs.extend(documents)

    def insert(self, content_hash: str, documents: list[Document], filters: Any = None) -> None:
        self.docs.extend(documents)

    def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[Document]:
        needle = query.lower()
        hits = [doc for doc in self.docs if needle in doc.content.lower()]
        return hits[:limit]

    async def async_search(self, query: str, limit: int = 5, **kwargs: Any) -> list[Document]:
        return self.search(query, limit=limit, **kwargs)


class SqliteDatabase:
    """Relational stand-in. Tables are queried with real SQL in deliveries."""

    def __init__(self, name: str = "sqlite") -> None:
        self.name = name
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.records: list[dict[str, Any]] = []

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        cursor = self.conn.execute(sql, params)
        self.conn.commit()
        return cursor

    def insert_row(self, table: str, record: dict[str, Any]) -> None:
        columns = list(record.keys())
        placeholders = ", ".join("?" for _ in columns)
        col_sql = ", ".join(columns)
        self.conn.execute(
            f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})",
            tuple(record[column] for column in columns),
        )
        self.conn.commit()
        self.records.append({"table": table, **record})

    def insert(self, record: Any) -> None:
        """Knowledge-compatible insert: persist a dict or Document-like payload."""
        if isinstance(record, dict) and "table" in record:
            table = str(record["table"])
            payload = {key: value for key, value in record.items() if key != "table"}
            self.insert_row(table, payload)
            return
        self.execute(
            "CREATE TABLE IF NOT EXISTS knowledge_contents ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "name TEXT NOT NULL DEFAULT '', "
            "content TEXT NOT NULL)"
        )
        content = getattr(record, "content", record)
        self.insert_row(
            "knowledge_contents",
            {
                "name": str(getattr(record, "name", "") or ""),
                "content": str(content),
            },
        )

    def fetch_all(self, table: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(f"SELECT * FROM {table}").fetchall()
        return [dict(row) for row in rows]

    def read_sql(
        self,
        query: str,
        schema: Any = None,
        params: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        records = [dict(row) for row in self.conn.execute(query, params).fetchall()]
        if schema is None:
            return records
        return [schema.model_validate(record).model_dump() for record in records]


def _as_documents(data: Any) -> list[Document]:
    if data is None:
        return []
    items = data if isinstance(data, list) else [data]
    documents: list[Document] = []
    for item in items:
        if isinstance(item, Document):
            documents.append(item)
        elif isinstance(item, str):
            documents.append(Document(content=item))
        elif isinstance(item, dict):
            documents.append(
                Document(
                    name=item.get("name"),
                    content=str(item.get("content", item)),
                    meta_data=item.get("meta_data", {}),
                )
            )
        else:
            documents.append(Document(content=str(item)))
    return documents


class ScenarioKnowledge:
    """Knowledge-shaped corpus that stays on the in-memory stores.

    Mirrors insert / search / transform enough for scenario graphs without
    importing optional cloud loaders required by ``neosyntropy.knowledge.Knowledge``.
    """

    def __init__(
        self,
        *,
        vector_db: InMemoryVectorDb | None = None,
        database: SqliteDatabase | None = None,
        name: str = "knowledge",
        transform: Callable[..., Any] | None = None,
    ) -> None:
        self.vector_db = vector_db
        self.database = database
        self.name = name
        self.transform_pipeline = transform

    def insert(self, data: Any, **kwargs: Any) -> bool:
        documents = _as_documents(data)
        if self.vector_db is not None and documents:
            self.vector_db.upsert("ingest", documents)
        return True

    def search(self, query: str, limit: int = 5, **kwargs: Any) -> list[Document]:
        if self.vector_db is None:
            return []
        return self.vector_db.search(query, limit=limit, **kwargs)

    def transform(
        self,
        source: Any = None,
        destination: Any = None,
        **kwargs: Any,
    ) -> Any:
        if source is None:
            raw: Iterable[Any] = []
        elif hasattr(source, "load"):
            raw = source.load(**kwargs)
        else:
            raw = source
        transformed = raw
        if callable(self.transform_pipeline):
            transformed = self.transform_pipeline(raw, **kwargs)
        target = destination if destination is not None else self
        if hasattr(target, "insert"):
            target.insert(transformed, **kwargs)
        return transformed


def run_fsm(fsm: Any, payload: Any, *, state: dict[str, Any] | None = None) -> Any:
    """Run an authored FSM locally (no remote backend required)."""
    return fsm.run(payload, state=state or {}, client=None)
