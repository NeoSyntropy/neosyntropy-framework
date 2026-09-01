import asyncio
from typing import Any, List
import pytest

from neosyntropy.knowledge import (
    Knowledge,
    KnowledgeProtocol,
    KnowledgeTransformProtocol,
)
from neosyntropy.knowledge.document import Document
from neosyntropy.knowledge.transform import transform, Input, Output


class DummyVectorDb:
    def __init__(self, name: str = "vdb"):
        self.name = name
        self.docs = []
        self.upsert_called = False

    def upsert_available(self) -> bool:
        return True

    def upsert(self, content_hash: str, documents: List[Document], filters: Any = None) -> None:
        self.upsert_called = True
        self.docs.extend(documents)

    def insert(self, content_hash: str, documents: List[Document], filters: Any = None) -> None:
        self.docs.extend(documents)

    def search(self, query: str, limit: int = 5, **kwargs) -> List[Document]:
        return [doc for doc in self.docs if query.lower() in doc.content.lower()][:limit]

    async def async_search(self, query: str, limit: int = 5, **kwargs) -> List[Document]:
        return self.search(query, limit=limit, **kwargs)


class DummyDatabase:
    def __init__(self, name: str = "db"):
        self.name = name
        self.records = []

    def read_sql(self, query: str, schema: Any = None) -> Any:
        return [{"col": "val_1"}, {"col": "val_2"}]

    def insert(self, record: Any) -> None:
        self.records.append(record)


def test_knowledge_instantiation_and_protocols():
    vdb = DummyVectorDb()
    db = DummyDatabase()

    knowledge = Knowledge(
        vector_db=vdb,
        database=db,
        contents=["init document"],
        name="TestKnowledge",
    )

    assert isinstance(knowledge, KnowledgeTransformProtocol)
    assert isinstance(knowledge, KnowledgeProtocol)
    assert knowledge.name == "TestKnowledge"
    assert knowledge.database is db


def test_knowledge_multi_vector_dbs():
    vdb1 = DummyVectorDb("vdb1")
    vdb2 = DummyVectorDb("vdb2")
    db1 = DummyDatabase("db1")
    db2 = DummyDatabase("db2")

    knowledge = Knowledge(
        vector_dbs=[vdb1, vdb2],
        databases=[db1, db2],
    )

    assert len(knowledge.vector_dbs) == 2
    assert len(knowledge.databases) == 2
    assert knowledge.get_vector_db("vdb1") is vdb1
    assert knowledge.get_vector_db("vdb2") is vdb2

    # Store into all vector DBs
    knowledge.store([Document(content="multi vector db test")])
    assert len(vdb1.docs) == 1
    assert len(vdb2.docs) == 1

    # Search across all vector DBs (deduplicated)
    results = knowledge.search("multi")
    assert len(results) == 1
    assert results[0].content == "multi vector db test"


def test_knowledge_multi_vector_db_dict_init():
    vdb1 = DummyVectorDb("vdb1")
    vdb2 = DummyVectorDb("vdb2")

    knowledge = Knowledge(vector_dbs={"primary": vdb1, "secondary": vdb2})

    assert knowledge.get_vector_db("primary") is vdb1
    assert knowledge.get_vector_db("secondary") is vdb2

    # Targeted search
    vdb1.insert("h1", [Document(content="doc in primary")])
    vdb2.insert("h2", [Document(content="doc in secondary")])

    res_primary = knowledge.search("doc", vector_db_name="primary")
    assert len(res_primary) == 1
    assert res_primary[0].content == "doc in primary"


def test_knowledge_store_and_search():
    vdb = DummyVectorDb()
    knowledge = Knowledge(vector_db=vdb)

    doc = Document(content="NeoSyntropy framework for agentic workflows")
    knowledge.store([doc])

    assert vdb.upsert_called
    assert len(vdb.docs) == 1

    results = knowledge.search("NeoSyntropy")
    assert len(results) == 1
    assert results[0].content == "NeoSyntropy framework for agentic workflows"


def test_knowledge_transform_from_knowledge_to_knowledge():
    # Source Knowledge instance
    source_vdb = DummyVectorDb()
    source_knowledge = Knowledge(vector_db=source_vdb, contents=["source data 1", "source data 2"])

    # Custom transform logic
    def custom_transform(data: List[Any], **kwargs) -> List[Document]:
        return [Document(content=f"transformed_{item}") for item in data]

    # Destination Knowledge instance
    dest_vdb = DummyVectorDb()
    dest_knowledge = Knowledge(vector_db=dest_vdb)

    # Transformer Knowledge instance
    transformer = Knowledge(transform=custom_transform)

    # Knowledge to Knowledge transformation
    result = transformer.transform(source=source_knowledge, destination=dest_knowledge)

    assert len(result) == 2
    assert result[0].content == "transformed_source data 1"
    assert len(dest_vdb.docs) == 2
    assert dest_vdb.docs[0].content == "transformed_source data 1"


def test_knowledge_async_search():
    vdb1 = DummyVectorDb("v1")
    vdb2 = DummyVectorDb("v2")
    vdb1.insert("hash_1", [Document(content="async test content 1")])
    vdb2.insert("hash_2", [Document(content="async test content 2")])
    knowledge = Knowledge(vector_dbs=[vdb1, vdb2])

    results = asyncio.run(knowledge.asearch("async"))
    assert len(results) == 2
