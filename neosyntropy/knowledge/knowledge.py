"""
Knowledge Class
===============
Core Knowledge implementation for managing knowledge bases, vector DBs,
database connections, embedders, rerankers, and ETL transformations.

Implements KnowledgeTransformProtocol, KnowledgeReasoningProtocol, and VectorDbKnowledgeProtocol,
enabling loading, transforming, storing, and multi-vector-db search capabilities.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from neosyntropy.knowledge.content import Content, ContentStatus
from neosyntropy.knowledge.document import Document
from neosyntropy.knowledge.protocol import (
    KnowledgeProtocol,
    KnowledgeReasoningProtocol,
    KnowledgeTransformProtocol,
)
from neosyntropy.knowledge.remote_content.base import BaseStorageConfig
from neosyntropy.knowledge.remote_knowledge import RemoteKnowledge
from neosyntropy.utils.log import log_debug, log_warning
from neosyntropy.utils.string import generate_id


class Knowledge(RemoteKnowledge, KnowledgeTransformProtocol, KnowledgeProtocol):
    """Unified Knowledge class representing a knowledge base.

    Can store and query data using vector databases, relational/NoSQL databases,
    embedders, rerankers, and custom transformation pipelines. Can hold single or
    multiple vector DB instances and database instances.

    Example:
        ```python
        from neosyntropy.knowledge import Knowledge
        from neosyntropy.vectordb.lancedb import LanceDb
        from neosyntropy.vectordb.chroma import ChromaDb

        vdb1 = LanceDb(table_name="docs_1")
        vdb2 = ChromaDb(collection_name="docs_2")

        # Knowledge holding multiple vector DB instances
        knowledge = Knowledge(vector_dbs=[vdb1, vdb2], databases=[postgres_db])

        # Search across all vector DB instances
        results = knowledge.search("quantum computing")
        ```
    """

    def __init__(
        self,
        *,
        vector_db: Optional[Union[Any, List[Any], Dict[str, Any]]] = None,
        vector_dbs: Optional[Union[List[Any], Dict[str, Any]]] = None,
        database: Optional[Union[Any, List[Any], Dict[str, Any]]] = None,
        databases: Optional[Union[List[Any], Dict[str, Any]]] = None,
        contents_db: Optional[Any] = None,
        embedder: Optional[Any] = None,
        reranker: Optional[Any] = None,
        transform: Optional[Any] = None,
        contents: Optional[List[Any]] = None,
        content_sources: Optional[List[BaseStorageConfig]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        topics: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        self.vector_dbs: List[Any] = []
        self.vector_dbs_map: Dict[str, Any] = {}

        self.databases: List[Any] = []
        self.databases_map: Dict[str, Any] = {}

        # Register vector DBs (single, list, or dict)
        if vector_dbs is not None:
            self._register_vector_dbs(vector_dbs)
        if vector_db is not None:
            self._register_vector_dbs(vector_db)

        # Register databases (single, list, or dict)
        if databases is not None:
            self._register_databases(databases)
        if database is not None:
            self._register_databases(database)
        if contents_db is not None:
            self._register_databases(contents_db)

        self.embedder = embedder
        self.reranker = reranker
        self.transform_pipeline = transform
        self.contents: List[Any] = contents if contents is not None else []
        self.content_sources: Optional[List[BaseStorageConfig]] = content_sources
        self.name = name or self.__class__.__name__
        self.description = description
        self.topics: List[str] = topics or []
        self.kwargs = kwargs

    def _register_vector_dbs(self, vdb_arg: Any) -> None:
        if isinstance(vdb_arg, list):
            for item in vdb_arg:
                self.add_vector_db(item)
        elif isinstance(vdb_arg, dict):
            for k, v in vdb_arg.items():
                self.add_vector_db(v, name=k)
        elif vdb_arg is not None:
            self.add_vector_db(vdb_arg)

    def _register_databases(self, db_arg: Any) -> None:
        if isinstance(db_arg, list):
            for item in db_arg:
                self.add_database(item)
        elif isinstance(db_arg, dict):
            for k, v in db_arg.items():
                self.add_database(v, name=k)
        elif db_arg is not None:
            self.add_database(db_arg)

    # ==========================================
    # Backward Compatibility Properties
    # ==========================================

    @property
    def vector_db(self) -> Optional[Any]:
        return self.vector_dbs[0] if self.vector_dbs else None

    @vector_db.setter
    def vector_db(self, value: Optional[Any]) -> None:
        if value is not None:
            if not self.vector_dbs or self.vector_dbs[0] != value:
                self.add_vector_db(value)
        elif self.vector_dbs:
            self.vector_dbs.clear()
            self.vector_dbs_map.clear()

    @property
    def database(self) -> Optional[Any]:
        return self.databases[0] if self.databases else None

    @database.setter
    def database(self, value: Optional[Any]) -> None:
        if value is not None:
            if not self.databases or self.databases[0] != value:
                self.add_database(value)
        elif self.databases:
            self.databases.clear()
            self.databases_map.clear()

    @property
    def contents_db(self) -> Optional[Any]:
        return self.database

    @contents_db.setter
    def contents_db(self, value: Optional[Any]) -> None:
        self.database = value

    # ==========================================
    # VectorDbKnowledgeProtocol Implementation
    # ==========================================

    def add_vector_db(self, vector_db: Any, name: Optional[str] = None) -> None:
        """Register a vector database instance in this knowledge base."""
        vdb_name = name or getattr(vector_db, "name", None) or getattr(vector_db, "id", None) or f"vdb_{len(self.vector_dbs)}"
        if vector_db not in self.vector_dbs:
            self.vector_dbs.append(vector_db)
        self.vector_dbs_map[vdb_name] = vector_db

    def get_vector_db(self, name: str) -> Optional[Any]:
        """Retrieve a specific vector database instance by name or ID."""
        return self.vector_dbs_map.get(name)

    def add_database(self, database: Any, name: Optional[str] = None) -> None:
        """Register a database instance (PostgresReader, MongoReader, etc.)."""
        db_name = name or getattr(database, "name", None) or f"db_{len(self.databases)}"
        if database not in self.databases:
            self.databases.append(database)
        self.databases_map[db_name] = database

    def get_database(self, name: str) -> Optional[Any]:
        """Retrieve a specific database instance by name."""
        return self.databases_map.get(name)

    def search_vector_dbs(self, query: str, limit: int = 5, **kwargs: Any) -> List[Document]:
        """Search across all managed vector database instances."""
        vector_db_name = kwargs.get("vector_db_name")
        if vector_db_name:
            target_vdb = self.get_vector_db(vector_db_name)
            if target_vdb and hasattr(target_vdb, "search"):
                return target_vdb.search(query, limit=limit, **kwargs)
            return []

        all_results: List[Document] = []
        seen_contents = set()

        for vdb in self.vector_dbs:
            if hasattr(vdb, "search"):
                try:
                    res = vdb.search(query, limit=limit, **kwargs)
                    for doc in res:
                        if doc.content not in seen_contents:
                            seen_contents.add(doc.content)
                            all_results.append(doc)
                except Exception as exc:
                    log_warning(f"Error searching vector db {vdb}: {exc}")

        if self.reranker is not None and all_results:
            if hasattr(self.reranker, "rerank"):
                all_results = self.reranker.rerank(query, all_results)

        return all_results[:limit]

    async def asearch_vector_dbs(self, query: str, limit: int = 5, **kwargs: Any) -> List[Document]:
        """Async search across all managed vector database instances."""
        import asyncio

        vector_db_name = kwargs.get("vector_db_name")
        if vector_db_name:
            target_vdb = self.get_vector_db(vector_db_name)
            if target_vdb:
                if hasattr(target_vdb, "async_search"):
                    return await target_vdb.async_search(query, limit=limit, **kwargs)
                elif hasattr(target_vdb, "search"):
                    return target_vdb.search(query, limit=limit, **kwargs)
            return []

        async def _search_one(vdb: Any) -> List[Document]:
            if hasattr(vdb, "async_search"):
                return await vdb.async_search(query, limit=limit, **kwargs)
            elif hasattr(vdb, "search"):
                return vdb.search(query, limit=limit, **kwargs)
            return []

        tasks = [_search_one(vdb) for vdb in self.vector_dbs]
        if not tasks:
            return []

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        all_results: List[Document] = []
        seen_contents = set()

        for res in batch_results:
            if isinstance(res, list):
                for doc in res:
                    if doc.content not in seen_contents:
                        seen_contents.add(doc.content)
                        all_results.append(doc)

        if self.reranker is not None and all_results:
            if hasattr(self.reranker, "async_rerank"):
                all_results = await self.reranker.async_rerank(query, all_results)
            elif hasattr(self.reranker, "rerank"):
                all_results = self.reranker.rerank(query, all_results)

        return all_results[:limit]

    # ==========================================
    # KnowledgeTransformProtocol Implementation
    # ==========================================

    def load(self, source: Any = None, **kwargs: Any) -> Any:
        """Fetch raw data, documents, or records from source, databases, vector DBs, or internal contents.

        Args:
            source: Source Knowledge instance, list, dataframe, SQL query, or string.
            **kwargs: Additional parameters (e.g., query, schema, limit).

        Returns:
            Loaded raw data or list of documents.
        """
        if source is not None:
            if isinstance(source, Knowledge):
                return source.load(**kwargs)
            if hasattr(source, "load"):
                return source.load(**kwargs)
            if hasattr(source, "read_sql") and "query" in kwargs:
                schema = kwargs.get("schema")
                return source.read_sql(kwargs["query"], schema=schema)
            if isinstance(source, list):
                return source
            if isinstance(source, str):
                return [Document(content=source)]
            return source

        if self.contents:
            return self.contents

        # Try loading from registered databases
        if self.databases and "query" in kwargs:
            db_name = kwargs.get("db_name")
            target_db = self.get_database(db_name) if db_name else self.databases[0]
            if hasattr(target_db, "read_sql"):
                schema = kwargs.get("schema")
                return target_db.read_sql(kwargs["query"], schema=schema)

        # Try searching vector dbs
        if self.vector_dbs:
            query = kwargs.get("query", "")
            limit = kwargs.get("limit", 100)
            return self.search_vector_dbs(query, limit=limit, **kwargs)

        return []

    def build_transform_fsm(self, **kwargs: Any) -> Any:
        """Build and return an FSM/Workflow to process each document or row."""
        if hasattr(self.transform_pipeline, "build_transform_fsm"):
            return self.transform_pipeline.build_transform_fsm(**kwargs)
        return None

    def transform(
        self,
        source: Optional[Any] = None,
        destination: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the transformation pipeline from source Knowledge (or data) to destination Knowledge (or self).

        Args:
            source: Source Knowledge instance, raw data, or iterable. Defaults to self if None.
            destination: Destination Knowledge instance, vector_db, or DB. Defaults to self if None.
            **kwargs: Additional parameters passed to load/transform/store.

        Returns:
            Transformed data or list of documents.
        """
        # 1. Load from source
        if source is None:
            raw_data = self.load(**kwargs)
        elif isinstance(source, Knowledge) or hasattr(source, "load"):
            raw_data = source.load(**kwargs)
        else:
            raw_data = self.load(source=source, **kwargs)

        # 2. Execute transform pipeline
        transformed_data = raw_data
        if self.transform_pipeline is not None:
            if hasattr(self.transform_pipeline, "transform"):
                transformed_data = self.transform_pipeline.transform(raw_data, destination=destination, **kwargs)
            elif callable(self.transform_pipeline):
                transformed_data = self.transform_pipeline(raw_data, **kwargs)
        else:
            fsm = self.build_transform_fsm(**kwargs)
            if fsm is not None and hasattr(fsm, "run_batch") and isinstance(raw_data, list):
                transformed_data = fsm.run_batch(raw_data)

        # 3. Store into destination
        target = destination if destination is not None else self
        if target is not self:
            if isinstance(target, Knowledge) or hasattr(target, "store"):
                target.store(transformed_data, **kwargs)
            elif hasattr(target, "insert"):
                target.insert(transformed_data, **kwargs)
            elif hasattr(target, "upsert"):
                target.upsert(transformed_data, **kwargs)
        else:
            self.store(transformed_data, **kwargs)

        return transformed_data

    def store(self, data: Any, destination: Optional[Any] = None, **kwargs: Any) -> Any:
        """Store processed results into vector DBs, database, contents list, or destination Knowledge.

        Args:
            data: Data or documents to store.
            destination: Optional destination target override.
        """
        target = destination if destination is not None else self

        if target is not self:
            if isinstance(target, Knowledge) or hasattr(target, "store"):
                return target.store(data, **kwargs)
            if hasattr(target, "insert"):
                return target.insert(data, **kwargs)
            if hasattr(target, "upsert"):
                return target.upsert(data, **kwargs)

        docs_to_insert: List[Document] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, Document):
                    docs_to_insert.append(item)
                elif isinstance(item, str):
                    docs_to_insert.append(Document(content=item))
                elif isinstance(item, dict):
                    content_text = item.get("content") or str(item)
                    docs_to_insert.append(Document(content=content_text, meta_data=item))
                else:
                    self.contents.append(item)
        elif isinstance(data, Document):
            docs_to_insert.append(data)
        elif data is not None:
            self.contents.append(data)

        if docs_to_insert:
            self.contents.extend(docs_to_insert)

            target_vdb_name = kwargs.get("vector_db_name")
            target_vdbs = [self.get_vector_db(target_vdb_name)] if target_vdb_name else self.vector_dbs

            for vdb in target_vdbs:
                if vdb is not None:
                    content_hash = kwargs.get("content_hash") or generate_id(16)
                    if hasattr(vdb, "upsert") and vdb.upsert_available():
                        vdb.upsert(content_hash, docs_to_insert)
                    elif hasattr(vdb, "insert"):
                        vdb.insert(content_hash, docs_to_insert)

        return True

    # ==========================================
    # KnowledgeProtocol Implementation
    # ==========================================
    
    def insert(self, data: Any, **kwargs: Any) -> Any:
        """Insert data into the knowledge base."""
        return self.store(data, **kwargs)
        
    def delete(self, **kwargs: Any) -> Any:
        """Delete data from the knowledge base."""
        target_vdb_name = kwargs.get("vector_db_name")
        target_vdbs = [self.get_vector_db(target_vdb_name)] if target_vdb_name else self.vector_dbs
        for vdb in target_vdbs:
            if vdb is not None and hasattr(vdb, "delete"):
                vdb.delete(**kwargs)
                
        # Handle database deletions if applicable
        db_name = kwargs.get("db_name")
        target_dbs = [self.get_database(db_name)] if db_name else self.databases
        for db in target_dbs:
            if db is not None and hasattr(db, "delete"):
                db.delete(**kwargs)
        return True

    def search(self, query: str, limit: int = 5, **kwargs: Any) -> List[Document]:
        """Search the knowledge base for relevant documents or context across vector DBs and contents."""
        results: List[Document] = []

        if self.vector_dbs:
            results = self.search_vector_dbs(query, limit=limit, **kwargs)
        elif self.contents:
            for item in self.contents:
                if isinstance(item, Document):
                    if query.lower() in item.content.lower():
                        results.append(item)
                elif isinstance(item, str):
                    if query.lower() in item.lower():
                        results.append(Document(content=item))
                elif isinstance(item, Content):
                    name = item.name or ""
                    desc = item.description or ""
                    if query.lower() in name.lower() or query.lower() in desc.lower():
                        results.append(Document(content=f"{name}: {desc}", name=name))

        if getattr(self, "reranker", None) is not None and results and not self.vector_dbs:
            if hasattr(self.reranker, "rerank"):
                results = self.reranker.rerank(query, results)

        return results[:limit]

    async def asearch(self, query: str, limit: int = 5, **kwargs: Any) -> List[Document]:
        """Async version of search."""
        if self.vector_dbs:
            return await self.asearch_vector_dbs(query, limit=limit, **kwargs)

        return self.search(query, limit=limit, **kwargs)

    # ==========================================
    # RemoteKnowledge Loader Helper Implementation
    # ==========================================

    def _build_content_hash(self, content: Content) -> str:
        """Build content hash for tracking and deduplication."""
        identifier = content.path or content.url or content.name or str(content.id)
        import hashlib
        return hashlib.md5(identifier.encode("utf-8")).hexdigest()

    def _should_skip(self, content: Content, upsert: bool, skip_if_exists: bool) -> bool:
        """Check if content insertion should be skipped based on existence in vector_db/contents_db."""
        if not skip_if_exists:
            return False
        if content.content_hash:
            for vdb in self.vector_dbs:
                if hasattr(vdb, "content_hash_exists") and vdb.content_hash_exists(content.content_hash):
                    return True
        return False

    def _prepare_documents_for_insert(self, documents: List[Document], content_id: str) -> None:
        """Attach metadata to documents prior to insertion."""
        for doc in documents:
            if not doc.content_id:
                doc.content_id = content_id

    def _handle_vector_db_insert(self, content: Content, documents: List[Document], upsert: bool) -> None:
        """Insert or upsert documents into vector_dbs."""
        if not self.vector_dbs or not content.content_hash:
            return
        for vdb in self.vector_dbs:
            if upsert and hasattr(vdb, "upsert_available") and vdb.upsert_available():
                vdb.upsert(content.content_hash, documents)
            elif hasattr(vdb, "insert"):
                vdb.insert(content.content_hash, documents)

    async def _ahandle_vector_db_insert(self, content: Content, documents: List[Document], upsert: bool) -> None:
        """Async insert or upsert documents into vector_dbs."""
        if not self.vector_dbs or not content.content_hash:
            return
        for vdb in self.vector_dbs:
            if upsert and hasattr(vdb, "upsert_available") and vdb.upsert_available():
                await vdb.async_upsert(content.content_hash, documents)
            elif hasattr(vdb, "async_insert"):
                await vdb.async_insert(content.content_hash, documents)

    def _insert_contents_db(self, content: Content) -> None:
        """Insert content record into contents_db/databases."""
        for db in self.databases:
            if hasattr(db, "insert"):
                db.insert(content)
        self.contents.append(content)

    async def _ainsert_contents_db(self, content: Content) -> None:
        """Async insert content record into contents_db/databases."""
        for db in self.databases:
            if hasattr(db, "async_insert"):
                await db.async_insert(content)
            elif hasattr(db, "insert"):
                db.insert(content)
        self.contents.append(content)

    def _update_content(self, content: Content) -> None:
        """Update content record in databases/contents."""
        for db in self.databases:
            if hasattr(db, "update"):
                db.update(content)

    async def _aupdate_content(self, content: Content) -> None:
        """Async update content record."""
        for db in self.databases:
            if hasattr(db, "async_update"):
                await db.async_update(content)
            elif hasattr(db, "update"):
                db.update(content)
