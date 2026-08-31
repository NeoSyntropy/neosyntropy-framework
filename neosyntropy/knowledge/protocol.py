"""
Knowledge Protocol
==================
Defines the minimal interface that knowledge implementations must implement.

This protocol enables:
- Custom knowledge bases to be used with agents
- Each implementation defines its own tools and context
- Flexible tool naming (not forced to use 'search')
- Type safety with Protocol typing
"""

from typing import Any, Callable, Iterable, List, Optional, Protocol, runtime_checkable

from neosyntropy.knowledge.document import Document




@runtime_checkable
class KnowledgeProtocol(Protocol):
    """Protocol for a core knowledge base that manages storage and vector DBs."""
    
    vector_dbs: List[Any]
    databases: List[Any]

    def insert(self, data: Any, **kwargs: Any) -> Any:
        """Insert data into the knowledge base."""
        ...
        
    def delete(self, **kwargs: Any) -> Any:
        """Delete data from the knowledge base."""
        ...

    def get(self, **kwargs: Any) -> Any:
        """get items from the knowledge base."""
        ...
        
@runtime_checkable
class KnowledgeTransformProtocol(Protocol):
    """Protocol for knowledge ingestion, transformation, and storage.
    
    This protocol defines the ETL pipeline for a knowledge base. It loads raw data (`load`),
    builds the processing workflow (`build_workflow`), executes the transform (`transform`),
    and stores the processed results (`store`).
    """

    def load(self, source: Any, **kwargs) -> Any:
        """Fetch raw data and perform initial processing (like chunking).
        
        Returns:
            A dataframe, table, or iterable of documents that can be used as input
            for the transformation workflow.
        """
        ...

    def build_transform_fsm(self, **kwargs) -> Any:
        """Build and return an FSM/Workflow to process each document or row.

        The `transform` process will run this FSM (which may contain SchemaNodes, etc.)
        on all data loaded by `load()`.
        """
        ...

    def transform(self, source: "KnowledgeProtocol", destination: Optional["KnowledgeProtocol"] = None, **kwargs) -> Any:
        """Execute the transformation pipeline between knowledge bases.
        
        Gets the knowledge source, uses `build_transform_fsm()` to process/transform 
        the data using an FSM, and stores the results in the knowledge destination.
        """
        ...

    def store(self, data: Any, destination: Any, **kwargs) -> Any:
        """Store the processed results (e.g., into a Vector DB or storage)."""
        ...
       
@runtime_checkable
class KnowledgeReasoningProtocol(Protocol):
    """Protocol for knowledge retrieval and reasoning.
    
    This acts as a retrieval node interface that helps developers build an FSM 
    specifically for reasoning over knowledge. It takes the heavy lifting of 
    retrieval and tools away from the main fsm.
    """

    def build_reasoning_fsm(self, knowledge: KnowledgeProtocol, **kwargs) -> Any:
        """Build and return a knowledge distillation/reasoning FSM.
        
        This workflow is executed to distill information, execute schema steps, 
        and perform reasoning before returning the final context to the user or agent.
        """
        ...

    def search(self, knowledge: KnowledgeProtocol, **kwargs: Any) -> Any:
        """Search the knowledge base or run reasoning over it.
        
        This handles search logic and uses `build_reasoning_workflow()` to process,
        distill, and execute reasoning. The input query/input schema is passed via kwargs.
        """
        ...
        
    async def asearch(self, knowledge: KnowledgeProtocol, **kwargs: Any) -> Any:
        """Async version of search, using `build_reasoning_workflow()` for reasoning."""
        ...



