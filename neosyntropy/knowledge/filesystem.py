"""
FileSystem Knowledge
====================
A Knowledge implementation that allows retrieval from files in a local directory.

Implements the KnowledgeProtocol and provides three tools:
- grep_file: Search for patterns in file contents
- list_files: List files matching a glob pattern
- get_file: Read the full contents of a specific file
"""

from dataclasses import dataclass, field
from os import walk as os_walk
from pathlib import Path
from re import IGNORECASE
from re import compile as re_compile
from re import error as re_error
from re import escape as re_escape
from typing import Any, List, Optional

from neosyntropy.exceptions import PathSecurityError
from neosyntropy.knowledge.document import Document
from neosyntropy.utils.log import log_debug, log_warning
from neosyntropy.utils.path_safety import safe_join_relative_path


@dataclass
class FileSystemKnowledge:
    """Knowledge implementation that searches files in a local directory.

    Implements the KnowledgeProtocol and provides three tools to agents:
    - grep_file(query): Search for patterns in file contents
    - list_files(pattern): List files matching a glob pattern
    - get_file(path): Read the full contents of a specific file

    Example:
        ```python
        # from neosyntropy.agent import Agent  # REMOVED
        from neosyntropy.knowledge.filesystem import FileSystemKnowledge
        from neosyntropy.models.openai import OpenAIChat

        # Create knowledge for a directory
        fs_knowledge = FileSystemKnowledge(base_dir="/path/to/code")

        # Agent automatically gets grep_file, list_files, get_file tools
        # agent = Agent(
            model=OpenAIChat(id="gpt-5.5"),
            knowledge=fs_knowledge,
            search_knowledge=True,
        )

        # Agent can now search, list, and read files
        agent.print_response("Find where main() is defined")
        ```
    """

    base_dir: str
    max_results: int = 50
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(
        default_factory=lambda: [".git", "__pycache__", "node_modules", ".venv", "venv"]
    )
    vector_dbs: List[Any] = field(default_factory=list)
    databases: List[Any] = field(default_factory=list)

    def __post_init__(self):
        self.base_path = Path(self.base_dir).resolve()
        if not self.base_path.exists():
            raise ValueError(f"Directory does not exist: {self.base_dir}")
        if not self.base_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.base_dir}")

    def _should_include_file(self, file_path: Path) -> bool:
        """Check if a file should be included based on patterns."""
        path_str = str(file_path)

        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern in path_str:
                return False

        # Check include patterns (if specified)
        if self.include_patterns:
            import fnmatch

            for pattern in self.include_patterns:
                if fnmatch.fnmatch(file_path.name, pattern):
                    return True
            return False

        return True

    def _list_files(self, query: str, max_results: Optional[int] = None) -> List[Document]:
        """List files matching the query pattern (glob-style)."""
        import fnmatch

        results: List[Document] = []
        limit = max_results or self.max_results

        for root, dirs, files in os_walk(self.base_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(excl in d for excl in self.exclude_patterns)]

            for filename in files:
                if len(results) >= limit:
                    break

                file_path = Path(root) / filename
                if not self._should_include_file(file_path):
                    continue

                rel_path = file_path.relative_to(self.base_path)

                # Match against query pattern (check both filename and relative path)
                if query and query != "*":
                    if not (fnmatch.fnmatch(filename, query) or fnmatch.fnmatch(rel_path.as_posix(), query)):
                        continue
                results.append(
                    Document(
                        name=rel_path.as_posix(),
                        content=rel_path.as_posix(),
                        meta_data={
                            "type": "file_listing",
                            "absolute_path": str(file_path),
                            "extension": file_path.suffix,
                            "size": file_path.stat().st_size,
                        },
                    )
                )

            if len(results) >= limit:
                break

        log_debug(f"Found {len(results)} files matching pattern: {query}")
        return results

    def _get_file(self, query: str) -> List[Document]:
        """Get the contents of a specific file."""
        # Resolve within base_dir and reject path traversal
        try:
            file_path = safe_join_relative_path(self.base_path, query)
        except PathSecurityError:
            return []

        if not file_path.exists():
            log_warning(f"File not found: {query}")
            return []

        if not file_path.is_file():
            log_warning(f"Path is not a file: {query}")
            return []

        content = file_path.read_text(encoding="utf-8", errors="replace")
        rel_path = file_path.relative_to(self.base_path) if file_path.is_relative_to(self.base_path) else file_path

        return [
            Document(
                name=rel_path.as_posix(),
                content=content,
                meta_data={
                    "type": "file_content",
                    "absolute_path": str(file_path),
                    "extension": file_path.suffix,
                    "size": len(content),
                    "lines": content.count("\n") + 1,
                },
            )
        ]

    def _grep(self, query: str, max_results: Optional[int] = None) -> List[Document]:
        """Search for a pattern within file contents."""
        results: List[Document] = []
        limit = max_results or self.max_results

        try:
            pattern = re_compile(query, IGNORECASE)
        except re_error:
            # If not a valid regex, treat as literal string
            pattern = re_compile(re_escape(query), IGNORECASE)

        for root, dirs, files in os_walk(self.base_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(excl in d for excl in self.exclude_patterns)]

            for filename in files:
                if len(results) >= limit:
                    break

                file_path = Path(root) / filename
                if not self._should_include_file(file_path):
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    matches = list(pattern.finditer(content))

                    if matches:
                        # Extract matching lines with context
                        lines = content.split("\n")
                        matching_lines: List[dict[str, Any]] = []

                        for match in matches[:10]:  # Limit matches per file
                            # Find the line number
                            line_start = content.count("\n", 0, match.start())
                            line_num = line_start + 1

                            # Get context (1 line before and after)
                            start_idx = max(0, line_start - 1)
                            end_idx = min(len(lines), line_start + 2)
                            context_lines = lines[start_idx:end_idx]

                            matching_lines.append(
                                {
                                    "line": line_num,
                                    "match": match.group(),
                                    "context": "\n".join(context_lines),
                                }
                            )

                        rel_path = file_path.relative_to(self.base_path)
                        results.append(
                            Document(
                                name=rel_path.as_posix(),
                                content="\n---\n".join(str(m["context"]) for m in matching_lines),
                                meta_data={
                                    "type": "grep_result",
                                    "absolute_path": str(file_path),
                                    "match_count": len(matches),
                                    "matches": matching_lines[:5],  # Include first 5 match details
                                },
                            )
                        )

                except Exception as e:
                    # Skip files that can't be read (binary, permissions, etc.)
                    log_debug(f"Skipping file {file_path}: {e}")
                    continue

            if len(results) >= limit:
                break

        log_debug(f"Found {len(results)} files with matches for: {query}")
        return results

    # ========================================================================
    # Protocol Implementation (get_tools, retrieve)
    # ========================================================================

    def get_tools(self, **kwargs) -> List[Any]:
        """Get tools to expose to the agent.

        Returns three filesystem tools: grep_file, list_files, get_file.

        Args:
            **kwargs: Additional context (unused).

        Returns:
            List of filesystem tools.
        """
        return [
            self._create_grep_tool(),
            self._create_list_files_tool(),
            self._create_get_file_tool(),
        ]

    async def aget_tools(self, **kwargs) -> List[Any]:
        """Async version of get_tools."""
        return self.get_tools(**kwargs)

    def _create_grep_tool(self) -> Any:
        """Create the grep_file tool."""
        from neosyntropy.tools.core.function import Function

        def grep_file(query: str, max_results: int = 20) -> str:
            """Search the knowledge base files for a keyword or pattern.

            Use this tool to find information in the documents. Search for relevant
            terms from the user's question to find answers.

            Args:
                query: The keyword or pattern to search for (e.g., "coffee", "cappuccino", "brewing").
                max_results: Maximum number of files to return (default: 20).

            Returns:
                Matching content from files with context around each match.
            """
            docs = self._grep(query, max_results=max_results)

            if not docs:
                return f"No matches found for: {query}"

            results = []
            for doc in docs:
                results.append(f"### {doc.name}\n{doc.content}")

            return "\n\n".join(results)

        return Function.from_callable(grep_file, name="grep_file")

    def _create_list_files_tool(self) -> Any:
        """Create the list_files tool."""
        from neosyntropy.tools.core.function import Function

        def list_files(pattern: str = "*", max_results: int = 50) -> str:
            """List available files in the knowledge base.

            Use this to see what documents are available to search.

            Args:
                pattern: Glob pattern to match (e.g., "*.md", "*.txt"). Default: "*" for all files.
                max_results: Maximum number of files to return (default: 50).

            Returns:
                List of available file paths.
            """
            docs = self._list_files(pattern, max_results=max_results)

            if not docs:
                return f"No files found matching: {pattern}"

            file_list = [doc.name for doc in docs]
            return f"Found {len(file_list)} files:\n" + "\n".join(f"- {f}" for f in file_list)

        return Function.from_callable(list_files, name="list_files")

    def _create_get_file_tool(self) -> Any:
        """Create the get_file tool."""
        from neosyntropy.tools.core.function import Function

        def get_file(path: str) -> str:
            """Read the full contents of a document from the knowledge base.

            Use this after list_files to read a specific document.

            Args:
                path: Path to the file (e.g., "coffee.md", "guide.txt").

            Returns:
                The full file contents.
            """
            try:
                docs = self._get_file(path)
            except Exception as e:
                log_warning(f"Error reading file {path}: {str(e)}")
                return f"Error reading file {path}: {e}"

            if not docs:
                return f"File not found: {path}"

            doc = docs[0]
            return f"### {doc.name}\n```\n{doc.content}\n```"

        return Function.from_callable(get_file, name="get_file")

    def search(self, query: str | None = None, **kwargs: Any) -> Any:
        """Search the knowledge base for relevant documents or context.

        Uses grep as the default search method. Executes reasoning using `build_reasoning_fsm`.

        Args:
            query: Optional query string.
            **kwargs: Additional parameters including `query` and `max_results`.

        Returns:
            List of Document objects or generic result.
        """
        search_query = query or kwargs.get("query", "")
        max_results = kwargs.get("max_results")
        return self._grep(search_query, max_results=max_results or 10)

    # Backwards compatibility alias
    retrieve = search

    async def asearch(self, **kwargs: Any) -> Any:
        """Async version of search."""
        return self.search(**kwargs)

    # Backwards compatibility alias
    aretrieve = asearch

    def build_reasoning_fsm(self, steps: Optional[List[Any]] = None, **kwargs) -> Any:
        """Build and return a multi-step reasoning FSM.
        
        Uses `ReasoningStep`s to sequence exploration through the filesystem.
        """
        from neosyntropy.core.node import ReasoningNode, ReasoningStep, SchemaNode
        from neosyntropy.core.graph import Workflow
        
        if not steps:
            steps = [
                ReasoningStep(
                    instruction="List available files to find documents relevant to the query.",
                    tools=["list_files"]
                ),
                ReasoningStep(
                    instruction="Search for specific keywords within the identified files.",
                    tools=["grep_file"]
                ),
                ReasoningStep(
                    instruction="Read the full contents of the most relevant files.",
                    tools=["get_file"]
                )
            ]
            
        return ReasoningNode(
            id="fs_reasoning",
            steps=steps,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )

    # ========================================================================
    # KnowledgeTransformProtocol & KnowledgeProtocol Implementation
    # ========================================================================

    def insert(self, data: Any, **kwargs: Any) -> Any:
        """Insert data into the knowledge base (stub for filesystem)."""
        log_warning("insert() is not fully supported for FileSystemKnowledge")
        return self.store(data, destination=self, **kwargs)

    def delete(self, **kwargs: Any) -> Any:
        """Delete data from the knowledge base (stub for filesystem)."""
        log_warning("delete() is not fully supported for FileSystemKnowledge")
        return False

    def load(self, source: Any = None, **kwargs) -> Any:
        """Fetch raw data from the filesystem."""
        return self._list_files("*", max_results=1000)

    def build_transform_fsm(self, **kwargs) -> Any:
        """Build a basic workflow for chunking/transforming file contents."""
        from neosyntropy.core.node import node, SchemaNode
        from neosyntropy.core.graph import Workflow
        
        @node(id="read_and_chunk", input_schema={"type": "object"}, output_schema={"type": "object"})
        def read_and_chunk(state: dict) -> dict:
            # Basic dummy transform logic
            return {"status": "transformed"}
            
        fallback = SchemaNode(
            id="fs_transform_fallback",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            prompt="Fallback logic for filesystem transform",
            is_fallback=True
        )
        return Workflow([read_and_chunk], fallback=fallback, entry=read_and_chunk)

    def transform(self, source: Any = None, destination: Optional[Any] = None, **kwargs) -> Any:
        """Execute the loading and transformation pipeline."""
        from neosyntropy.knowledge.transform import transform as transform_decorator
        
        data = self.load(source, **kwargs)
        processed = [{"content": doc.content, "meta": doc.meta_data} for doc in data]
        if destination:
            self.store(processed, destination, **kwargs)
        return processed

    def store(self, data: Any, destination: Any, **kwargs) -> Any:
        """Store transformed data to the destination (e.g., VectorDb)."""
        if hasattr(destination, "add_documents"):
            destination.add_documents(data)
        return True

