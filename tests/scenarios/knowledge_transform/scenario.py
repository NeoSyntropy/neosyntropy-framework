"""Scenario: knowledge transform.

Take a filesystem corpus, transform each document into a one-line summary,
and store the result in a destination knowledge vector DB. A
``transform_jobs`` row records that the ETL actually ran.

```mermaid
flowchart LR
    SourceFS["FileSystemKnowledge"] --> Transform
    Transform --> DestVdb["destination vector DB"]
    Transform --> Jobs["transform_jobs table"]
```
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from neosyntropy.knowledge.document import Document
from tests.scenarios.stores import InMemoryVectorDb, ScenarioKnowledge, SqliteDatabase

SCENARIO_ID = "knowledge_transform"
SCENARIO_TITLE = "Knowledge transform"


@dataclass
class CorpusSource:
    """Load file contents so the transform sees real text."""

    base_dir: Path

    def load(self, **kwargs: object) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(self.base_dir.glob("*.txt")):
            documents.append(Document(name=path.name, content=path.read_text(encoding="utf-8")))
        return documents


@dataclass
class KnowledgeTransformScenario:
    db: SqliteDatabase
    source: CorpusSource
    destination: ScenarioKnowledge
    vector_db: InMemoryVectorDb
    corpus_dir: Path

    def run(self, job_id: str = "job-1") -> list[Document]:
        def summarize(raw_data: Iterable[Document], **kwargs: object) -> list[Document]:
            summaries: list[Document] = []
            for doc in raw_data:
                lines = [line.strip() for line in doc.content.splitlines() if line.strip()]
                first_line = lines[0] if lines else doc.content.strip()
                summaries.append(
                    Document(
                        name=f"{doc.name}.summary",
                        content=f"{doc.name}: {first_line}",
                        meta_data={
                            "source_name": doc.name,
                            "source_lines": len(lines),
                            "job_id": job_id,
                        },
                    )
                )
            return summaries

        transformer = ScenarioKnowledge(transform=summarize, name="summary_pipeline")
        transformed = transformer.transform(source=self.source, destination=self.destination)
        self.db.insert_row(
            "transform_jobs",
            {
                "job_id": job_id,
                "source_files": len(self.source.load()),
                "output_docs": len(transformed),
                "status": "completed",
            },
        )
        return transformed


def _write_corpus(base_dir: Path) -> None:
    (base_dir / "customer_policy.txt").write_text(
        "Policy: renewals are billed on the first of the month.\n"
        "Grace period: 7 days after the due date.\n",
        encoding="utf-8",
    )
    (base_dir / "ops_playbook.txt").write_text(
        "Playbook: if billing retries fail, notify support and finance.\n"
        "Escalation: open an incident if the queue is blocked for 30 minutes.\n",
        encoding="utf-8",
    )


def build_scenario(tmp_path: Path) -> KnowledgeTransformScenario:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    _write_corpus(corpus_dir)

    db = SqliteDatabase(name="transform_db")
    db.execute(
        "CREATE TABLE transform_jobs ("
        "job_id TEXT PRIMARY KEY, "
        "source_files INTEGER NOT NULL, "
        "output_docs INTEGER NOT NULL, "
        "status TEXT NOT NULL)"
    )
    vector_db = InMemoryVectorDb(name="summaries")
    destination = ScenarioKnowledge(vector_db=vector_db, name="summary_knowledge")
    source = CorpusSource(base_dir=corpus_dir)
    return KnowledgeTransformScenario(
        db=db,
        source=source,
        destination=destination,
        vector_db=vector_db,
        corpus_dir=corpus_dir,
    )
