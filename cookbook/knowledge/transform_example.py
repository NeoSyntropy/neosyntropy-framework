from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, List

from neosyntropy.knowledge.document import Document
from neosyntropy.knowledge.filesystem import FileSystemKnowledge
from neosyntropy.knowledge.knowledge import Knowledge


def _write_fixture_corpus(base_dir: Path) -> None:
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


@dataclass
class SearchSource:
    knowledge: FileSystemKnowledge
    query: str

    def load(self, **kwargs: object) -> list[Document]:
        return self.knowledge.search(self.query)


def summarize_documents(raw_data: Iterable[Document], **kwargs: object) -> List[Document]:
    summaries: List[Document] = []
    for doc in raw_data:
        lines = [line.strip() for line in doc.content.splitlines() if line.strip()]
        first_line = lines[0] if lines else doc.content.strip()
        summary = f"{doc.name}: {first_line}"
        summaries.append(
            Document(
                name=f"{doc.name}.summary",
                content=summary,
                meta_data={
                    "source_name": doc.name,
                    "source_lines": len(lines),
                    "word_count": len(doc.content.split()),
                },
            )
        )
    return summaries


def main() -> None:
    with TemporaryDirectory(prefix="neosyntropy-knowledge-transform-") as temp_dir:
        base_dir = Path(temp_dir)
        _write_fixture_corpus(base_dir)

        source_knowledge = FileSystemKnowledge(base_dir=str(base_dir))
        transformed_knowledge = Knowledge(transform=summarize_documents, name="summary_knowledge")
        search_source = SearchSource(source_knowledge, "billing")

        transformed = transformed_knowledge.transform(source=search_source)

        print(f"Transformed documents: {len(transformed)}")
        print()
        for doc in transformed:
            print(f"Document: {doc.name}")
            print(doc.content)
            print(f"Metadata: {doc.meta_data}")
            print()


if __name__ == "__main__":
    main()
