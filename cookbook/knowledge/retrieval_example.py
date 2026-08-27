from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from neosyntropy.knowledge.filesystem import FileSystemKnowledge


def _write_fixture_corpus(base_dir: Path) -> None:
    (base_dir / "incident_status.txt").write_text(
        "Incident status: the payment queue is delayed.\n"
        "Owner: finance operations.\n"
        "Next step: confirm the retry job is running.\n",
        encoding="utf-8",
    )
    (base_dir / "support_notes.txt").write_text(
        "Support notes: customer asked about renewal timing.\n"
        "Suggested answer: explain the billing date and the grace period.\n",
        encoding="utf-8",
    )
    (base_dir / "shipping_update.txt").write_text(
        "Shipping update: the parcel is in transit.\n"
        "Expected delivery: tomorrow afternoon.\n",
        encoding="utf-8",
    )


def main() -> None:
    with TemporaryDirectory(prefix="neosyntropy-knowledge-retrieval-") as temp_dir:
        base_dir = Path(temp_dir)
        _write_fixture_corpus(base_dir)

        knowledge = FileSystemKnowledge(base_dir=str(base_dir))
        query = "renewal"
        results = knowledge.search(query)

        print(f"Query: {query}")
        print(f"Matched documents: {len(results)}")

        for doc in results:
            print()
            print(f"Document: {doc.name}")
            print(doc.content)
            if doc.meta_data:
                print(f"Metadata: {doc.meta_data}")


if __name__ == "__main__":
    main()
