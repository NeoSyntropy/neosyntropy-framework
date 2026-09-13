"""Delivery: knowledge transform stores summaries and a completed job row."""

from pathlib import Path

from tests.scenarios.knowledge_transform.scenario import build_scenario


def test_transform_writes_summaries_and_job_row(tmp_path: Path) -> None:
    scenario = build_scenario(tmp_path)
    transformed = scenario.run("job-billing")

    assert len(transformed) == 2
    assert len(scenario.vector_db.docs) == 2
    assert any("renewals are billed" in doc.content.lower() for doc in scenario.vector_db.docs)

    jobs = scenario.db.read_sql(
        "SELECT job_id, source_files, output_docs, status FROM transform_jobs"
    )
    assert jobs == [
        {
            "job_id": "job-billing",
            "source_files": 2,
            "output_docs": 2,
            "status": "completed",
        }
    ]

    source_names = {path.name for path in scenario.corpus_dir.glob("*.txt")}
    assert source_names == {"customer_policy.txt", "ops_playbook.txt"}
    assert scenario.corpus_dir.is_dir()
