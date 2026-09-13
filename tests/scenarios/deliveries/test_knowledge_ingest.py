"""Delivery: knowledge ingest lands documents in vector DB and contents table."""

from tests.scenarios.knowledge_ingest.scenario import IngestRequest, build_scenario


def test_ingest_writes_vector_docs_and_contents_rows() -> None:
    scenario = build_scenario()
    result = scenario.run(IngestRequest(corpus_id="corpus-1"))

    assert not result.rejected
    assert result.final_state == "End"
    assert len(scenario.vector_db.docs) == 2
    assert scenario.vector_db.upsert_called

    contents = scenario.db.read_sql("SELECT corpus_id, name FROM knowledge_contents ORDER BY name")
    names = [row["name"] for row in contents]
    assert names == ["refund_policy.txt", "renewal_policy.txt"]
    assert {row["corpus_id"] for row in contents} == {"corpus-1"}

    hits = scenario.knowledge.search("refund")
    assert len(hits) == 1
    assert "auto-approved" in hits[0].content
