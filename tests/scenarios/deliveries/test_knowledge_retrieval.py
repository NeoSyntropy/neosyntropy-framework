"""Delivery: knowledge retrieval logs hits into retrieval_log."""

from tests.scenarios.knowledge_retrieval.scenario import SearchRequest, build_scenario


def test_matching_query_logs_hits_and_leaves_vector_index_intact() -> None:
    scenario = build_scenario()
    result = scenario.run(SearchRequest(query="renewal", request_id="search-1"))

    assert not result.rejected
    assert result.final_state == "End"
    assert result.state.get("hit_count") == 1

    rows = scenario.db.read_sql(
        "SELECT request_id, query, doc_name FROM retrieval_log WHERE request_id = ?",
        params=("search-1",),
    )
    assert len(rows) == 1
    assert rows[0]["query"] == "renewal"
    assert rows[0]["doc_name"] == "renewal_policy.txt"

    # Related vector DB still holds the seeded corpus.
    assert len(scenario.vector_db.docs) == 2
    assert len(scenario.knowledge.search("delivery")) == 1


def test_unmatched_query_writes_no_retrieval_log_rows() -> None:
    scenario = build_scenario()
    result = scenario.run(SearchRequest(query="quantum", request_id="search-miss"))

    assert not result.rejected
    assert result.final_state == "End"
    assert result.state.get("hit_count") == 0
    assert scenario.db.fetch_all("retrieval_log") == []
