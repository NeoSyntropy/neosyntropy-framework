"""Shared delivery assertions for cookbook scenario copies."""

from __future__ import annotations

from tests.scenarios.cookbook_support import SPECS, build_from_spec


def assert_cookbook_delivery(spec_id: str) -> None:
    spec = SPECS[spec_id]
    scenario = build_from_spec(spec_id)
    result = scenario.run()

    apis = scenario.db.read_sql(
        "SELECT method, path, when_to_validate, purpose FROM backend_apis "
        "WHERE scenario_id = ? ORDER BY path, method",
        params=(spec_id,),
    )
    expected = sorted(
        [
            {
                "method": api.method,
                "path": api.path,
                "when_to_validate": api.when,
                "purpose": api.purpose,
            }
            for api in spec.backend_apis
        ],
        key=lambda row: (row["path"], row["method"]),
    )
    assert apis == expected, spec_id
    assert apis, spec_id

    runs = scenario.db.read_sql(
        "SELECT final_state, rejected FROM cookbook_runs WHERE scenario_id = ?",
        params=(spec_id,),
    )
    assert runs, spec_id
    if spec.require_end:
        assert runs[0]["rejected"] == 0, (spec_id, runs[0])
    if spec.kind == "fsm":
        if spec.require_end:
            assert not result.rejected, result.rejection
            assert result.final_state == "End"
        executed = {item.node_id for step in result.steps for item in step.results}
        missing = [node_id for node_id in spec.expected_nodes if node_id not in executed]
        assert missing == [], f"{spec_id} missing nodes {missing}; ran {sorted(executed)}"

    if spec.kind == "filesystem":
        files = scenario.db.read_sql(
            "SELECT name FROM local_artifacts WHERE scenario_id = ?",
            params=(spec_id,),
        )
        assert files == [{"name": "notes.txt"}]

    if spec.kind == "knowledge_retrieval":
        docs = scenario.db.read_sql(
            "SELECT name FROM local_artifacts WHERE scenario_id = ?",
            params=(spec_id,),
        )
        names = {row["name"] for row in docs}
        assert "support_notes.txt" in names

    if spec.kind == "knowledge_transform":
        docs = scenario.db.read_sql(
            "SELECT name FROM local_artifacts WHERE scenario_id = ?",
            params=(spec_id,),
        )
        assert docs, spec_id

    if spec.kind == "email":
        rows = scenario.db.read_sql(
            "SELECT detail FROM local_artifacts WHERE scenario_id = ?",
            params=(spec_id,),
        )
        assert rows
        assert "NeoSyntropy cookbook email" in rows[0]["detail"]

    if spec.kind == "web_search":
        rows = scenario.db.read_sql(
            "SELECT name FROM local_artifacts WHERE scenario_id = ?",
            params=(spec_id,),
        )
        assert rows[0]["name"] == "Python pathlib Path documentation"
