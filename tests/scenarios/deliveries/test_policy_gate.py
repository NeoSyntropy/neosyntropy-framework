"""Delivery: policy gate persists approve/deny rows in decisions."""

from tests.scenarios.policy_gate.scenario import PolicyRequest, build_scenario


def test_eligible_account_writes_approved_decision() -> None:
    scenario = build_scenario()
    result = scenario.run(
        PolicyRequest(request_id="req_old", text="refund please", account_age_days=45)
    )

    assert not result.rejected
    assert result.final_state == "End"
    rows = scenario.db.read_sql(
        "SELECT request_id, eligible, decision FROM decisions WHERE request_id = ?",
        params=("req_old",),
    )
    assert rows == [{"request_id": "req_old", "eligible": 1, "decision": "approved"}]


def test_new_account_writes_denied_decision() -> None:
    scenario = build_scenario()
    result = scenario.run(
        PolicyRequest(request_id="req_new", text="refund please", account_age_days=3)
    )

    assert not result.rejected
    assert result.final_state == "End"
    rows = scenario.db.read_sql(
        "SELECT eligible, decision FROM decisions WHERE request_id = ?",
        params=("req_new",),
    )
    assert rows == [{"eligible": 0, "decision": "denied"}]
    approved = scenario.db.read_sql("SELECT * FROM decisions WHERE decision = 'approved'")
    assert approved == []
