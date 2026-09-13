"""Delivery: billing payment records captured or declined rows."""

from tests.scenarios.billing_payment.scenario import CardRequest, build_scenario


def test_valid_visa_is_captured_in_payments_table() -> None:
    scenario = build_scenario()
    result = scenario.run(
        CardRequest(payment_id="pay_visa", card_number="4111222233334444", amount=25.0)
    )

    assert not result.rejected
    assert result.final_state == "End"
    rows = scenario.db.read_sql(
        "SELECT payment_id, card_last4, amount, status FROM payments WHERE payment_id = ?",
        params=("pay_visa",),
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "captured"
    assert rows[0]["card_last4"] == "4444"
    assert rows[0]["amount"] == 25.0


def test_non_visa_is_declined_in_payments_table() -> None:
    scenario = build_scenario()
    result = scenario.run(
        CardRequest(payment_id="pay_amex", card_number="378282246310005", amount=25.0)
    )

    assert not result.rejected
    assert result.final_state == "End"
    rows = scenario.db.read_sql(
        "SELECT status FROM payments WHERE payment_id = ?", params=("pay_amex",)
    )
    assert rows == [{"status": "declined"}]
    captured = scenario.db.read_sql("SELECT * FROM payments WHERE status = 'captured'")
    assert captured == []
