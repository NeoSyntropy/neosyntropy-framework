"""Delivery: order refund writes refunds and updates the orders table."""

from tests.scenarios.order_refund.scenario import RefundRequest, build_scenario


def test_valid_refund_posts_row_and_marks_order_refunded() -> None:
    scenario = build_scenario()
    result = scenario.run(RefundRequest(order_id="ORD-001", amount=40.0))

    assert not result.rejected
    assert result.final_state == "End"
    refunds = scenario.db.read_sql("SELECT * FROM refunds WHERE order_id = ?", params=("ORD-001",))
    assert len(refunds) == 1
    assert refunds[0]["refund_id"] == "ref_ORD-001"
    assert refunds[0]["amount"] == 40.0
    assert refunds[0]["status"] == "posted"

    orders = scenario.db.read_sql(
        "SELECT status FROM orders WHERE order_id = ?", params=("ORD-001",)
    )
    assert orders[0]["status"] == "refunded"


def test_oversize_refund_does_not_write_refund_row() -> None:
    scenario = build_scenario()
    result = scenario.run(RefundRequest(order_id="ORD-001", amount=999.0))

    assert not result.rejected
    assert result.final_state == "End"
    assert scenario.db.fetch_all("refunds") == []
    orders = scenario.db.read_sql(
        "SELECT status FROM orders WHERE order_id = ?", params=("ORD-001",)
    )
    assert orders[0]["status"] == "paid"


def test_unknown_order_does_not_touch_refunds_table() -> None:
    scenario = build_scenario()
    result = scenario.run(RefundRequest(order_id="ORD-MISSING", amount=10.0))

    assert not result.rejected
    assert scenario.db.fetch_all("refunds") == []
    still_paid = scenario.db.read_sql("SELECT status FROM orders")
    assert still_paid == [{"status": "paid"}]
