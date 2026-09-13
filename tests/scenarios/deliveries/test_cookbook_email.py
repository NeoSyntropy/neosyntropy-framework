"""Delivery: Cookbook: email tools copy lands and declares backend APIs."""

from tests.scenarios.deliveries.cookbook_checks import assert_cookbook_delivery


def test_cookbook_email_lands_and_declares_backend_apis() -> None:
    assert_cookbook_delivery("cookbook_email")
