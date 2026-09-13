"""Delivery: Cookbook: group path validation copy lands and declares backend APIs."""

from tests.scenarios.deliveries.cookbook_checks import assert_cookbook_delivery


def test_cookbook_group_path_validation_lands_and_declares_backend_apis() -> None:
    assert_cookbook_delivery("cookbook_group_path_validation")
