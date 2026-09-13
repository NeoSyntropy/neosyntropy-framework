"""Delivery: Cookbook: FSM path validation copy lands and declares backend APIs."""

from tests.scenarios.deliveries.cookbook_checks import assert_cookbook_delivery


def test_cookbook_fsm_path_validation_lands_and_declares_backend_apis() -> None:
    assert_cookbook_delivery("cookbook_fsm_path_validation")
