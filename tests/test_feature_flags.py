from __future__ import annotations

import pytest

from neosyntropy import FSM, SchemaNode, Workflow
from neosyntropy._features import monitor_enabled, remote_execution_enabled
from neosyntropy.backend import BackendClient
from neosyntropy.control.manager import ControlManager


def _graph():
    node = SchemaNode(
        id="Only",
        prompt="Only.",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    fallback = SchemaNode(
        id="Fallback",
        prompt="Fallback.",
        is_fallback=True,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )
    return Workflow([node], fallback=fallback)


@pytest.mark.parametrize("value", [None, "", "true", "True", "1", "yes", "on"])
def test_feature_flags_require_literal_uppercase_true(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    for name in ("NEOSYNTROPY_MONITOR", "NEO_REMOTE_EXECUTION"):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert monitor_enabled() is False
    assert remote_execution_enabled() is False


def test_monitor_flag_enables_only_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEOSYNTROPY_MONITOR", "TRUE")
    monkeypatch.delenv("NEO_REMOTE_EXECUTION", raising=False)

    assert monitor_enabled() is True
    assert remote_execution_enabled() is False


def test_remote_execution_implies_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEOSYNTROPY_MONITOR", raising=False)
    monkeypatch.setenv("NEO_REMOTE_EXECUTION", "TRUE")

    assert remote_execution_enabled() is True
    assert monitor_enabled() is True


def test_configured_backend_does_not_enable_remote_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEOSYNTROPY_MONITOR", raising=False)
    monkeypatch.delenv("NEO_REMOTE_EXECUTION", raising=False)
    backend = BackendClient("https://example.invalid", api_key="test")

    manager = ControlManager(_graph(), backend=backend)

    assert manager._backend is None
    assert manager._monitor_backend is None


def test_remote_flag_enables_remote_control_and_monitor_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEOSYNTROPY_MONITOR", raising=False)
    monkeypatch.setenv("NEO_REMOTE_EXECUTION", "TRUE")
    backend = BackendClient("https://example.invalid", api_key="test")

    manager = ControlManager(_graph(), backend=backend)

    assert manager._backend is backend
    assert manager._monitor_backend is backend


def test_fsm_load_is_blocked_before_network_when_remote_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEO_REMOTE_EXECUTION", raising=False)

    class Client:
        def get_graph(self, graph_id: str) -> dict[str, object]:
            raise AssertionError("disabled remote loading must not access the backend")

    with pytest.raises(RuntimeError, match="NEO_REMOTE_EXECUTION=TRUE"):
        FSM.load("graph-1", client=Client())
