"""Scenario: policy gate.

A deterministic eligibility rule (account age >= 30 days) decides approve
or deny. The decision is written to the decisions table so deliveries can
prove the gate actually committed.

```mermaid
flowchart LR
    CheckPolicy --> EligibilityGate
    EligibilityGate -->|eligible| ApproveRequest --> End
    EligibilityGate -->|not eligible| DenyRequest --> End
    CheckPolicy -.-> OutOfScope
```
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from neosyntropy import (
    FSM,
    DeterministicRouter,
    NodeContext,
    OpenInput,
    TextOutput,
    edge_deterministic,
    edge_fallback,
    node,
)
from tests.scenarios.stores import SqliteDatabase, run_fsm

SCENARIO_ID = "policy_gate"
SCENARIO_TITLE = "Policy gate"


class PolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    text: str
    account_age_days: int


class DecisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    decision: str


@dataclass
class PolicyGateScenario:
    fsm: FSM
    db: SqliteDatabase

    def run(self, payload: PolicyRequest):
        return run_fsm(self.fsm, payload)


def _schema(db: SqliteDatabase) -> None:
    db.execute(
        "CREATE TABLE decisions ("
        "request_id TEXT PRIMARY KEY, "
        "account_age_days INTEGER NOT NULL, "
        "eligible INTEGER NOT NULL, "
        "decision TEXT NOT NULL)"
    )


def build_scenario() -> PolicyGateScenario:
    db = SqliteDatabase(name="policy_db")
    _schema(db)

    @node(id="CheckPolicy", input_schema=PolicyRequest, output_schema=TextOutput)
    def check_policy(ctx: NodeContext) -> object:
        eligible = int(ctx.input["account_age_days"]) >= 30
        return ctx.result(
            output={"message": "policy evaluated"},
            state_updates={
                "eligible": eligible,
                "request_id": ctx.input["request_id"],
                "account_age_days": ctx.input["account_age_days"],
            },
        )

    @node(id="ApproveRequest", input_schema=OpenInput, output_schema=DecisionOutput)
    def approve_request(ctx: NodeContext) -> object:
        request_id = str(ctx.state["request_id"])
        db.insert_row(
            "decisions",
            {
                "request_id": request_id,
                "account_age_days": int(ctx.state["account_age_days"]),
                "eligible": 1,
                "decision": "approved",
            },
        )
        return ctx.result(
            output={"request_id": request_id, "decision": "approved"},
            state_updates={"approved": True},
            next_state="End",
        )

    @node(id="DenyRequest", input_schema=OpenInput, output_schema=DecisionOutput)
    def deny_request(ctx: NodeContext) -> object:
        request_id = str(ctx.state["request_id"])
        db.insert_row(
            "decisions",
            {
                "request_id": request_id,
                "account_age_days": int(ctx.state["account_age_days"]),
                "eligible": 0,
                "decision": "denied",
            },
        )
        return ctx.result(
            output={"request_id": request_id, "decision": "denied"},
            state_updates={"approved": False},
            next_state="End",
        )

    @node(id="OutOfScope", is_fallback=True, input_schema=OpenInput, output_schema=TextOutput)
    def out_of_scope(ctx: NodeContext) -> object:
        return ctx.result(output={"message": "Cannot process request."})

    eligibility = DeterministicRouter(
        id="EligibilityGate",
        rules=[
            (lambda ctx: ctx.state.get("eligible") is True, "ApproveRequest"),
            (lambda ctx: ctx.state.get("eligible") is False, "DenyRequest"),
        ],
    )
    fsm = FSM(
        entry=check_policy,
        nodes=[check_policy, approve_request, deny_request, out_of_scope],
        routers=[eligibility],
        edges=[
            edge_deterministic("CheckPolicy", "EligibilityGate"),
            edge_deterministic("EligibilityGate", "ApproveRequest"),
            edge_deterministic("EligibilityGate", "DenyRequest"),
            edge_deterministic("ApproveRequest", "End"),
            edge_deterministic("DenyRequest", "End"),
            edge_fallback("CheckPolicy", "OutOfScope"),
        ],
    )
    return PolicyGateScenario(fsm=fsm, db=db)
