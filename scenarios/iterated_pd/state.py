"""Iterated Prisoner's Dilemma (2-player, simultaneous, N rounds) state.

Isolated from CPR / bargaining state — no shared pool, no proposer/responder
asymmetry. Both agents move simultaneously each round and observe the full
joint history afterward.

Payoff matrix (classic PD, own choice vs. opponent choice):

    T (temptation to defect) = 5  — I defect, opponent cooperates
    R (reward for mutual coop) = 3  — both cooperate
    P (punishment for mutual defection) = 1  — both defect
    S (sucker's payoff) = 0  — I cooperate, opponent defects

Standard PD ordering holds: T > R > P > S and 2R > T + S, so mutual
cooperation is the group-optimal outcome but individually tempting to
deviate from — the canonical social-dilemma tension.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.state import TraitProfile

AGENT_A_ID = "agent_a"
AGENT_B_ID = "agent_b"

Choice = Literal["C", "D"]

# --- Payoff matrix constants (see module docstring) ---
T_TEMPTATION = 5.0
R_REWARD = 3.0
P_PUNISHMENT = 1.0
S_SUCKER = 0.0

# Keyed by (own_choice, opponent_choice) -> (own_payoff, opponent_payoff)
PAYOFF_MATRIX: dict[tuple[Choice, Choice], tuple[float, float]] = {
    ("C", "C"): (R_REWARD, R_REWARD),
    ("D", "D"): (P_PUNISHMENT, P_PUNISHMENT),
    ("C", "D"): (S_SUCKER, T_TEMPTATION),
    ("D", "C"): (T_TEMPTATION, S_SUCKER),
}


class IPDDecision(BaseModel):
    """Structured LLM output for a single round choice."""

    choice: Choice
    justification: str = Field(max_length=2000)


class IPDRoundMetrics(BaseModel):
    """Round-level, Referee-computed metrics (LLM-free)."""

    model_config = ConfigDict(frozen=True)

    round_number: int = Field(ge=0)
    agent_a_choice: Choice
    agent_b_choice: Choice
    agent_a_payoff: float = Field(ge=0.0)
    agent_b_payoff: float = Field(ge=0.0)
    agent_a_score: float = Field(ge=0.0)  # cumulative through this round
    agent_b_score: float = Field(ge=0.0)
    mutual_cooperation: bool
    mutual_defection: bool


class IPDAgentView(BaseModel):
    """Read-only input for one agent (never the full IPDState).

    Includes the full round-by-round history of BOTH agents' past choices,
    per the design requirement that each agent observes the complete
    joint history, not just its own moves.
    """

    model_config = ConfigDict(frozen=True)

    experiment_id: str = "iterated_pd_v1"
    own_trait: TraitProfile
    round_number: int = Field(ge=0)
    max_rounds: int = Field(gt=0)
    own_id: str
    opponent_id: str
    own_score: float = Field(ge=0.0)
    opponent_score: float = Field(ge=0.0)
    # Full joint history: list of dicts with round_number, agent_a_choice,
    # agent_b_choice, agent_a_payoff, agent_b_payoff.
    history: List[Dict[str, Any]] = Field(default_factory=list)


class IPDState(BaseModel):
    """LangGraph state for the 2-player iterated prisoner's dilemma."""

    experiment_id: str
    run_id: str
    agent_a_trait: TraitProfile
    agent_b_trait: TraitProfile
    max_rounds: int = Field(gt=0)

    round_number: int = 0

    current_choice_a: Optional[Choice] = None
    current_choice_b: Optional[Choice] = None
    current_justification_a: Optional[str] = None
    current_justification_b: Optional[str] = None

    agent_a_score: float = 0.0
    agent_b_score: float = 0.0

    history: List[Dict[str, Any]] = Field(default_factory=list)
    metrics_history: List[IPDRoundMetrics] = Field(default_factory=list)

    is_terminated: bool = False
    termination_reason: Optional[str] = None
