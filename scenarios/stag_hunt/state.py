"""Stag Hunt (2-player, simultaneous, N rounds) state.

Isolated from CPR / bargaining / iterated_pd state. Both agents move
simultaneously each round and observe the full joint history afterward.

Payoff matrix (own choice vs. opponent choice):

    Stag/Stag = (5, 5)  — best mutual outcome, but risky: requires trust
    Hare/Hare = (3, 3)  — safe mutual outcome, risk-dominant
    Hare/Stag = (4, 0)  — hunter of hare gets 4 (safe payoff), the lone
                          stag-hunter gets 0 (stag requires the partner)
    Stag/Hare = (0, 4)  — mirror of the above: lone stag-hunter gets 0,
                          partner playing hare gets 4

Ordering: mutual stag (5,5) > mutual hare (3,3) > hare-when-partner-stags
(4 for the hare player) > stag-when-partner-hares (0 for the stag player).
This is the classic trust/coordination dilemma: stag is payoff-dominant
but risk-dominated by hare.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.state import TraitProfile

AGENT_A_ID = "agent_a"
AGENT_B_ID = "agent_b"

Choice = Literal["stag", "hare"]

# --- Payoff matrix constants (see module docstring) ---
STAG_STAG = 5.0
HARE_HARE = 3.0
HARE_WHEN_PARTNER_STAGS = 4.0  # payoff to the hare player in a Stag/Hare mismatch
STAG_WHEN_PARTNER_HARES = 0.0  # payoff to the stag player in a Stag/Hare mismatch

# Keyed by (own_choice, opponent_choice) -> (own_payoff, opponent_payoff)
PAYOFF_MATRIX: dict[tuple[Choice, Choice], tuple[float, float]] = {
    ("stag", "stag"): (STAG_STAG, STAG_STAG),
    ("hare", "hare"): (HARE_HARE, HARE_HARE),
    ("hare", "stag"): (HARE_WHEN_PARTNER_STAGS, STAG_WHEN_PARTNER_HARES),
    ("stag", "hare"): (STAG_WHEN_PARTNER_HARES, HARE_WHEN_PARTNER_STAGS),
}


class StagHuntDecision(BaseModel):
    """Structured LLM output for a single round choice."""

    choice: Choice
    justification: str = Field(max_length=2000)


class StagHuntRoundMetrics(BaseModel):
    """Round-level, Referee-computed metrics (LLM-free)."""

    model_config = ConfigDict(frozen=True)

    round_number: int = Field(ge=0)
    agent_a_choice: Choice
    agent_b_choice: Choice
    agent_a_payoff: float = Field(ge=0.0)
    agent_b_payoff: float = Field(ge=0.0)
    agent_a_score: float = Field(ge=0.0)  # cumulative through this round
    agent_b_score: float = Field(ge=0.0)
    mutual_stag: bool
    mutual_hare: bool
    miscoordinated: bool  # one stag, one hare


class StagHuntAgentView(BaseModel):
    """Read-only input for one agent (never the full StagHuntState).

    Includes the full round-by-round history of BOTH agents' past choices.
    """

    model_config = ConfigDict(frozen=True)

    experiment_id: str = "stag_hunt_v1"
    own_trait: TraitProfile
    round_number: int = Field(ge=0)
    max_rounds: int = Field(gt=0)
    own_id: str
    opponent_id: str
    own_score: float = Field(ge=0.0)
    opponent_score: float = Field(ge=0.0)
    history: List[Dict[str, Any]] = Field(default_factory=list)


class StagHuntState(BaseModel):
    """LangGraph state for the 2-player iterated stag hunt."""

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
    metrics_history: List[StagHuntRoundMetrics] = Field(default_factory=list)

    is_terminated: bool = False
    termination_reason: Optional[str] = None
