"""Bargaining (ultimatum) game state — isolated from CPR SimulationState."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from core.state import EnvironmentSnapshot, TraitProfile

PIE_SIZE = 100.0
PROPOSER_ID = "proposer"
RESPONDER_ID = "responder"


class ProposerOffer(BaseModel):
    """Structured LLM output for the proposer (keep_amount = X for self)."""

    keep_amount: float = Field(ge=0.0, le=PIE_SIZE)
    # Soft cap applied in agents after parse (avoid structured-output None on long text)
    justification: str = Field(max_length=2000)


class ResponderDecision(BaseModel):
    """Structured LLM output for the responder."""

    accept: bool
    justification: str = Field(max_length=2000)


class BargainingMetricsSnapshot(BaseModel):
    """Round-level, Referee-computed metrics (LLM-free)."""

    model_config = ConfigDict(frozen=True)

    round_number: int = Field(ge=0)
    keep_amount: float = Field(ge=0.0)
    offer_to_responder: float = Field(ge=0.0)
    accepted: bool
    proposer_payoff: float = Field(ge=0.0)
    responder_payoff: float = Field(ge=0.0)
    keep_fraction: float = Field(ge=0.0, le=1.0)
    # Proposer coop fidelity proxy: high coop → low keep (expected negative corr)
    proposer_coop_alignment: float = Field(ge=-1.0, le=1.0)
    # Responder coop fidelity proxy: high coop → accept lower offers
    responder_coop_alignment: float = Field(ge=-1.0, le=1.0)


class BargainingProposerView(BaseModel):
    """Read-only input for the proposer agent (never full BargainingState)."""

    model_config = ConfigDict(frozen=True)

    own_trait: TraitProfile
    env_snapshot: EnvironmentSnapshot
    time_pressure: float = Field(ge=0.0, le=1.0)
    resource_scarcity: float = Field(ge=0.0, le=1.0)
    round_number: int = Field(ge=0)
    pie_size: float = Field(gt=0.0)
    recent_history: List[Dict[str, Any]] = Field(default_factory=list)


class BargainingResponderView(BaseModel):
    """Read-only input for the responder agent."""

    model_config = ConfigDict(frozen=True)

    own_trait: TraitProfile
    env_snapshot: EnvironmentSnapshot
    time_pressure: float = Field(ge=0.0, le=1.0)
    resource_scarcity: float = Field(ge=0.0, le=1.0)
    round_number: int = Field(ge=0)
    pie_size: float = Field(gt=0.0)
    current_offer_keep: float = Field(ge=0.0)
    offer_to_me: float = Field(ge=0.0)
    recent_history: List[Dict[str, Any]] = Field(default_factory=list)


class BargainingState(BaseModel):
    """LangGraph state for the 2-player ultimatum bargaining game."""

    experiment_id: str
    run_id: str
    proposer_trait: TraitProfile
    responder_trait: TraitProfile
    max_rounds: int = Field(gt=0)

    env_snapshot: EnvironmentSnapshot
    time_pressure: float = Field(default=0.3, ge=0.0, le=1.0)
    resource_scarcity: float = Field(default=0.3, ge=0.0, le=1.0)

    current_offer: Optional[float] = None  # proposer's keep_amount X
    current_accept: Optional[bool] = None
    current_justification_proposer: Optional[str] = None
    current_justification_responder: Optional[str] = None

    round_number: int = 0
    history: List[Dict[str, Any]] = Field(default_factory=list)
    metrics_history: List[BargainingMetricsSnapshot] = Field(default_factory=list)

    is_terminated: bool = False
    termination_reason: Optional[str] = None

    @property
    def pie_size(self) -> float:
        return float(self.env_snapshot.pool_capacity)
