from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Optional
from enum import Enum

# --- 4.1 TraitProfile ---
class TraitProfile(BaseModel):
    model_config = ConfigDict(frozen=True)
    agent_id: str
    cooperation_assigned: float = Field(ge=0.0, le=1.0)
    risk_tolerance_assigned: float = Field(ge=0.0, le=1.0)
    profile_label: str

# --- 4.2 EnvironmentSnapshot ---
class EnvironmentSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    pool_current: float = Field(ge=0.0)
    pool_capacity: float = Field(gt=0.0)
    regen_rate: float = Field(gt=0.0)
    max_extractable_this_round: float = Field(ge=0.0)
    round_number: int = Field(ge=0)
    is_collapsed: bool = False

# --- 4.3 AgentDecision ---
class AgentDecision(BaseModel):
    agent_id: str
    round_number: int
    extraction_amount: float = Field(ge=0.0)
    justification: str = Field(max_length=500)
    declared_max: float = Field(ge=0.0)

# --- 4.4 ShockEvent ---
class ShockType(str, Enum):
    CAPACITY_DROP = "capacity_drop"
    REGEN_BOOST = "regen_boost"
    DEMAND_SURGE = "demand_surge"

class ShockEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    round_number: int
    shock_type: ShockType
    magnitude: float
    seed_source: str

# --- 4.5 MetricsSnapshot ---
class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    round_number: int
    gini_coefficient: float = Field(ge=0.0, le=1.0)
    cooperation_score_avg: float = Field(ge=0.0, le=1.0)
    total_extraction: float = Field(ge=0.0)
    pool_after: float = Field(ge=0.0)
    constraint_violations: int = Field(ge=0, default=0)

# --- 4.6 SimulationState ---
class SimulationState(BaseModel):
    experiment_id: str
    run_id: str
    agent_traits: Dict[str, TraitProfile]
    shock_schedule: List[ShockEvent]
    max_rounds: int = Field(gt=0)
    environment: EnvironmentSnapshot
    round_decisions: List[AgentDecision] = []
    metrics_history: List[MetricsSnapshot] = []
    round_number: int = 0
    is_terminated: bool = False
    termination_reason: Optional[str] = None

# --- 4.7 AgentInputView (Read-Only Enforcement) ---
class AgentInputView(BaseModel):
    model_config = ConfigDict(frozen=True)
    own_trait: TraitProfile
    environment: EnvironmentSnapshot
    round_number: int