"""Iterated Prisoner's Dilemma scenario package — isolated from CPR/bargaining."""

from scenarios.iterated_pd.graph import app
from scenarios.iterated_pd.state import (
    AGENT_A_ID,
    AGENT_B_ID,
    P_PUNISHMENT,
    R_REWARD,
    S_SUCKER,
    T_TEMPTATION,
    IPDState,
)

__all__ = [
    "app",
    "IPDState",
    "AGENT_A_ID",
    "AGENT_B_ID",
    "T_TEMPTATION",
    "R_REWARD",
    "P_PUNISHMENT",
    "S_SUCKER",
]
