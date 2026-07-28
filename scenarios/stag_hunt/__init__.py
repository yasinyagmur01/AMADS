"""Stag Hunt scenario package — isolated from CPR/bargaining/iterated_pd."""

from scenarios.stag_hunt.graph import app
from scenarios.stag_hunt.state import (
    AGENT_A_ID,
    AGENT_B_ID,
    HARE_HARE,
    HARE_WHEN_PARTNER_STAGS,
    STAG_STAG,
    STAG_WHEN_PARTNER_HARES,
    StagHuntState,
)

__all__ = [
    "app",
    "StagHuntState",
    "AGENT_A_ID",
    "AGENT_B_ID",
    "STAG_STAG",
    "HARE_HARE",
    "HARE_WHEN_PARTNER_STAGS",
    "STAG_WHEN_PARTNER_HARES",
]
