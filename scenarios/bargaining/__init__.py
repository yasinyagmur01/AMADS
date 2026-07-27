"""Bargaining (ultimatum) game scenario package — isolated from CPR."""

from scenarios.bargaining.graph import app
from scenarios.bargaining.state import BargainingState, PIE_SIZE

__all__ = ["app", "BargainingState", "PIE_SIZE"]
