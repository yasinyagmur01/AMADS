"""LLM-free Referee for the bargaining (ultimatum) game.

Deterministic parse → payoffs → fidelity proxies. Never calls an LLM.
"""

from __future__ import annotations

from typing import Any

from core.state import EnvironmentSnapshot
from scenarios.bargaining.persistence import save_bargaining_round
from scenarios.bargaining.state import (
    PIE_SIZE,
    BargainingMetricsSnapshot,
    BargainingState,
    ProposerOffer,
    ResponderDecision,
)


def parse_offer(offer: ProposerOffer | float | dict[str, Any], pie_size: float = PIE_SIZE) -> float:
    """Extract and clamp proposer's keep_amount X ∈ [0, pie_size]."""
    if isinstance(offer, ProposerOffer):
        raw = offer.keep_amount
    elif isinstance(offer, dict):
        raw = float(offer.get("keep_amount", offer.get("current_offer", 0.0)))
    else:
        raw = float(offer)
    return max(0.0, min(float(pie_size), raw))


def parse_decision(decision: ResponderDecision | bool | dict[str, Any]) -> bool:
    """Extract accept/reject boolean."""
    if isinstance(decision, ResponderDecision):
        return bool(decision.accept)
    if isinstance(decision, dict):
        return bool(decision.get("accept", decision.get("current_accept", False)))
    return bool(decision)


def compute_payoffs(
    keep_amount: float,
    accepted: bool,
    pie_size: float = PIE_SIZE,
) -> tuple[float, float]:
    """Classic ultimatum: accept → (X, pie−X); reject → (0, 0)."""
    keep = parse_offer(keep_amount, pie_size)
    offer_to_responder = pie_size - keep
    if accepted:
        return keep, offer_to_responder
    return 0.0, 0.0


def compute_fidelity_metrics(
    *,
    keep_amount: float,
    accepted: bool,
    proposer_coop: float,
    responder_coop: float,
    pie_size: float = PIE_SIZE,
    round_number: int = 0,
) -> BargainingMetricsSnapshot:
    """
    Round-level fidelity proxies (pure math).

    Proposer: high cooperation → low keep_fraction expected.
      proposer_coop_alignment = coop − keep_fraction  (positive = expected direction)

    Responder: high cooperation → lower acceptance threshold (accept low offers).
      accepted:  coop − offer_frac   (accepting a low offer with high coop → positive)
      rejected:  offer_frac − coop   (rejecting a low offer with high coop → negative)
    Primary cross-run metrics (Pearson r, mean_diff) are computed in analysis.
    """
    keep = parse_offer(keep_amount, pie_size)
    offer_to_responder = pie_size - keep
    keep_fraction = keep / pie_size if pie_size > 0 else 0.0
    offer_frac = offer_to_responder / pie_size if pie_size > 0 else 0.0
    proposer_payoff, responder_payoff = compute_payoffs(keep, accepted, pie_size)

    proposer_coop_alignment = float(proposer_coop) - keep_fraction
    if accepted:
        responder_coop_alignment = float(responder_coop) - offer_frac
    else:
        responder_coop_alignment = offer_frac - float(responder_coop)

    return BargainingMetricsSnapshot(
        round_number=round_number,
        keep_amount=keep,
        offer_to_responder=offer_to_responder,
        accepted=accepted,
        proposer_payoff=proposer_payoff,
        responder_payoff=responder_payoff,
        keep_fraction=keep_fraction,
        proposer_coop_alignment=max(-1.0, min(1.0, proposer_coop_alignment)),
        responder_coop_alignment=max(-1.0, min(1.0, responder_coop_alignment)),
    )


def _env_params_for_round(round_number: int) -> tuple[float, float]:
    """Deterministic mild variation of contextual pressure (prompt context only)."""
    time_pressure = max(0.0, min(1.0, 0.20 + 0.05 * (round_number % 5)))
    resource_scarcity = max(0.0, min(1.0, 0.25 + 0.05 * ((round_number // 3) % 5)))
    return time_pressure, resource_scarcity


def run_referee(state: BargainingState) -> dict[str, Any]:
    """Referee node: payoffs + fidelity + history + termination (no LLM)."""
    pie = state.pie_size
    if state.current_offer is None or state.current_accept is None:
        raise ValueError("Referee requires current_offer and current_accept")

    keep = parse_offer(state.current_offer, pie)
    accepted = parse_decision(state.current_accept)
    proposer_payoff, responder_payoff = compute_payoffs(keep, accepted, pie)

    metrics = compute_fidelity_metrics(
        keep_amount=keep,
        accepted=accepted,
        proposer_coop=state.proposer_trait.cooperation_assigned,
        responder_coop=state.responder_trait.cooperation_assigned,
        pie_size=pie,
        round_number=state.round_number,
    )

    history_entry: dict[str, Any] = {
        "round_number": state.round_number,
        "keep_amount": keep,
        "offer_to_responder": pie - keep,
        "accepted": accepted,
        "proposer_payoff": proposer_payoff,
        "responder_payoff": responder_payoff,
        "time_pressure": state.time_pressure,
        "resource_scarcity": state.resource_scarcity,
        "justification_proposer": state.current_justification_proposer,
        "justification_responder": state.current_justification_responder,
    }
    new_history = [*state.history, history_entry]
    new_metrics = [*state.metrics_history, metrics]

    save_bargaining_round(
        experiment_id=state.experiment_id,
        run_id=state.run_id,
        history_entry=history_entry,
        metrics=metrics,
        proposer_trait=state.proposer_trait,
        responder_trait=state.responder_trait,
    )

    new_round = state.round_number + 1
    is_terminated = new_round >= state.max_rounds
    termination_reason = "completed" if is_terminated else None

    time_pressure, resource_scarcity = _env_params_for_round(new_round)
    new_env = EnvironmentSnapshot(
        pool_current=pie,
        pool_capacity=pie,
        regen_rate=1.0,
        max_extractable_this_round=pie,
        round_number=new_round,
        is_collapsed=False,
    )

    return {
        "history": new_history,
        "metrics_history": new_metrics,
        "round_number": new_round,
        "current_offer": None,
        "current_accept": None,
        "current_justification_proposer": None,
        "current_justification_responder": None,
        "time_pressure": time_pressure,
        "resource_scarcity": resource_scarcity,
        "env_snapshot": new_env,
        "is_terminated": is_terminated,
        "termination_reason": termination_reason,
    }
