"""LLM-free Referee for the iterated prisoner's dilemma.

Deterministic parse -> payoff-matrix lookup -> running score -> behavioral
metrics (forgiveness, early/late cooperation). Never calls an LLM.
"""

from __future__ import annotations

from typing import Any

from scenarios.iterated_pd.persistence import save_ipd_round, save_ipd_summary
from scenarios.iterated_pd.state import (
    AGENT_A_ID,
    AGENT_B_ID,
    PAYOFF_MATRIX,
    Choice,
    IPDRoundMetrics,
    IPDState,
)


def parse_choice(choice: Choice | str | dict[str, Any]) -> Choice:
    """Extract and validate a C/D choice."""
    if isinstance(choice, dict):
        raw = choice.get("choice", "D")
    else:
        raw = choice
    raw = str(raw).strip().upper()
    return "C" if raw == "C" else "D"


def compute_payoffs(choice_a: Choice, choice_b: Choice) -> tuple[float, float]:
    """Pure payoff-matrix lookup for (agent_a, agent_b)."""
    a = parse_choice(choice_a)
    b = parse_choice(choice_b)
    return PAYOFF_MATRIX[(a, b)]


def compute_round_metrics(
    *,
    round_number: int,
    choice_a: Choice,
    choice_b: Choice,
    running_score_a: float,
    running_score_b: float,
) -> IPDRoundMetrics:
    """Round-level metrics: payoffs + updated cumulative scores (pure math)."""
    a = parse_choice(choice_a)
    b = parse_choice(choice_b)
    payoff_a, payoff_b = compute_payoffs(a, b)
    return IPDRoundMetrics(
        round_number=round_number,
        agent_a_choice=a,
        agent_b_choice=b,
        agent_a_payoff=payoff_a,
        agent_b_payoff=payoff_b,
        agent_a_score=running_score_a + payoff_a,
        agent_b_score=running_score_b + payoff_b,
        mutual_cooperation=(a == "C" and b == "C"),
        mutual_defection=(a == "D" and b == "D"),
    )


def _agent_choice_key(agent: str) -> str:
    return "agent_a_choice" if agent == AGENT_A_ID else "agent_b_choice"


def _opponent_choice_key(agent: str) -> str:
    return "agent_b_choice" if agent == AGENT_A_ID else "agent_a_choice"


def compute_forgiveness_rate(history: list[dict[str, Any]], agent: str) -> float:
    """
    Forgiveness rate for `agent`: among rounds where the OPPONENT defected
    against `agent` in round n, the fraction of those rounds where `agent`
    nonetheless cooperates in round n+1.

    Returns 0.0 if the opponent never defected (no denominator).
    """
    own_key = _agent_choice_key(agent)
    opp_key = _opponent_choice_key(agent)
    ordered = sorted(history, key=lambda h: h["round_number"])

    defected_against = 0
    forgave = 0
    for i in range(len(ordered) - 1):
        cur, nxt = ordered[i], ordered[i + 1]
        if cur[opp_key] == "D":
            defected_against += 1
            if nxt[own_key] == "C":
                forgave += 1
    if defected_against == 0:
        return 0.0
    return forgave / defected_against


def compute_early_late_coop_rates(
    history: list[dict[str, Any]],
    agent: str,
    max_rounds: int,
) -> tuple[float, float]:
    """
    Split the run into first-half / second-half rounds (by round_number,
    0-indexed) and return (early_coop_rate, late_coop_rate) for `agent`.
    """
    own_key = _agent_choice_key(agent)
    ordered = sorted(history, key=lambda h: h["round_number"])
    half = max_rounds / 2.0

    early = [h for h in ordered if h["round_number"] < half]
    late = [h for h in ordered if h["round_number"] >= half]

    def _coop_rate(rounds: list[dict[str, Any]]) -> float:
        if not rounds:
            return 0.0
        coop = sum(1 for h in rounds if h[own_key] == "C")
        return coop / len(rounds)

    return _coop_rate(early), _coop_rate(late)


def run_referee(state: IPDState) -> dict[str, Any]:
    """Referee node: payoff lookup, running score, history, termination (no LLM)."""
    if state.current_choice_a is None or state.current_choice_b is None:
        raise ValueError("Referee requires current_choice_a and current_choice_b")

    choice_a = parse_choice(state.current_choice_a)
    choice_b = parse_choice(state.current_choice_b)

    metrics = compute_round_metrics(
        round_number=state.round_number,
        choice_a=choice_a,
        choice_b=choice_b,
        running_score_a=state.agent_a_score,
        running_score_b=state.agent_b_score,
    )

    history_entry: dict[str, Any] = {
        "round_number": state.round_number,
        "agent_a_choice": choice_a,
        "agent_b_choice": choice_b,
        "agent_a_payoff": metrics.agent_a_payoff,
        "agent_b_payoff": metrics.agent_b_payoff,
        "justification_a": state.current_justification_a,
        "justification_b": state.current_justification_b,
    }
    new_history = [*state.history, history_entry]
    new_metrics = [*state.metrics_history, metrics]

    save_ipd_round(
        experiment_id=state.experiment_id,
        run_id=state.run_id,
        history_entry=history_entry,
        metrics=metrics,
    )

    new_round = state.round_number + 1
    is_terminated = new_round >= state.max_rounds
    termination_reason = "completed" if is_terminated else None

    if is_terminated:
        forgiveness_a = compute_forgiveness_rate(new_history, AGENT_A_ID)
        forgiveness_b = compute_forgiveness_rate(new_history, AGENT_B_ID)
        early_a, late_a = compute_early_late_coop_rates(
            new_history, AGENT_A_ID, state.max_rounds
        )
        early_b, late_b = compute_early_late_coop_rates(
            new_history, AGENT_B_ID, state.max_rounds
        )
        save_ipd_summary(
            experiment_id=state.experiment_id,
            run_id=state.run_id,
            final_score_a=metrics.agent_a_score,
            final_score_b=metrics.agent_b_score,
            forgiveness_rate_a=forgiveness_a,
            forgiveness_rate_b=forgiveness_b,
            early_coop_rate_a=early_a,
            late_coop_rate_a=late_a,
            early_coop_rate_b=early_b,
            late_coop_rate_b=late_b,
        )

    return {
        "history": new_history,
        "metrics_history": new_metrics,
        "round_number": new_round,
        "agent_a_score": metrics.agent_a_score,
        "agent_b_score": metrics.agent_b_score,
        "current_choice_a": None,
        "current_choice_b": None,
        "current_justification_a": None,
        "current_justification_b": None,
        "is_terminated": is_terminated,
        "termination_reason": termination_reason,
    }
