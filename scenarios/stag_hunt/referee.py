"""LLM-free Referee for the iterated stag hunt.

Deterministic parse -> payoff-matrix lookup -> running score -> behavioral
metrics (stag rate, coordination rate, early/late stag rates). Never calls
an LLM.
"""

from __future__ import annotations

from typing import Any

from scenarios.stag_hunt.persistence import save_stag_hunt_round, save_stag_hunt_summary
from scenarios.stag_hunt.state import (
    AGENT_A_ID,
    AGENT_B_ID,
    PAYOFF_MATRIX,
    Choice,
    StagHuntRoundMetrics,
    StagHuntState,
)


def parse_choice(choice: Choice | str | dict[str, Any]) -> Choice:
    """Extract and validate a stag/hare choice."""
    if isinstance(choice, dict):
        raw = choice.get("choice", "hare")
    else:
        raw = choice
    raw = str(raw).strip().lower()
    return "stag" if raw == "stag" else "hare"


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
) -> StagHuntRoundMetrics:
    """Round-level metrics: payoffs + updated cumulative scores (pure math)."""
    a = parse_choice(choice_a)
    b = parse_choice(choice_b)
    payoff_a, payoff_b = compute_payoffs(a, b)
    return StagHuntRoundMetrics(
        round_number=round_number,
        agent_a_choice=a,
        agent_b_choice=b,
        agent_a_payoff=payoff_a,
        agent_b_payoff=payoff_b,
        agent_a_score=running_score_a + payoff_a,
        agent_b_score=running_score_b + payoff_b,
        mutual_stag=(a == "stag" and b == "stag"),
        mutual_hare=(a == "hare" and b == "hare"),
        miscoordinated=(a != b),
    )


def _agent_choice_key(agent: str) -> str:
    return "agent_a_choice" if agent == AGENT_A_ID else "agent_b_choice"


def compute_stag_rate(history: list[dict[str, Any]], agent: str) -> float:
    """Fraction of rounds in which `agent` chose stag."""
    if not history:
        return 0.0
    key = _agent_choice_key(agent)
    stag = sum(1 for h in history if h[key] == "stag")
    return stag / len(history)


def compute_coordination_rate(history: list[dict[str, Any]]) -> float:
    """Fraction of rounds where both agents chose the same option (stag/stag or hare/hare)."""
    if not history:
        return 0.0
    matched = sum(
        1 for h in history if h["agent_a_choice"] == h["agent_b_choice"]
    )
    return matched / len(history)


def compute_early_late_stag_rates(
    history: list[dict[str, Any]],
    agent: str,
    max_rounds: int,
) -> tuple[float, float]:
    """
    Split the run into first-half / second-half rounds (by round_number,
    0-indexed) and return (early_stag_rate, late_stag_rate) for `agent`.
    """
    key = _agent_choice_key(agent)
    ordered = sorted(history, key=lambda h: h["round_number"])
    half = max_rounds / 2.0

    early = [h for h in ordered if h["round_number"] < half]
    late = [h for h in ordered if h["round_number"] >= half]

    def _stag_rate(rounds: list[dict[str, Any]]) -> float:
        if not rounds:
            return 0.0
        stag = sum(1 for h in rounds if h[key] == "stag")
        return stag / len(rounds)

    return _stag_rate(early), _stag_rate(late)


def run_referee(state: StagHuntState) -> dict[str, Any]:
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

    save_stag_hunt_round(
        experiment_id=state.experiment_id,
        run_id=state.run_id,
        history_entry=history_entry,
        metrics=metrics,
    )

    new_round = state.round_number + 1
    is_terminated = new_round >= state.max_rounds
    termination_reason = "completed" if is_terminated else None

    if is_terminated:
        stag_rate_a = compute_stag_rate(new_history, AGENT_A_ID)
        stag_rate_b = compute_stag_rate(new_history, AGENT_B_ID)
        coordination_rate = compute_coordination_rate(new_history)
        early_a, late_a = compute_early_late_stag_rates(
            new_history, AGENT_A_ID, state.max_rounds
        )
        early_b, late_b = compute_early_late_stag_rates(
            new_history, AGENT_B_ID, state.max_rounds
        )
        save_stag_hunt_summary(
            experiment_id=state.experiment_id,
            run_id=state.run_id,
            final_score_a=metrics.agent_a_score,
            final_score_b=metrics.agent_b_score,
            stag_rate_a=stag_rate_a,
            stag_rate_b=stag_rate_b,
            coordination_rate=coordination_rate,
            early_stag_rate_a=early_a,
            late_stag_rate_a=late_a,
            early_stag_rate_b=early_b,
            late_stag_rate_b=late_b,
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
