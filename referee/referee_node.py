from core.config import settings
from core.database import RESULTS_DB_PATH, save_round_to_db
from core.state import (
    AgentDecision,
    EnvironmentSnapshot,
    MetricsSnapshot,
    ShockType,
    SimulationState,
)

def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _gini_coefficient(extractions: list[float]) -> float:
    """Bölüm 6: standart Gini, extraction_amount dizisi üzerinden."""
    if not extractions:
        return 0.0
    sorted_vals = sorted(extractions)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0.0:
        return 0.0
    weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return _clamp((2.0 * weighted_sum) / (n * total) - (n + 1) / n, 0.0, 1.0)


def _cooperation_score_avg(
    extractions: list[float], max_extractable_this_round: float
) -> float:
    """Bölüm 5/6: ortalama(1 - extraction_amount / max_extractable_this_round)."""
    if not extractions:
        return 0.0
    if max_extractable_this_round == 0.0:
        # Çekilebilir kaynak yok; extraction zorunlu olarak 0 → tam işbirliği.
        return 1.0
    scores = [
        _clamp(1.0 - (e / max_extractable_this_round), 0.0, 1.0)
        for e in extractions
    ]
    return sum(scores) / len(scores)


def _collapse_epsilon(pool_capacity: float) -> float:
    return pool_capacity * settings.COLLAPSE_EPSILON_RATIO


def _is_collapsed(pool_after: float, pool_capacity: float) -> bool:
    return pool_after <= _collapse_epsilon(pool_capacity)


def _apply_pool_formula(
    pool_current: float,
    pool_capacity: float,
    regen_rate: float,
    total_extraction: float,
) -> float:
    """Bölüm 2.2: pool[t+1] = clamp(pool[t] - Σ(extraction), 0, capacity) * regen_rate."""
    depleted = pool_current - total_extraction
    return _clamp(depleted, 0.0, pool_capacity) * regen_rate


def _compute_metrics_snapshot(
    round_number: int,
    round_decisions: list[AgentDecision],
    max_extractable_this_round: float,
    pool_after: float,
) -> MetricsSnapshot:
    """
    Bölüm 4.5 / 6: round-level frozen MetricsSnapshot.
    Yalnızca extraction_amount ve declared_max kullanılır; justification okunmaz.
    Sustainability Index, Shock Resilience, Trait-Behavior Fidelity burada hesaplanmaz.
    """
    extractions = [d.extraction_amount for d in round_decisions]
    return MetricsSnapshot(
        round_number=round_number,
        gini_coefficient=_gini_coefficient(extractions),
        cooperation_score_avg=_cooperation_score_avg(
            extractions, max_extractable_this_round
        ),
        total_extraction=sum(extractions),
        pool_after=pool_after,
        constraint_violations=sum(
            1 for d in round_decisions if d.extraction_amount > d.declared_max
        ),
    )


def _apply_shocks(
    pool_capacity: float,
    regen_rate: float,
    pool_current: float,
    shock_schedule,
    round_number: int,
) -> tuple[float, float, float, float]:
    max_extractable_multiplier = 1.0

    for shock in shock_schedule:
        if shock.round_number != round_number:
            continue
        if shock.shock_type == ShockType.CAPACITY_DROP:
            scale = 1.0 + shock.magnitude
            pool_capacity = max(1e-9, pool_capacity * scale)
            pool_current = max(1e-9, min(pool_current * scale, pool_capacity))
        elif shock.shock_type == ShockType.REGEN_BOOST:
            regen_rate = max(1e-9, regen_rate * (1.0 + shock.magnitude))
        elif shock.shock_type == ShockType.DEMAND_SURGE:
            max_extractable_multiplier *= 1.0 + shock.magnitude

    return pool_capacity, regen_rate, pool_current, max_extractable_multiplier


def run_referee(state: SimulationState) -> dict:
    current_round = state.round_number
    env = state.environment

    round_decisions = [
        d for d in state.round_decisions if d.round_number == current_round
    ]

    total_extraction = sum(d.extraction_amount for d in round_decisions)

    pool_after = _apply_pool_formula(
        env.pool_current,
        env.pool_capacity,
        env.regen_rate,
        total_extraction,
    )

    pool_capacity, regen_rate, pool_after, max_extractable_multiplier = _apply_shocks(
        env.pool_capacity,
        env.regen_rate,
        pool_after,
        state.shock_schedule,
        current_round,
    )
    pool_after = _clamp(pool_after, 0.0, pool_capacity)

    metrics = _compute_metrics_snapshot(
        current_round,
        round_decisions,
        env.max_extractable_this_round,
        pool_after,
    )

    save_round_to_db(
        state.model_copy(
            update={"metrics_history": [*state.metrics_history, metrics]}
        ),
        RESULTS_DB_PATH,
    )

    next_max_extractable = min(
        pool_after * settings.EXTRACTION_LIMIT_RATIO * max_extractable_multiplier,
        pool_after,
    )

    new_round_number = current_round + 1
    is_terminated = False
    termination_reason = None

    if _is_collapsed(pool_after, pool_capacity):
        is_terminated = True
        termination_reason = "collapse"
    elif new_round_number >= state.max_rounds:
        is_terminated = True
        termination_reason = "completed"

    new_environment = EnvironmentSnapshot(
        pool_current=pool_after,
        pool_capacity=pool_capacity,
        regen_rate=regen_rate,
        max_extractable_this_round=next_max_extractable,
        round_number=new_round_number,
        is_collapsed=_is_collapsed(pool_after, pool_capacity),
    )

    return {
        "environment": new_environment,
        "metrics_history": [*state.metrics_history, metrics],
        "round_number": new_round_number,
        "is_terminated": is_terminated,
        "termination_reason": termination_reason,
    }
