"""
Category-trait pilot (English): 4 category traits × {0.2, 0.8} × 5 runs each.

All other traits fixed at 0.5 (mid); every trait appears in the system prompt
and AgentInputView. Single agent, single round, real Haiku LLM. No DB writes.

Usage (repo root):
    python analysis/trait_category_pilot.py

Estimated cost: ~40 calls × $0.002 ≈ $0.08 (safety cap $0.20).
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from pathlib import Path

from anthropic import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.decision_agent import (
    _INPUT_COST_PER_M,
    _OUTPUT_COST_PER_M,
    _get_structured_llm,
    _record_usage,
    reset_token_usage,
    token_usage,
)
from core.config import settings
from core.state import AgentDecision, AgentInputView, EnvironmentSnapshot, SimulationState, TraitProfile

TRAIT_VALUES = (0.2, 0.8)
RUNS_PER_VALUE = 5
FIXED_TRAIT_VALUE = 0.5
COST_CAP_USD = 0.20
AGENT_ID = "agent_1"

TRAIT_DEFINITIONS: dict[str, str] = {
    "cooperation_assigned": "0=fully selfish, 1=fully cooperative",
    "risk_tolerance_assigned": "0=very cautious, 1=very risk-taking",
    "aggression_assigned": "0=very passive, 1=very aggressive",
    "impatience_assigned": "0=very patient, 1=very impatient",
    "tolerance_assigned": "0=very intolerant, 1=very tolerant",
    "fairness_assigned": "0=very unfair/self-favoring, 1=very fair/equitable",
    "creativity_assigned": "0=very conventional/predictable, 1=very creative/unconventional in extraction strategy",
    "greed_assigned": "0=not greedy at all, 1=very greedy",
    "hoarding_assigned": "0=never hoards resources, 1=always hoards/stockpiles resources",
    "trust_assigned": "0=does not trust other agents at all, 1=fully trusts other agents",
    "caution_assigned": "0=not cautious at all, 1=very cautious",
}

# Previously piloted (batch 1): aggression, impatience, tolerance, fairness, creativity
PILOT_TRAITS: tuple[str, ...] = (
    "greed_assigned",
    "hoarding_assigned",
    "trust_assigned",
    "caution_assigned",
)


def _build_system_prompt_en(agent_input: AgentInputView) -> str:
    trait = agent_input.own_trait
    clauses = [
        f"your {name} value is {getattr(trait, name):.2f} ({definition})"
        for name, definition in TRAIT_DEFINITIONS.items()
    ]
    return (
        "You are an agent extracting from a shared resource pool. "
        + ", ".join(clauses)
        + ". Make your decision in line with these tendencies, but do not repeat "
        "or explain these numbers in your output."
    )


def _build_human_prompt_en(agent_input: AgentInputView) -> str:
    env = agent_input.environment
    return (
        f"Make your extraction decision for round {agent_input.round_number}.\n"
        f"- Pool: {env.pool_current:.2f} / {env.pool_capacity:.2f}\n"
        f"- Maximum extractable this round: {env.max_extractable_this_round:.2f}\n"
        f"- Regeneration rate: {env.regen_rate:.2f}\n"
        f"- Pool collapsed: {env.is_collapsed}\n\n"
        "Structured output: extraction_amount (between 0 and maximum), "
        "justification (brief rationale, up to 500 characters), "
        "declared_max (>= extraction_amount)."
    )


def _make_trait_profile(varying_trait: str, varying_value: float) -> TraitProfile:
    values = dict.fromkeys(TRAIT_DEFINITIONS, FIXED_TRAIT_VALUE)
    values[varying_trait] = varying_value
    return TraitProfile(
        agent_id=AGENT_ID,
        cooperation_assigned=values["cooperation_assigned"],
        risk_tolerance_assigned=values["risk_tolerance_assigned"],
        aggression_assigned=values["aggression_assigned"],
        impatience_assigned=values["impatience_assigned"],
        tolerance_assigned=values["tolerance_assigned"],
        fairness_assigned=values["fairness_assigned"],
        creativity_assigned=values["creativity_assigned"],
        greed_assigned=values["greed_assigned"],
        hoarding_assigned=values["hoarding_assigned"],
        trust_assigned=values["trust_assigned"],
        caution_assigned=values["caution_assigned"],
        profile_label=f"{varying_trait}_{varying_value:.1f}",
    )


def _make_state(varying_trait: str, varying_value: float, run_index: int) -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id="_scratch",
        run_id=f"cat_{varying_trait}_{varying_value:.1f}_{run_index}",
        max_rounds=1,
        agent_traits={
            AGENT_ID: _make_trait_profile(varying_trait, varying_value),
        },
        shock_schedule=[],
        environment=EnvironmentSnapshot(
            pool_current=pool,
            pool_capacity=pool,
            regen_rate=1.15,
            max_extractable_this_round=pool * settings.EXTRACTION_LIMIT_RATIO,
            round_number=0,
            is_collapsed=False,
        ),
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True,
)
async def _decide_with_anthropic_en(agent_input: AgentInputView) -> AgentDecision:
    messages = [
        ("system", _build_system_prompt_en(agent_input)),
        ("human", _build_human_prompt_en(agent_input)),
    ]
    result = await _get_structured_llm().ainvoke(messages)
    _record_usage(result["raw"])
    decision = AgentDecision.model_validate(result["parsed"])
    return decision.model_copy(
        update={
            "agent_id": agent_input.own_trait.agent_id,
            "round_number": agent_input.round_number,
        }
    )


async def _run_single_agent_en(state: SimulationState) -> AgentDecision:
    trait = state.agent_traits[AGENT_ID]
    agent_input = AgentInputView(
        own_trait=trait,
        environment=state.environment,
        round_number=state.round_number,
    )
    return await _decide_with_anthropic_en(agent_input)


def _print_trait_summary(
    varying_trait: str,
    extractions: dict[float, list[float]],
    decisions: dict[float, list[tuple[int, AgentDecision]]],
) -> None:
    print(f"\n{'=' * 72}")
    print(f"TRAIT: {varying_trait}")
    print(f"  definition: {TRAIT_DEFINITIONS[varying_trait]}")
    print(f"  other traits: {FIXED_TRAIT_VALUE} (fixed)")
    print(f"{'=' * 72}")

    means: dict[float, float] = {}
    for value in TRAIT_VALUES:
        samples = extractions.get(value, [])
        if not samples:
            print(f"  {varying_trait}={value:.1f}: (no data)")
            continue
        mean_val = statistics.mean(samples)
        means[value] = mean_val
        stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
        print(
            f"  {varying_trait}={value:.1f}: "
            f"n={len(samples)}, mean extraction={mean_val:.4f}, stdev={stdev:.4f}"
        )

    if 0.2 in means and 0.8 in means:
        diff = means[0.8] - means[0.2]
        print(f"\n  Mean difference (0.8 − 0.2): {diff:+.4f}  (n=5 per cell, descriptive)")
        if diff > 0:
            print(f"  → Higher {varying_trait} extracted more on average.")
        elif diff < 0:
            print(f"  → Higher {varying_trait} extracted less on average.")
        else:
            print("  → Both levels have equal means.")

    print("\n  Sample justifications:")
    shown = 0
    for value in TRAIT_VALUES:
        for run_index, decision in decisions.get(value, []):
            if shown >= 3:
                break
            print(
                f"    [{varying_trait}={value:.1f}, run={run_index}] "
                f"extraction={decision.extraction_amount:.4f}: "
                f"{decision.justification}"
            )
            shown += 1
        if shown >= 3:
            break


async def main() -> None:
    reset_token_usage()

    total_calls = len(PILOT_TRAITS) * len(TRAIT_VALUES) * RUNS_PER_VALUE
    print("Category-trait pilot (English, single agent, single round, real LLM)")
    print(f"  model                  : {settings.ANTHROPIC_MODEL}")
    print(f"  temperature            : {settings.TEMPERATURE}")
    print(f"  fixed traits           : all non-varying at {FIXED_TRAIT_VALUE}")
    print(f"  pilot traits           : {list(PILOT_TRAITS)}")
    print(f"  levels per trait       : {list(TRAIT_VALUES)} × {RUNS_PER_VALUE} runs")
    print(f"  total planned calls    : {total_calls}")
    print(f"  cost safety cap        : ${COST_CAP_USD:.2f}")
    print(f"  DB                     : not written\n")

    results: dict[str, dict[float, list[float]]] = {}
    decisions_by_trait: dict[str, dict[float, list[tuple[int, AgentDecision]]]] = {}
    stopped_early = False

    for varying_trait in PILOT_TRAITS:
        extractions: dict[float, list[float]] = {v: [] for v in TRAIT_VALUES}
        decisions: dict[float, list[tuple[int, AgentDecision]]] = {v: [] for v in TRAIT_VALUES}

        for value in TRAIT_VALUES:
            for run_index in range(1, RUNS_PER_VALUE + 1):
                if token_usage.estimated_cost_usd() >= COST_CAP_USD:
                    print(
                        f"\n⚠ Cost cap (${COST_CAP_USD:.2f}) reached — "
                        "remaining calls skipped."
                    )
                    stopped_early = True
                    break

                state = _make_state(varying_trait, value, run_index)
                decision = await _run_single_agent_en(state)

                extractions[value].append(decision.extraction_amount)
                decisions[value].append((run_index, decision))
                print(
                    f"  {varying_trait}={value:.1f} run={run_index}: "
                    f"extraction_amount={decision.extraction_amount:.4f}, "
                    f"declared_max={decision.declared_max:.4f}"
                )

            if stopped_early:
                break

        results[varying_trait] = extractions
        decisions_by_trait[varying_trait] = decisions
        _print_trait_summary(varying_trait, extractions, decisions)

        if stopped_early:
            break

    print(f"\n{'=' * 72}")
    print("SUMMARY (all pilot traits)")
    print(f"{'=' * 72}")
    for varying_trait in PILOT_TRAITS:
        extractions = results.get(varying_trait, {})
        low = extractions.get(0.2, [])
        high = extractions.get(0.8, [])
        if not low or not high:
            print(f"  {varying_trait}: incomplete data")
            continue
        diff = statistics.mean(high) - statistics.mean(low)
        print(f"  {varying_trait}: mean diff (0.8−0.2) = {diff:+.4f}")

    print("\n--- Token usage and estimated cost ---")
    print(f"  input_tokens  : {token_usage.input_tokens}")
    print(f"  output_tokens : {token_usage.output_tokens}")
    print(f"  total tokens  : {token_usage.input_tokens + token_usage.output_tokens}")
    print(f"  estimated cost: ${token_usage.estimated_cost_usd():.6f} USD")
    print(
        f"  (pricing: ${_INPUT_COST_PER_M:.2f}/M input, "
        f"${_OUTPUT_COST_PER_M:.2f}/M output — Claude Haiku 4.5)"
    )


if __name__ == "__main__":
    asyncio.run(main())
