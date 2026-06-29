"""
Prompt A/B micro (English): cooperation_assigned {0.2, 0.8} × 5 runs, single agent, single round.

risk_tolerance_assigned=0.2 fixed; each condition calls the real Haiku LLM directly
(English system/human prompts, no mock). Results are printed to the terminal only;
data/results.db is not touched.

Usage (repo root):
    python analysis/prompt_ab_micro_en.py

Estimated cost: ~10 calls × $0.002 ≈ $0.02 (safety cap $0.10).
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

COOPERATION_VALUES = (0.2, 0.8)
RUNS_PER_VALUE = 5
RISK_TOLERANCE = 0.2
COST_CAP_USD = 0.10
AGENT_ID = "agent_1"


def _build_system_prompt_en(agent_input: AgentInputView) -> str:
    trait = agent_input.own_trait
    return (
        "You are an agent extracting from a shared resource pool. "
        f"Your cooperation_assigned value is {trait.cooperation_assigned:.2f} "
        "(0=fully selfish, 1=fully cooperative), "
        f"your risk_tolerance_assigned value is {trait.risk_tolerance_assigned:.2f} "
        "(0=very cautious, 1=very risk-taking). "
        "Make your decision in line with these tendencies, but do not repeat "
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


def _make_state(cooperation: float, run_index: int) -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id="_scratch",
        run_id=f"micro_en_{cooperation:.1f}_{run_index}",
        max_rounds=1,
        agent_traits={
            AGENT_ID: TraitProfile(
                agent_id=AGENT_ID,
                cooperation_assigned=cooperation,
                risk_tolerance_assigned=RISK_TOLERANCE,
                profile_label=f"coop_{cooperation:.1f}",
            )
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


async def main() -> None:
    reset_token_usage()

    print("Prompt A/B micro EN (single agent, single round, real LLM)")
    print(f"  model                  : {settings.ANTHROPIC_MODEL}")
    print(f"  temperature            : {settings.TEMPERATURE}")
    print(f"  risk_tolerance_assigned: {RISK_TOLERANCE} (fixed)")
    print(f"  cooperation_assigned   : {list(COOPERATION_VALUES)} × {RUNS_PER_VALUE} runs")
    print(f"  cost safety cap        : ${COST_CAP_USD:.2f}")
    print(f"  DB                     : not written\n")

    extractions: dict[float, list[float]] = {coop: [] for coop in COOPERATION_VALUES}
    decisions_by_coop: dict[float, list[tuple[int, AgentDecision]]] = {
        coop: [] for coop in COOPERATION_VALUES
    }
    stopped_early = False

    for cooperation in COOPERATION_VALUES:
        for run_index in range(1, RUNS_PER_VALUE + 1):
            if token_usage.estimated_cost_usd() >= COST_CAP_USD:
                print(
                    f"\n⚠ Cost cap (${COST_CAP_USD:.2f}) reached — "
                    "remaining calls skipped."
                )
                stopped_early = True
                break

            state = _make_state(cooperation, run_index)
            decision = await _run_single_agent_en(state)

            extractions[cooperation].append(decision.extraction_amount)
            decisions_by_coop[cooperation].append((run_index, decision))
            print(
                f"  coop={cooperation:.1f} run={run_index}: "
                f"extraction_amount={decision.extraction_amount:.4f}, "
                f"declared_max={decision.declared_max:.4f}"
            )
            print(f"    justification: {decision.justification}")

        if stopped_early:
            break

    high_coop_runs = decisions_by_coop.get(0.8, [])
    if high_coop_runs:
        print("\n--- cooperation=0.8 full justifications ---")
        for run_index, decision in high_coop_runs:
            print(f"\n  run {run_index}:")
            print(f"    extraction_amount={decision.extraction_amount:.4f}")
            print(f"    declared_max={decision.declared_max:.4f}")
            print(f"    justification: {decision.justification}")

    print("\n--- Group means (extraction_amount) ---")
    means: dict[float, float] = {}
    for cooperation in COOPERATION_VALUES:
        values = extractions[cooperation]
        if not values:
            print(f"  cooperation={cooperation:.1f}: (no data)")
            continue
        mean_val = statistics.mean(values)
        means[cooperation] = mean_val
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        print(
            f"  cooperation={cooperation:.1f}: "
            f"n={len(values)}, mean={mean_val:.4f}, stdev={stdev:.4f}"
        )

    if 0.2 in means and 0.8 in means:
        diff = means[0.8] - means[0.2]
        print(f"\n  Mean difference (coop=0.8 − coop=0.2): {diff:+.4f}")
        if diff < 0:
            print("  → Higher-cooperation group extracted less.")
        elif diff > 0:
            print("  → Higher-cooperation group extracted more.")
        else:
            print("  → Both groups have equal means.")

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
