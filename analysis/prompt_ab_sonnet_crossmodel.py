"""
Cross-model micro replication: Sonnet 4.6, 2×2 trait grid, single agent, round 0.

cooperation_assigned × risk_tolerance_assigned ∈ {0.2, 0.8}², n=10 per cell.
Locked Haiku primary study: full_experiment_v1. This run tests directional
replication on a second Claude model only.

Usage (repo root):
    python analysis/prompt_ab_sonnet_crossmodel.py
    python analysis/prompt_ab_sonnet_crossmodel.py --dry-run

Output: data/sonnet_crossmodel_v1.csv (not results.db)
Cost cap: $2.00 (expected ~$0.30–0.50 for 40 calls)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from anthropic import RateLimitError
from langchain_anthropic import ChatAnthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.decision_agent import (
    _build_human_prompt,
    _build_system_prompt,
    _require_anthropic_key,
)
from core.config import settings
from core.state import AgentDecision, AgentInputView, EnvironmentSnapshot, SimulationState, TraitProfile

EXPERIMENT_ID = "sonnet_crossmodel_v1"
SONNET_MODEL = "claude-sonnet-4-6"
TRAIT_LEVELS = (0.2, 0.8)
RUNS_PER_CELL = 10
COST_CAP_USD = 2.00
AGENT_ID = "agent_1"
OUTPUT_CSV = _ROOT / "data" / "sonnet_crossmodel_v1.csv"

# Sonnet 4.6 pricing (Bölüm 10.2)
_INPUT_COST_PER_M = 3.00
_OUTPUT_COST_PER_M = 15.00

# Haiku round-0 reference (full_experiment_v1, locked)
HAIKU_COOP_R = 0.4563
HAIKU_RISK_R = 0.6783


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, inp: int, out: int) -> None:
        self.input_tokens += inp
        self.output_tokens += out

    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens * _INPUT_COST_PER_M / 1_000_000
            + self.output_tokens * _OUTPUT_COST_PER_M / 1_000_000
        )


token_usage = TokenUsage()


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return num / (var_x * var_y) ** 0.5


def _record_usage(raw_message) -> None:
    usage = getattr(raw_message, "usage_metadata", None) or {}
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    if inp is None or out is None:
        meta = getattr(raw_message, "response_metadata", None) or {}
        api_usage = meta.get("usage") or {}
        inp = inp if inp is not None else api_usage.get("input_tokens", 0)
        out = out if out is not None else api_usage.get("output_tokens", 0)
    token_usage.add(int(inp or 0), int(out or 0))


def _make_state(coop: float, risk: float, run_index: int) -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id=EXPERIMENT_ID,
        run_id=f"sonnet_{coop:.1f}_{risk:.1f}_rep{run_index}",
        max_rounds=1,
        agent_traits={
            AGENT_ID: TraitProfile(
                agent_id=AGENT_ID,
                cooperation_assigned=coop,
                risk_tolerance_assigned=risk,
                profile_label=f"coop_{coop:.1f}_risk_{risk:.1f}",
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
async def _decide_sonnet(
    llm_structured,
    agent_input: AgentInputView,
) -> AgentDecision:
    messages = [
        ("system", _build_system_prompt(agent_input)),
        ("human", _build_human_prompt(agent_input)),
    ]
    result = await llm_structured.ainvoke(messages)
    _record_usage(result["raw"])
    decision = AgentDecision.model_validate(result["parsed"])
    return decision.model_copy(
        update={
            "agent_id": agent_input.own_trait.agent_id,
            "round_number": agent_input.round_number,
        }
    )


async def _run_single(
    llm_structured,
    coop: float,
    risk: float,
    run_index: int,
) -> tuple[AgentDecision, float]:
    state = _make_state(coop, risk, run_index)
    trait = state.agent_traits[AGENT_ID]
    agent_input = AgentInputView(
        own_trait=trait,
        environment=state.environment,
        round_number=state.round_number,
    )
    decision = await _decide_sonnet(llm_structured, agent_input)
    declared = decision.declared_max
    fraction = decision.extraction_amount / declared if declared > 0 else 0.0
    return decision, fraction


def _write_csv(rows: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "model",
        "run_id",
        "coop_value",
        "risk_value",
        "replication",
        "extraction_amount",
        "declared_max",
        "extraction_fraction",
        "justification",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def main(dry_run: bool = False) -> None:
    global token_usage
    token_usage = TokenUsage()

    cells = list(product(TRAIT_LEVELS, TRAIT_LEVELS))
    total_calls = len(cells) * RUNS_PER_CELL

    print("Sonnet cross-model micro replication")
    print(f"  experiment_id : {EXPERIMENT_ID}")
    print(f"  model         : {SONNET_MODEL}")
    print(f"  design        : 2×2 trait grid, n={RUNS_PER_CELL}/cell")
    print(f"  total calls   : {total_calls}")
    print(f"  cost cap      : ${COST_CAP_USD:.2f}")
    print(f"  output        : {OUTPUT_CSV}")
    print(f"  dry_run       : {dry_run}\n")

    if dry_run:
        est = total_calls * 0.008
        print(f"  estimated cost (rough): ~${est:.2f}")
        return

    _require_anthropic_key()
    llm = ChatAnthropic(
        model=SONNET_MODEL,
        temperature=settings.TEMPERATURE,
        api_key=settings.ANTHROPIC_API_KEY,
    )
    llm_structured = llm.with_structured_output(AgentDecision, include_raw=True)

    csv_rows: list[dict] = []
    coops: list[float] = []
    risks: list[float] = []
    fractions: list[float] = []
    stopped = False

    for coop, risk in cells:
        for run_index in range(1, RUNS_PER_CELL + 1):
            if token_usage.estimated_cost_usd() >= COST_CAP_USD:
                print(f"\n⚠ Cost cap ${COST_CAP_USD:.2f} reached — stopping.")
                stopped = True
                break

            decision, fraction = await _run_single(
                llm_structured, coop, risk, run_index
            )
            coops.append(coop)
            risks.append(risk)
            fractions.append(fraction)
            csv_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "model": SONNET_MODEL,
                    "run_id": f"sonnet_{coop:.1f}_{risk:.1f}_rep{run_index}",
                    "coop_value": coop,
                    "risk_value": risk,
                    "replication": run_index,
                    "extraction_amount": f"{decision.extraction_amount:.4f}",
                    "declared_max": f"{decision.declared_max:.4f}",
                    "extraction_fraction": f"{fraction:.6f}",
                    "justification": decision.justification,
                }
            )
            print(
                f"  coop={coop:.1f} risk={risk:.1f} rep={run_index}: "
                f"extract={decision.extraction_amount:.2f} frac={fraction:.3f} "
                f"cost=${token_usage.estimated_cost_usd():.4f}"
            )

        if stopped:
            break

    if not csv_rows:
        print("No results collected.")
        return

    _write_csv(csv_rows)

    coop_r = _pearson_r(coops, fractions)
    risk_r = _pearson_r(risks, fractions)

    print("\n--- Sonnet directional replication ---")
    print(f"  n observations : {len(fractions)}")
    print(f"  coop → fraction: r = {coop_r:+.4f}  (Haiku ref: +{HAIKU_COOP_R:.4f})")
    print(f"  risk → fraction: r = {risk_r:+.4f}  (Haiku ref: +{HAIKU_RISK_R:.4f})")

    coop_ok = coop_r > 0
    risk_ok = risk_r > 0
    if coop_ok and risk_ok:
        print("  → Both traits same sign as Haiku (Claude cross-model direction match).")
    else:
        parts = []
        if not coop_ok:
            parts.append("cooperation sign mismatch")
        if not risk_ok:
            parts.append("risk sign mismatch")
        print(f"  → Partial mismatch: {', '.join(parts)}")

    print("\n--- Cost ---")
    print(f"  input_tokens  : {token_usage.input_tokens}")
    print(f"  output_tokens : {token_usage.output_tokens}")
    print(f"  estimated cost: ${token_usage.estimated_cost_usd():.4f} USD")
    print(f"  CSV written   : {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only, no API calls",
    )
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
