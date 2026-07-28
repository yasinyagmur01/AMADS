"""
full_experiment_groq_v1 — CPR factorial on Groq (English symbolic prompts).

Same design as full_experiment_v1 (EXTRACTION_LIMIT_RATIO=0.12, Referee unchanged).
Uses Groq llama-3.3-70b-versatile via core/llm_providers registry.

Usage (from repo root):
    python experiments/cpr/run_groq_experiment.py --plan
    python experiments/cpr/run_groq_experiment.py --micro-pilot   # 10 runs
    python experiments/cpr/run_groq_experiment.py                 # 45 runs (only if GATE 4 says so)

Requirements: GROQ_API_KEY (.env). Cost treated as free (Groq free tier).
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.decision_agent import (
    _INPUT_COST_PER_M,
    _OUTPUT_COST_PER_M,
    reset_token_usage,
    token_usage,
)
from core.config import settings
from core.database import RESULTS_DB_PATH, register_experiment_conditions
from core.graph import app
from core.state import EnvironmentSnapshot, SimulationState, TraitProfile
from environment.shocks import build_mock_dev_shock_schedule

EXPERIMENT_ID = "full_experiment_groq_v1"
MAX_ROUNDS = 15
REPLICATIONS = 5
COST_CAP_USD = 999.0  # Groq free tier; soft guard only
MICRO_COST_CAP_USD = 999.0

TRAIT_LEVELS: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
}

# Micro-pilot: cooperation ∈ {0.2, 0.8}, risk fixed at 0.2 → 2 × 5 = 10 run
MICRO_COOP_LEVELS = ("low", "high")
MICRO_RISK_LEVEL = "low"

AGENT_IDS = [f"agent_{i}" for i in range(1, settings.AGENT_COUNT + 1)]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    coop_level: str
    risk_level: str
    coop_value: float
    risk_value: float
    replication: int


@dataclass
class RunSummary:
    run_id: str
    coop_level: str
    risk_level: str
    rounds_played: int
    termination_reason: str | None
    cost_usd: float


def build_run_plan(*, micro_pilot: bool = False) -> list[RunSpec]:
    if micro_pilot:
        coop_levels = MICRO_COOP_LEVELS
        risk_levels = (MICRO_RISK_LEVEL,)
    else:
        coop_levels = tuple(TRAIT_LEVELS)
        risk_levels = tuple(TRAIT_LEVELS)

    specs: list[RunSpec] = []
    for coop_level, risk_level in product(coop_levels, risk_levels):
        coop_value = TRAIT_LEVELS[coop_level]
        risk_value = TRAIT_LEVELS[risk_level]
        for rep in range(1, REPLICATIONS + 1):
            specs.append(
                RunSpec(
                    run_id=f"cond_{coop_level}_{risk_level}_rep{rep}",
                    coop_level=coop_level,
                    risk_level=risk_level,
                    coop_value=coop_value,
                    risk_value=risk_value,
                    replication=rep,
                )
            )
    return specs


def _profile_label(coop_level: str, risk_level: str) -> str:
    return f"{coop_level.capitalize()}Coop_{risk_level.capitalize()}Risk"


def _make_traits(spec: RunSpec) -> dict[str, TraitProfile]:
    label = _profile_label(spec.coop_level, spec.risk_level)
    return {
        agent_id: TraitProfile(
            agent_id=agent_id,
            cooperation_assigned=spec.coop_value,
            risk_tolerance_assigned=spec.risk_value,
            profile_label=label,
        )
        for agent_id in AGENT_IDS
    }


def _make_initial_state(spec: RunSpec) -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id=EXPERIMENT_ID,
        run_id=spec.run_id,
        max_rounds=MAX_ROUNDS,
        agent_traits=_make_traits(spec),
        shock_schedule=build_mock_dev_shock_schedule(),
        environment=EnvironmentSnapshot(
            pool_current=pool,
            pool_capacity=pool,
            regen_rate=1.15,
            max_extractable_this_round=pool * settings.EXTRACTION_LIMIT_RATIO,
            round_number=0,
            is_collapsed=False,
        ),
    )


def _condition_key(spec: RunSpec) -> str:
    return f"cond_{spec.coop_level}_{spec.risk_level}"


def _register_plan(specs: list[RunSpec]) -> None:
    rows = [
        (
            s.run_id,
            s.coop_level,
            s.risk_level,
            s.coop_value,
            s.risk_value,
            s.replication,
        )
        for s in specs
    ]
    register_experiment_conditions(EXPERIMENT_ID, rows, RESULTS_DB_PATH)


async def _run_single(spec: RunSpec) -> RunSummary:
    reset_token_usage()
    cost_before = token_usage.estimated_cost_usd()

    print(
        f"\n  → {spec.run_id}  "
        f"(coop={spec.coop_value}, risk={spec.risk_value}, rep={spec.replication})"
    )

    final_state = await app.ainvoke(_make_initial_state(spec))
    cost_usd = token_usage.estimated_cost_usd() - cost_before

    return RunSummary(
        run_id=spec.run_id,
        coop_level=spec.coop_level,
        risk_level=spec.risk_level,
        rounds_played=len(final_state["metrics_history"]),
        termination_reason=final_state["termination_reason"],
        cost_usd=cost_usd,
    )


def _print_plan(specs: list[RunSpec], *, micro_pilot: bool) -> None:
    mode = "MICRO-PILOT" if micro_pilot else "FULL"
    print("=" * 72)
    print(f"FULL EXPERIMENT GROQ v1 — EXPERIMENT PLAN ({mode}, dry run / not executed)")
    print("=" * 72)
    print(f"  experiment_id          : {EXPERIMENT_ID}")
    print(f"  database               : {RESULTS_DB_PATH}")
    print(f"  prompt variant         : English symbolic (locked Turkish untouched)")
    print(f"  total runs             : {len(specs)}")
    print(f"  replications/condition : {REPLICATIONS}")
    print(f"  max_rounds             : {MAX_ROUNDS}")
    print(f"  model                  : {settings.GROQ_MODEL} (provider=groq)")
    print(f"  temperature            : {settings.TEMPERATURE}")
    print(f"  EXTRACTION_LIMIT_RATIO : {settings.EXTRACTION_LIMIT_RATIO}")
    print(f"  agent count            : {settings.AGENT_COUNT}")
    if micro_pilot:
        print(
            f"  filter                 : coop ∈ {{0.2, 0.8}}, risk=0.2 "
            f"({len(specs)} runs)"
        )
        print(f"  cost safety cap        : ${MICRO_COST_CAP_USD:.2f}")
    else:
        print(f"  conditions             : {len(TRAIT_LEVELS) ** 2} (3×3)")
        print(f"  cost safety cap        : ${COST_CAP_USD:.2f}")

    print(f"\n  {'#':>3}  {'run_id':<32} {'coop':>5} {'risk':>5}  rep")
    print("  " + "-" * 58)
    for i, spec in enumerate(specs, 1):
        print(
            f"  {i:3d}  {spec.run_id:<32} "
            f"{spec.coop_value:5.1f} {spec.risk_value:5.1f}  {spec.replication}"
        )


def _print_condition_checkpoint(
    *,
    condition_label: str,
    condition_specs: list[RunSpec],
    condition_summaries: list[RunSummary],
    runs_completed: int,
    total_runs: int,
    total_cost: float,
    cost_cap: float,
) -> None:
    cond_cost = sum(s.cost_usd for s in condition_summaries)
    spec = condition_specs[0]
    print(f"\n{'=' * 72}")
    print(f"CONDITION COMPLETE — {condition_label}")
    print(f"{'=' * 72}")
    print(
        f"  trait                  : coop={spec.coop_value} ({spec.coop_level}), "
        f"risk={spec.risk_value} ({spec.risk_level})"
    )
    print(f"  cost this condition    : ${cond_cost:.4f}")
    print(f"  progress               : {runs_completed}/{total_runs} runs")
    print(f"  cumulative cost        : ${total_cost:.4f} / ${cost_cap:.2f}")
    print("  replication summary:")
    for s in condition_summaries:
        print(
            f"    {s.run_id}: {s.rounds_played} round, "
            f"termination={s.termination_reason or '—'}, "
            f"${s.cost_usd:.4f}"
        )


def _print_final_summary(
    summaries: list[RunSummary],
    stopped_early: bool,
    cost_cap: float,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"EXPERIMENT SUMMARY — {EXPERIMENT_ID}")
    print(f"{'=' * 72}")
    total_cost = sum(s.cost_usd for s in summaries)
    print(f"  completed runs         : {len(summaries)}")
    print(f"  total cost             : ${total_cost:.4f}")
    if stopped_early:
        print(f"  ⚠ Stopped early (cost cap ${cost_cap:.2f})")

    conn = sqlite3.connect(RESULTS_DB_PATH)
    metrics = conn.execute(
        "SELECT COUNT(*) FROM metrics_snapshots WHERE experiment_id = ?",
        (EXPERIMENT_ID,),
    ).fetchone()[0]
    decisions = conn.execute(
        "SELECT COUNT(*) FROM agent_decisions WHERE experiment_id = ?",
        (EXPERIMENT_ID,),
    ).fetchone()[0]
    conditions = conn.execute(
        "SELECT COUNT(*) FROM experiment_conditions WHERE experiment_id = ?",
        (EXPERIMENT_ID,),
    ).fetchone()[0]
    conn.close()
    print(f"  DB — metrics_snapshots  : {metrics}")
    print(f"  DB — agent_decisions    : {decisions}")
    print(f"  DB — experiment_conditions: {conditions}")
    print(
        f"\n  (pricing: ${_INPUT_COST_PER_M:.2f}/M input, "
        f"${_OUTPUT_COST_PER_M:.2f}/M output — Claude Haiku 4.5)"
    )


def _completed_run_ids() -> set[str]:
    conn = sqlite3.connect(RESULTS_DB_PATH)
    rows = conn.execute(
        "SELECT DISTINCT run_id FROM metrics_snapshots WHERE experiment_id = ?",
        (EXPERIMENT_ID,),
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


async def run_experiment(
    *,
    dry_run: bool = False,
    micro_pilot: bool = False,
    resume: bool = False,
) -> None:
    specs = build_run_plan(micro_pilot=micro_pilot)
    cost_cap = MICRO_COST_CAP_USD if micro_pilot else COST_CAP_USD

    if dry_run:
        _print_plan(specs, micro_pilot=micro_pilot)
        print("\n--plan mode: no runs executed.")
        return

    done = _completed_run_ids() if resume else set()
    if resume and done:
        print(f"  resume: skipping {len(done)} completed run_id(s)")

    mode = "MICRO-PILOT" if micro_pilot else "FULL"
    print("=" * 72)
    print(f"FULL EXPERIMENT GROQ v1 — {mode} STARTING")
    print("=" * 72)
    print(f"  experiment_id : {EXPERIMENT_ID}")
    print(f"  total runs    : {len(specs)}")
    print(f"  cost cap      : ${cost_cap:.2f}")

    _register_plan(specs)
    print(f"\n  Wrote/updated {len(specs)} rows in experiment_conditions.")

    summaries: list[RunSummary] = []
    total_cost = 0.0
    stopped_early = False
    condition_buffer: list[RunSummary] = []
    current_condition: str | None = None
    current_condition_specs: list[RunSpec] = []

    for i, spec in enumerate(specs):
        cond = _condition_key(spec)
        if current_condition is None:
            current_condition = cond
            current_condition_specs = []
        if cond != current_condition:
            if condition_buffer:
                _print_condition_checkpoint(
                    condition_label=current_condition,
                    condition_specs=current_condition_specs,
                    condition_summaries=condition_buffer,
                    runs_completed=len(summaries),
                    total_runs=len(specs),
                    total_cost=total_cost,
                    cost_cap=cost_cap,
                )
            condition_buffer = []
            current_condition = cond
            current_condition_specs = []

        if spec.run_id in done:
            print(f"  skip (resume) {spec.run_id}")
            continue

        if total_cost >= cost_cap:
            print(
                f"\n⚠ Cost cap (${cost_cap:.2f}) exceeded — "
                f"skipping run: {spec.run_id} and remaining."
            )
            stopped_early = True
            break

        current_condition_specs.append(spec)
        summary = await _run_single(spec)
        summaries.append(summary)
        condition_buffer.append(summary)
        total_cost += summary.cost_usd

        if total_cost > cost_cap:
            print(
                f"\n⚠ Total cost ${total_cost:.4f} — "
                f"exceeded ${cost_cap:.2f} cap. Remaining conditions stopped."
            )
            stopped_early = True
            _print_condition_checkpoint(
                condition_label=current_condition,
                condition_specs=current_condition_specs,
                condition_summaries=condition_buffer,
                runs_completed=len(summaries),
                total_runs=len(specs),
                total_cost=total_cost,
                cost_cap=cost_cap,
            )
            break

        if (i + 1) % REPLICATIONS == 0 and condition_buffer:
            _print_condition_checkpoint(
                condition_label=current_condition,
                condition_specs=current_condition_specs,
                condition_summaries=condition_buffer,
                runs_completed=len(summaries) + len(done),
                total_runs=len(specs),
                total_cost=total_cost,
                cost_cap=cost_cap,
            )
            condition_buffer = []
            current_condition_specs = []
            current_condition = None

    if condition_buffer and current_condition:
        _print_condition_checkpoint(
            condition_label=current_condition,
            condition_specs=current_condition_specs,
            condition_summaries=condition_buffer,
            runs_completed=len(summaries) + len(done),
            total_runs=len(specs),
            total_cost=total_cost,
            cost_cap=cost_cap,
        )

    _print_final_summary(summaries, stopped_early, cost_cap)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "full_experiment_groq_v1: English CPR on Groq, "
            "same design as full_experiment_v1."
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the run plan and exit (no API calls)",
    )
    parser.add_argument(
        "--micro-pilot",
        action="store_true",
        help="Only coop∈{0.2,0.8}, risk=0.2, 5 reps/cell = 10 runs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip run_ids that already have metrics_snapshots",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(
        run_experiment(
            dry_run=args.plan,
            micro_pilot=args.micro_pilot,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
