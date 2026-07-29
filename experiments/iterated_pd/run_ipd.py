"""
iterated_pd_groq_v1 — 2-player iterated Prisoner's Dilemma on Groq.

English symbolic prompts (new experiment_id — see .cursorrules). Referee is
LLM-free (payoff matrix lookup + running score + forgiveness / early-late
cooperation metrics). Runs are sequential for now (no cross-run concurrency);
within a round, both agents decide in parallel via asyncio.gather.

Usage (from repo root):
    python experiments/iterated_pd/run_ipd.py --plan
    python experiments/iterated_pd/run_ipd.py --plan --micro-pilot
    python experiments/iterated_pd/run_ipd.py --micro-pilot        # 10 runs (API)
    python experiments/iterated_pd/run_ipd.py                      # full grid (API)

Requirements: GROQ_API_KEY (.env). Cost treated as free (Groq free tier);
token accounting kept for informational parity with Anthropic-priced scenarios.
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

from core.llm_providers import resolve_llm_config
from core.state import TraitProfile
from scenarios.iterated_pd.agents import (
    _INPUT_COST_PER_M,
    _OUTPUT_COST_PER_M,
    reset_token_usage,
    token_usage,
)
from scenarios.iterated_pd.graph import app
from scenarios.iterated_pd.persistence import (
    IPD_DB_PATH,
    init_ipd_db,
    register_ipd_conditions,
)
from scenarios.iterated_pd.state import AGENT_A_ID, AGENT_B_ID, IPDState

EXPERIMENT_ID = "iterated_pd_groq_v1"
MAX_ROUNDS = 12
REPLICATIONS = 5
COST_CAP_USD = 999.0  # Groq free tier; soft guard only
MICRO_COST_CAP_USD = 999.0

TRAIT_LEVELS: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
}

# Coop micro-pilot: cooperation ∈ {0.2, 0.8}, risk fixed at 0.5 → 2 × 5 = 10 runs.
MICRO_COOP_LEVELS = ("low", "high")
MICRO_RISK_LEVEL = "medium"

# Risk micro-pilot: risk ∈ {0.2, 0.8}, cooperation fixed at 0.5 → 2 × 5 = 10 runs.
RISK_MICRO_RISK_LEVELS = ("low", "high")
RISK_MICRO_COOP_LEVEL = "medium"


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
    final_score_a: float
    final_score_b: float


def build_run_plan(
    *,
    micro_pilot: bool = False,
    risk_micro_pilot: bool = False,
) -> list[RunSpec]:
    if risk_micro_pilot:
        coop_levels = (RISK_MICRO_COOP_LEVEL,)
        risk_levels = RISK_MICRO_RISK_LEVELS
    elif micro_pilot:
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
                    run_id=f"coop_{coop_level}_risk_{risk_level}_rep{rep}",
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


def _make_initial_state(spec: RunSpec, experiment_id: str) -> IPDState:
    label = _profile_label(spec.coop_level, spec.risk_level)
    return IPDState(
        experiment_id=experiment_id,
        run_id=spec.run_id,
        max_rounds=MAX_ROUNDS,
        agent_a_trait=TraitProfile(
            agent_id=AGENT_A_ID,
            cooperation_assigned=spec.coop_value,
            risk_tolerance_assigned=spec.risk_value,
            profile_label=label,
        ),
        agent_b_trait=TraitProfile(
            agent_id=AGENT_B_ID,
            cooperation_assigned=spec.coop_value,
            risk_tolerance_assigned=spec.risk_value,
            profile_label=label,
        ),
    )


def _condition_key(spec: RunSpec) -> str:
    return f"coop_{spec.coop_level}_risk_{spec.risk_level}"


def _register_plan(specs: list[RunSpec], experiment_id: str) -> None:
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
    register_ipd_conditions(experiment_id, rows, IPD_DB_PATH)


def _print_plan(
    specs: list[RunSpec],
    *,
    micro_pilot: bool,
    risk_micro_pilot: bool,
    experiment_id: str,
) -> None:
    if risk_micro_pilot:
        mode = "RISK-MICRO-PILOT"
    elif micro_pilot:
        mode = "MICRO-PILOT"
    else:
        mode = "FULL"

    print("=" * 72)
    print(f"ITERATED PD — EXPERIMENT PLAN ({mode}, dry run / not executed)")
    print("=" * 72)
    print(f"  experiment_id          : {experiment_id}")
    print(f"  game                   : iterated prisoner's dilemma (2-player, simultaneous)")
    print(f"  database               : {IPD_DB_PATH}")
    print(f"  prompt style           : English symbolic")
    print(f"  total runs             : {len(specs)}")
    print(f"  replications/condition : {REPLICATIONS}")
    print(f"  max_rounds             : {MAX_ROUNDS}")
    print(f"  payoffs (T,R,P,S)      : 5, 3, 1, 0")
    print(f"  agents/round           : 2 (parallel fan-out)")
    print(f"  LLM calls/run (upper)  : {2 * MAX_ROUNDS}")
    cfg = resolve_llm_config(experiment_id)
    print(f"  model                  : {cfg.model} (provider={cfg.provider})")
    print(f"  temperature            : {cfg.temperature}")
    if risk_micro_pilot:
        print(f"  filter                 : coop=0.5, risk ∈ {{0.2, 0.8}} (homogeneous)")
    elif micro_pilot:
        print(f"  filter                 : coop ∈ {{0.2, 0.8}}, risk=0.5 (homogeneous)")
    else:
        print(
            f"  grid                   : coop∈{{0.2,0.5,0.8}} × risk∈{{0.2,0.5,0.8}} "
            f"(homogeneous per run) → {len(TRAIT_LEVELS) ** 2} cells"
        )

    print(f"\n  {'#':>3}  {'run_id':<32} {'coop':>5} {'risk':>5}  rep")
    print("  " + "-" * 58)
    for i, spec in enumerate(specs, 1):
        print(
            f"  {i:3d}  {spec.run_id:<32} "
            f"{spec.coop_value:5.1f} {spec.risk_value:5.1f}  {spec.replication}"
        )

    print(
        f"\n  (pricing reference: ${_INPUT_COST_PER_M:.2f}/M input, "
        f"${_OUTPUT_COST_PER_M:.2f}/M output — informational only, Groq is free tier)"
    )


async def _run_single(spec: RunSpec, experiment_id: str) -> RunSummary:
    reset_token_usage()
    cost_before = token_usage.estimated_cost_usd()

    print(
        f"\n  → {spec.run_id}  "
        f"(coop={spec.coop_value}, risk={spec.risk_value}, rep={spec.replication})"
    )

    final_state = await app.ainvoke(_make_initial_state(spec, experiment_id))
    cost_usd = token_usage.estimated_cost_usd() - cost_before

    return RunSummary(
        run_id=spec.run_id,
        coop_level=spec.coop_level,
        risk_level=spec.risk_level,
        rounds_played=len(final_state["metrics_history"]),
        termination_reason=final_state.get("termination_reason"),
        cost_usd=cost_usd,
        final_score_a=final_state["agent_a_score"],
        final_score_b=final_state["agent_b_score"],
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
    print(f"  traits                 : coop={spec.coop_value}, risk={spec.risk_value}")
    print(f"  cost this condition    : ${cond_cost:.4f}")
    print(f"  progress               : {runs_completed}/{total_runs} runs")
    print(f"  cumulative cost        : ${total_cost:.4f} / ${cost_cap:.2f}")
    for s in condition_summaries:
        print(
            f"    {s.run_id}: {s.rounds_played} rounds, "
            f"scores=({s.final_score_a:.0f}, {s.final_score_b:.0f}), "
            f"${s.cost_usd:.4f}"
        )


def _print_final_summary(
    summaries: list[RunSummary],
    stopped_early: bool,
    cost_cap: float,
    experiment_id: str,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"EXPERIMENT SUMMARY — {experiment_id}")
    print(f"{'=' * 72}")
    total_cost = sum(s.cost_usd for s in summaries)
    print(f"  completed runs         : {len(summaries)}")
    print(f"  total cost             : ${total_cost:.4f}")
    if stopped_early:
        print(f"  ⚠ Stopped early (cost cap ${cost_cap:.2f})")

    init_ipd_db(IPD_DB_PATH)
    with sqlite3.connect(IPD_DB_PATH) as conn:
        rounds = conn.execute(
            "SELECT COUNT(*) FROM ipd_rounds WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]
        metrics = conn.execute(
            "SELECT COUNT(*) FROM ipd_metrics WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]
        conditions = conn.execute(
            "SELECT COUNT(*) FROM ipd_conditions WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]
        summary_rows = conn.execute(
            "SELECT COUNT(*) FROM ipd_summary WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]
    print(f"  DB — ipd_rounds        : {rounds}")
    print(f"  DB — ipd_metrics       : {metrics}")
    print(f"  DB — ipd_conditions    : {conditions}")
    print(f"  DB — ipd_summary       : {summary_rows}")


def _completed_run_ids(experiment_id: str) -> set[str]:
    init_ipd_db(IPD_DB_PATH)
    with sqlite3.connect(IPD_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT run_id FROM ipd_rounds
            WHERE experiment_id = ?
            GROUP BY run_id
            HAVING COUNT(*) >= ?
            """,
            (experiment_id, MAX_ROUNDS),
        ).fetchall()
    return {r[0] for r in rows}


async def run_experiment(
    *,
    dry_run: bool = False,
    micro_pilot: bool = False,
    risk_micro_pilot: bool = False,
    resume: bool = False,
) -> None:
    if micro_pilot and risk_micro_pilot:
        raise SystemExit("Use either --micro-pilot or --risk-micro-pilot, not both.")

    experiment_id = EXPERIMENT_ID
    specs = build_run_plan(micro_pilot=micro_pilot, risk_micro_pilot=risk_micro_pilot)
    cost_cap = MICRO_COST_CAP_USD if (micro_pilot or risk_micro_pilot) else COST_CAP_USD

    if dry_run:
        _print_plan(
            specs,
            micro_pilot=micro_pilot,
            risk_micro_pilot=risk_micro_pilot,
            experiment_id=experiment_id,
        )
        print("\n--plan mode: no runs executed (no API calls).")
        return

    if risk_micro_pilot:
        mode = "RISK-MICRO-PILOT"
    elif micro_pilot:
        mode = "MICRO-PILOT"
    else:
        mode = "FULL"
    print("=" * 72)
    print(f"ITERATED PD — {mode} STARTING (sequential runs)")
    print("=" * 72)
    print(f"  experiment_id : {experiment_id}")
    print(f"  total runs    : {len(specs)}")
    print(f"  cost cap      : ${cost_cap:.2f}")

    _register_plan(specs, experiment_id)
    print(f"\n  Wrote/updated {len(specs)} rows in ipd_conditions.")

    done = _completed_run_ids(experiment_id) if resume else set()
    if done:
        print(f"  resume: skipping {len(done)} completed run_id(s)")

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
            if condition_buffer and current_condition_specs:
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

        if total_cost >= cost_cap:
            print(
                f"\n⚠ Cost cap (${cost_cap:.2f}) exceeded — "
                f"skipping run: {spec.run_id} and remaining."
            )
            stopped_early = True
            break

        if spec.run_id in done:
            print(f"  skip (resume) {spec.run_id}")
            continue

        current_condition_specs.append(spec)
        summary = await _run_single(spec, experiment_id)
        summaries.append(summary)
        condition_buffer.append(summary)
        total_cost += summary.cost_usd

        if (i + 1) % REPLICATIONS == 0:
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
            current_condition_specs = []
            current_condition = None

    if condition_buffer and current_condition_specs:
        _print_condition_checkpoint(
            condition_label=current_condition or "",
            condition_specs=current_condition_specs,
            condition_summaries=condition_buffer,
            runs_completed=len(summaries),
            total_runs=len(specs),
            total_cost=total_cost,
            cost_cap=cost_cap,
        )

    _print_final_summary(summaries, stopped_early, cost_cap, experiment_id)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iterated Prisoner's Dilemma runner (coop micro / risk micro / full).",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the run plan and exit (no API calls)",
    )
    parser.add_argument(
        "--micro-pilot",
        action="store_true",
        help="coop∈{0.2,0.8}, risk=0.5, 5 reps/cell = 10 runs",
    )
    parser.add_argument(
        "--risk-micro-pilot",
        action="store_true",
        help="coop=0.5, risk∈{0.2,0.8}, 5 reps/cell = 10 runs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip run_ids that already have MAX_ROUNDS rounds",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(
        run_experiment(
            dry_run=args.plan,
            micro_pilot=args.micro_pilot,
            risk_micro_pilot=args.risk_micro_pilot,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
