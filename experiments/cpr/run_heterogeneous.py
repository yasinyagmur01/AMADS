"""
heterogeneous_v1 — within-run heterogeneous trait populations.

4 population compositions × 5 replications = 20 runs.
Same commons dilemma, Referee, and metrics as full_experiment_v1;
only trait assignment differs (per-agent TraitProfile via agent_traits).

Usage (repo root):
    python experiments/cpr/run_heterogeneous.py --plan   # plan only, no API
    python experiments/cpr/run_heterogeneous.py          # run experiment
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from dataclasses import dataclass
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
from core.state import EnvironmentSnapshot, SimulationState
from environment.shocks import build_mock_dev_shock_schedule
from experiments.cpr.heterogeneous_conditions import (
    CONDITION_ORDER,
    CONDITIONS,
    EXPERIMENT_ID,
    REPLICATIONS,
    HeteroCondition,
    ensure_composition_table,
    experiment_condition_rows,
    make_agent_traits,
    run_id_for,
    upsert_compositions,
)

MAX_ROUNDS = 15
COST_CAP_USD = 3.50
# Empirical ballpark from full_experiment_v1 Haiku runs (~$0.10–0.15 / run).
ESTIMATED_COST_PER_RUN_USD = 0.12


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    condition_id: str
    condition: HeteroCondition
    replication: int


@dataclass
class RunSummary:
    run_id: str
    condition_id: str
    rounds_played: int
    termination_reason: str | None
    cost_usd: float


def build_run_plan() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for cid in CONDITION_ORDER:
        cond = CONDITIONS[cid]
        for rep in range(1, REPLICATIONS + 1):
            specs.append(
                RunSpec(
                    run_id=run_id_for(cid, rep),
                    condition_id=cid,
                    condition=cond,
                    replication=rep,
                )
            )
    return specs


def _make_initial_state(spec: RunSpec) -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id=EXPERIMENT_ID,
        run_id=spec.run_id,
        max_rounds=MAX_ROUNDS,
        agent_traits=make_agent_traits(spec.condition),
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


def _register_plan() -> None:
    rows = experiment_condition_rows()
    register_experiment_conditions(EXPERIMENT_ID, rows, RESULTS_DB_PATH)
    with sqlite3.connect(RESULTS_DB_PATH) as conn:
        ensure_composition_table(conn)
        upsert_compositions(conn, EXPERIMENT_ID)
        conn.commit()


async def _run_single(spec: RunSpec) -> RunSummary:
    reset_token_usage()
    cost_before = token_usage.estimated_cost_usd()

    print(
        f"\n  → {spec.run_id}  ({spec.condition.label}: {spec.condition.name}, "
        f"rep={spec.replication})"
    )

    final_state = await app.ainvoke(_make_initial_state(spec))
    cost_usd = token_usage.estimated_cost_usd() - cost_before

    return RunSummary(
        run_id=spec.run_id,
        condition_id=spec.condition_id,
        rounds_played=len(final_state["metrics_history"]),
        termination_reason=final_state["termination_reason"],
        cost_usd=cost_usd,
    )


def _print_plan(specs: list[RunSpec]) -> None:
    n_conditions = len(CONDITION_ORDER)
    est_total = len(specs) * ESTIMATED_COST_PER_RUN_USD

    print("=" * 72)
    print("HETEROGENEOUS_V1 — EXPERIMENT PLAN (dry run / not executed)")
    print("=" * 72)
    print(f"  experiment_id          : {EXPERIMENT_ID}")
    print(f"  database               : {RESULTS_DB_PATH}")
    print(f"  conditions             : {n_conditions} (fixed population compositions)")
    print(f"  replications/condition : {REPLICATIONS}")
    print(f"  total runs             : {len(specs)}")
    print(f"  max_rounds             : {MAX_ROUNDS}")
    print(f"  model                  : {settings.ANTHROPIC_MODEL}")
    print(f"  temperature            : {settings.TEMPERATURE}")
    print(f"  EXTRACTION_LIMIT_RATIO : {settings.EXTRACTION_LIMIT_RATIO}")
    print(
        f"  agent count            : {settings.AGENT_COUNT} "
        f"(real LLM; traits are per-agent / heterogeneous)"
    )
    print(f"  cost safety cap        : ${COST_CAP_USD:.2f}")
    print(
        f"  estimated cost         : ~${est_total:.2f} "
        f"(~${ESTIMATED_COST_PER_RUN_USD:.2f}/run × {len(specs)})"
    )
    print(f"  progress report        : every {REPLICATIONS} runs (when condition ends)")
    print()

    print("Population compositions:")
    for cid in CONDITION_ORDER:
        cond = CONDITIONS[cid]
        print(f"\n  {cond.label} — {cond.name}")
        print(
            f"    mean coop={cond.mean_coop():.3f}, mean risk={cond.mean_risk():.3f}"
        )
        for agent_id in sorted(cond.composition):
            pair = cond.composition[agent_id]
            print(
                f"      {agent_id}: coop={pair['coop']:.1f}, risk={pair['risk']:.1f}"
            )

    print(f"\n{len(specs)}-run list:")
    print(f"  {'#':>3}  {'run_id':<18} {'condition':<10} {'label':<10}  rep")
    print("  " + "-" * 56)
    for i, spec in enumerate(specs, 1):
        print(
            f"  {i:3d}  {spec.run_id:<18} "
            f"{spec.condition_id:<10} {spec.condition.label:<10}  {spec.replication}"
        )

    print("\nCondition groups (5 replications each):")
    for i, cid in enumerate(CONDITION_ORDER, 1):
        cond = CONDITIONS[cid]
        print(f"  Group {i}: {cond.label} — {cond.name}")


def _print_condition_checkpoint(
    *,
    condition: HeteroCondition,
    condition_summaries: list[RunSummary],
    runs_completed: int,
    total_runs: int,
    total_cost: float,
) -> None:
    cond_cost = sum(s.cost_usd for s in condition_summaries)
    print(f"\n{'=' * 72}")
    print(f"CONDITION COMPLETE — {condition.label} ({condition.name})")
    print(f"{'=' * 72}")
    print(f"  cost this condition    : ${cond_cost:.4f}")
    print(f"  progress               : {runs_completed}/{total_runs} runs")
    print(f"  cumulative cost        : ${total_cost:.4f} / ${COST_CAP_USD:.2f}")
    print("  replication summary:")
    for s in condition_summaries:
        print(
            f"    {s.run_id}: {s.rounds_played} round, "
            f"termination={s.termination_reason or '—'}, "
            f"${s.cost_usd:.4f}"
        )


def _print_final_summary(summaries: list[RunSummary], stopped_early: bool) -> None:
    print(f"\n{'=' * 72}")
    print(f"EXPERIMENT SUMMARY — {EXPERIMENT_ID}")
    print(f"{'=' * 72}")
    total_cost = sum(s.cost_usd for s in summaries)
    print(f"  completed runs         : {len(summaries)}")
    print(f"  total cost             : ${total_cost:.4f}")
    if stopped_early:
        print(f"  ⚠ Stopped early (cost cap ${COST_CAP_USD:.2f})")

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
    compositions = 0
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "heterogeneous_compositions" in tables:
        compositions = conn.execute(
            "SELECT COUNT(*) FROM heterogeneous_compositions WHERE experiment_id = ?",
            (EXPERIMENT_ID,),
        ).fetchone()[0]
    conn.close()
    print(f"  DB — metrics_snapshots          : {metrics}")
    print(f"  DB — agent_decisions            : {decisions}")
    print(f"  DB — experiment_conditions      : {conditions}")
    print(f"  DB — heterogeneous_compositions : {compositions}")
    print(
        f"\n  (pricing: ${_INPUT_COST_PER_M:.2f}/M input, "
        f"${_OUTPUT_COST_PER_M:.2f}/M output — Claude Haiku 4.5)"
    )


async def run_experiment(*, dry_run: bool = False) -> None:
    specs = build_run_plan()

    if dry_run:
        _print_plan(specs)
        print("\n--plan mode: no runs executed.")
        return

    print("=" * 72)
    print("HETEROGENEOUS_V1 — EXPERIMENT STARTING")
    print("=" * 72)
    print(f"  experiment_id : {EXPERIMENT_ID}")
    print(f"  total runs    : {len(specs)}")
    print(f"  cost cap      : ${COST_CAP_USD:.2f}")

    _register_plan()
    print(
        f"\n  Wrote experiment_conditions + heterogeneous_compositions "
        f"({len(specs)} runs / {len(CONDITION_ORDER)} compositions)."
    )

    summaries: list[RunSummary] = []
    total_cost = 0.0
    stopped_early = False
    condition_buffer: list[RunSummary] = []
    current_cid: str | None = None

    for i, spec in enumerate(specs):
        if current_cid is None:
            current_cid = spec.condition_id

        if spec.condition_id != current_cid:
            _print_condition_checkpoint(
                condition=CONDITIONS[current_cid],
                condition_summaries=condition_buffer,
                runs_completed=len(summaries),
                total_runs=len(specs),
                total_cost=total_cost,
            )
            condition_buffer = []
            current_cid = spec.condition_id

        if total_cost >= COST_CAP_USD:
            print(
                f"\n⚠ Cost cap (${COST_CAP_USD:.2f}) exceeded — "
                f"skipping run: {spec.run_id} and remaining."
            )
            stopped_early = True
            break

        summary = await _run_single(spec)
        summaries.append(summary)
        condition_buffer.append(summary)
        total_cost += summary.cost_usd

        if total_cost > COST_CAP_USD:
            print(
                f"\n⚠ Total cost ${total_cost:.4f} — "
                f"exceeded ${COST_CAP_USD:.2f} cap. Remaining conditions stopped."
            )
            stopped_early = True
            _print_condition_checkpoint(
                condition=CONDITIONS[current_cid],
                condition_summaries=condition_buffer,
                runs_completed=len(summaries),
                total_runs=len(specs),
                total_cost=total_cost,
            )
            break

        if (i + 1) % REPLICATIONS == 0:
            _print_condition_checkpoint(
                condition=CONDITIONS[current_cid],
                condition_summaries=condition_buffer,
                runs_completed=len(summaries),
                total_runs=len(specs),
                total_cost=total_cost,
            )
            condition_buffer = []
            current_cid = None

    _print_final_summary(summaries, stopped_early)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "heterogeneous_v1: 4 population compositions × 5 reps = 20 runs."
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the 20-run plan and exit (no API calls)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(run_experiment(dry_run=args.plan))


if __name__ == "__main__":
    main()
