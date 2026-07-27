"""
prompt_revision_v1 — aynı 3×3 faktöriyel tasarım, davranışsal cooperation prompt.

full_experiment_v1 ile parametreler aynı (9 koşul × 5 rep, Haiku 4.5, TR,
temperature=0.2, EXTRACTION_LIMIT_RATIO=0.12). Tek fark: decision_agent
cooperation satırı davranışsal tanım kullanır (experiment_id = prompt_revision_v1).

Kullanım (repo kökünden):
    python experiments/run_prompt_revision.py --plan
    python experiments/run_prompt_revision.py --micro-pilot   # 10 run, API
    python experiments/run_prompt_revision.py                 # 45 run, API

Gereksinimler: ANTHROPIC_API_KEY (.env). Mock yok.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
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

EXPERIMENT_ID = "prompt_revision_v1"
MAX_ROUNDS = 15
REPLICATIONS = 5
COST_CAP_USD = 7.00
MICRO_COST_CAP_USD = 2.00

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
    print(f"PROMPT REVISION v1 — DENEY PLANI ({mode}, çalıştırılmadı)")
    print("=" * 72)
    print(f"  experiment_id          : {EXPERIMENT_ID}")
    print(f"  database               : {RESULTS_DB_PATH}")
    print(f"  prompt variant         : behavioral cooperation anchors")
    print(f"  toplam run             : {len(specs)}")
    print(f"  tekrar/koşul (N)       : {REPLICATIONS}")
    print(f"  max_rounds             : {MAX_ROUNDS}")
    print(f"  model                  : {settings.ANTHROPIC_MODEL}")
    print(f"  temperature            : {settings.TEMPERATURE}")
    print(f"  EXTRACTION_LIMIT_RATIO : {settings.EXTRACTION_LIMIT_RATIO}")
    print(f"  agent sayısı           : {settings.AGENT_COUNT}")
    if micro_pilot:
        print(
            f"  filtre                 : coop ∈ {{0.2, 0.8}}, risk=0.2 "
            f"({len(specs)} run)"
        )
        print(f"  maliyet güvenlik sınırı: ${MICRO_COST_CAP_USD:.2f}")
    else:
        print(f"  koşul sayısı           : {len(TRAIT_LEVELS) ** 2} (3×3)")
        print(f"  maliyet güvenlik sınırı: ${COST_CAP_USD:.2f}")

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
    print(f"KOŞUL TAMAMLANDI — {condition_label}")
    print(f"{'=' * 72}")
    print(
        f"  trait                  : coop={spec.coop_value} ({spec.coop_level}), "
        f"risk={spec.risk_value} ({spec.risk_level})"
    )
    print(f"  bu koşul maliyeti      : ${cond_cost:.4f}")
    print(f"  ilerleme               : {runs_completed}/{total_runs} run")
    print(f"  kümülatif maliyet      : ${total_cost:.4f} / ${cost_cap:.2f}")
    print("  tekrar özeti:")
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
    print(f"DENEY ÖZET — {EXPERIMENT_ID}")
    print(f"{'=' * 72}")
    total_cost = sum(s.cost_usd for s in summaries)
    print(f"  tamamlanan run         : {len(summaries)}")
    print(f"  toplam maliyet         : ${total_cost:.4f}")
    if stopped_early:
        print(f"  ⚠ Erken durduruldu (maliyet sınırı ${cost_cap:.2f})")

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
        f"\n  (fiyatlandırma: ${_INPUT_COST_PER_M:.2f}/M input, "
        f"${_OUTPUT_COST_PER_M:.2f}/M output — Claude Haiku 4.5)"
    )


async def run_experiment(*, dry_run: bool = False, micro_pilot: bool = False) -> None:
    specs = build_run_plan(micro_pilot=micro_pilot)
    cost_cap = MICRO_COST_CAP_USD if micro_pilot else COST_CAP_USD

    if dry_run:
        _print_plan(specs, micro_pilot=micro_pilot)
        print("\n--plan modu: hiçbir koşu çalıştırılmadı.")
        return

    mode = "MICRO-PILOT" if micro_pilot else "FULL"
    print("=" * 72)
    print(f"PROMPT REVISION v1 — {mode} BAŞLIYOR")
    print("=" * 72)
    print(f"  experiment_id : {EXPERIMENT_ID}")
    print(f"  toplam run    : {len(specs)}")
    print(f"  maliyet sınırı: ${cost_cap:.2f}")

    _register_plan(specs)
    print(f"\n  experiment_conditions tablosuna {len(specs)} satır yazıldı/güncellendi.")

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
                f"\n⚠ Maliyet sınırı (${cost_cap:.2f}) aşıldı — "
                f"koşu atlandı: {spec.run_id} ve sonrası."
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
                f"\n⚠ Toplam maliyet ${total_cost:.4f} — "
                f"${cost_cap:.2f} sınırını aştı. Kalan koşular durduruldu."
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

    _print_final_summary(summaries, stopped_early, cost_cap)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "prompt_revision_v1: behavioral cooperation prompt, "
            "same design as full_experiment_v1."
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Run planını yazdır ve çık (API çağrısı yok)",
    )
    parser.add_argument(
        "--micro-pilot",
        action="store_true",
        help="Sadece coop∈{0.2,0.8}, risk=0.2, 5 rep/hücre = 10 run",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(
        run_experiment(dry_run=args.plan, micro_pilot=args.micro_pilot)
    )


if __name__ == "__main__":
    main()
