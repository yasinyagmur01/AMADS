"""
Bölüm 12 — tam faktöriyel deney koşusu (9 koşul × 5 tekrar = 45 run).

9 koşul: cooperation_assigned × risk_tolerance_assigned ∈ {0.2, 0.5, 0.8}².
Her koşulda 5 agent'a aynı trait çifti atanır (koşullar arası fark, agent içi yok).

Kullanım (repo kökünden):
    python experiments/run_full_experiment.py --plan     # 45 run planını göster, çalıştırma
    python experiments/run_full_experiment.py            # deneyi başlat

Gereksinimler: ANTHROPIC_API_KEY (.env), gerçek Claude Haiku 4.5 (mock yok).
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

EXPERIMENT_ID = "full_experiment_v1"
MAX_ROUNDS = 15
REPLICATIONS = 5
COST_CAP_USD = 7.00

TRAIT_LEVELS: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
}

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


def build_run_plan() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for coop_level, risk_level in product(TRAIT_LEVELS, TRAIT_LEVELS):
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


async def _run_single(spec: RunSpec, app) -> RunSummary:
    reset_token_usage()
    cost_before = token_usage.estimated_cost_usd()

    print(f"\n  → {spec.run_id}  (coop={spec.coop_value}, risk={spec.risk_value}, rep={spec.replication})")

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


def _print_plan(specs: list[RunSpec]) -> None:
    print("=" * 72)
    print("BÖLÜM 12 — DENEY PLANI (çalıştırılmadı)")
    print("=" * 72)
    print(f"  experiment_id          : {EXPERIMENT_ID}")
    print(f"  database               : {RESULTS_DB_PATH}")
    print(f"  koşul sayısı           : {len(TRAIT_LEVELS) ** 2} (3×3 kartezyen)")
    print(f"  tekrar/koşul (N)       : {REPLICATIONS}")
    print(f"  toplam run             : {len(specs)}")
    print(f"  max_rounds             : {MAX_ROUNDS}")
    print(f"  model                  : {settings.ANTHROPIC_MODEL}")
    print(f"  temperature            : {settings.TEMPERATURE}")
    print(f"  EXTRACTION_LIMIT_RATIO : {settings.EXTRACTION_LIMIT_RATIO}")
    print(f"  agent sayısı           : {settings.AGENT_COUNT} (hepsi gerçek LLM, aynı trait/koşul)")
    print(f"  maliyet güvenlik sınırı: ${COST_CAP_USD:.2f}")
    print(f"  ilerleme raporu        : her {REPLICATIONS} run (koşul bitince)")
    print()

    print("Koşul matrisi (coop \\ risk):")
    header = f"{'':>10}" + "".join(f"{r:>12}" for r in TRAIT_LEVELS)
    print(header)
    for coop in TRAIT_LEVELS:
        row = f"{coop:>10}"
        for risk in TRAIT_LEVELS:
            row += f"{TRAIT_LEVELS[coop]:.1f}/{TRAIT_LEVELS[risk]:.1f}".rjust(12)
        print(row)

    print("\n45 run listesi:")
    print(f"  {'#':>3}  {'run_id':<32} {'coop':>5} {'risk':>5}  rep")
    print("  " + "-" * 58)
    for i, spec in enumerate(specs, 1):
        print(
            f"  {i:3d}  {spec.run_id:<32} "
            f"{spec.coop_value:5.1f} {spec.risk_value:5.1f}  {spec.replication}"
        )

    print("\nKoşul grupları (her biri 5 tekrar):")
    current_key = None
    group_num = 0
    for spec in specs:
        key = _condition_key(spec)
        if key != current_key:
            group_num += 1
            current_key = key
            print(
                f"  Grup {group_num}: {_condition_key(spec)} "
                f"(coop={spec.coop_value}, risk={spec.risk_value})"
            )


def _print_condition_checkpoint(
    *,
    condition_label: str,
    condition_specs: list[RunSpec],
    condition_summaries: list[RunSummary],
    runs_completed: int,
    total_runs: int,
    total_cost: float,
) -> None:
    cond_cost = sum(s.cost_usd for s in condition_summaries)
    spec = condition_specs[0]
    print(f"\n{'=' * 72}")
    print(f"KOŞUL TAMAMLANDI — {condition_label}")
    print(f"{'=' * 72}")
    print(f"  trait                  : coop={spec.coop_value} ({spec.coop_level}), "
          f"risk={spec.risk_value} ({spec.risk_level})")
    print(f"  bu koşul maliyeti      : ${cond_cost:.4f}")
    print(f"  ilerleme               : {runs_completed}/{total_runs} run")
    print(f"  kümülatif maliyet      : ${total_cost:.4f} / ${COST_CAP_USD:.2f}")
    print("  tekrar özeti:")
    for s in condition_summaries:
        print(
            f"    {s.run_id}: {s.rounds_played} round, "
            f"termination={s.termination_reason or '—'}, "
            f"${s.cost_usd:.4f}"
        )


def _print_final_summary(summaries: list[RunSummary], stopped_early: bool) -> None:
    print(f"\n{'=' * 72}")
    print(f"DENEY ÖZET — {EXPERIMENT_ID}")
    print(f"{'=' * 72}")
    total_cost = sum(s.cost_usd for s in summaries)
    print(f"  tamamlanan run         : {len(summaries)}")
    print(f"  toplam maliyet         : ${total_cost:.4f}")
    if stopped_early:
        print(f"  ⚠ Erken durduruldu (maliyet sınırı ${COST_CAP_USD:.2f})")

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


async def run_experiment(*, dry_run: bool = False) -> None:
    specs = build_run_plan()

    if dry_run:
        _print_plan(specs)
        print("\n--plan modu: hiçbir koşu çalıştırılmadı.")
        return

    print("=" * 72)
    print("BÖLÜM 12 — TAM DENEY BAŞLIYOR")
    print("=" * 72)
    print(f"  experiment_id : {EXPERIMENT_ID}")
    print(f"  toplam run    : {len(specs)}")
    print(f"  maliyet sınırı: ${COST_CAP_USD:.2f}")

    _register_plan(specs)
    print(f"\n  experiment_conditions tablosuna {len(specs)} satır yazıldı.")

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
            )
            condition_buffer = []
            current_condition = cond
            current_condition_specs = []

        if total_cost >= COST_CAP_USD:
            print(
                f"\n⚠ Maliyet sınırı (${COST_CAP_USD:.2f}) aşıldı — "
                f"koşu atlandı: {spec.run_id} ve sonrası."
            )
            stopped_early = True
            break

        current_condition_specs.append(spec)
        summary = await _run_single(spec, app)
        summaries.append(summary)
        condition_buffer.append(summary)
        total_cost += summary.cost_usd

        if total_cost > COST_CAP_USD:
            print(
                f"\n⚠ Toplam maliyet ${total_cost:.4f} — "
                f"${COST_CAP_USD:.2f} sınırını aştı. Kalan koşular durduruldu."
            )
            stopped_early = True
            _print_condition_checkpoint(
                condition_label=current_condition,
                condition_specs=current_condition_specs,
                condition_summaries=condition_buffer,
                runs_completed=len(summaries),
                total_runs=len(specs),
                total_cost=total_cost,
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
            )
            condition_buffer = []
            current_condition_specs = []
            current_condition = None

    _print_final_summary(summaries, stopped_early)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bölüm 12 tam faktöriyel deney (9 koşul × 5 tekrar = 45 run).",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="45 run planını yazdır ve çık (API çağrısı yok)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(run_experiment(dry_run=args.plan))


if __name__ == "__main__":
    main()
