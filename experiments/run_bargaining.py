"""
bargaining_v1 — simplified ultimatum game (2-player sequential).

Isolated from CPR commons dilemma. Same abstract trait-prompt style as
full_experiment_v1 (not behavioral prompt_revision_v1 wording).

Usage (repo root):
    python experiments/run_bargaining.py --plan --micro-pilot
    python experiments/run_bargaining.py --plan
    python experiments/run_bargaining.py --micro-pilot   # Faz B — API
    python experiments/run_bargaining.py                 # Faz C — API (after B)

Gereksinimler (API koşuları): ANTHROPIC_API_KEY (.env). --plan API çağırmaz.
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

from core.config import settings
from core.state import EnvironmentSnapshot, TraitProfile
from scenarios.bargaining.agents import (
    _INPUT_COST_PER_M,
    _OUTPUT_COST_PER_M,
    reset_token_usage,
    token_usage,
)
from scenarios.bargaining.graph import app
from scenarios.bargaining.persistence import (
    BARGAINING_DB_PATH,
    init_bargaining_db,
    register_bargaining_conditions,
)
from scenarios.bargaining.referee import _env_params_for_round
from scenarios.bargaining.state import (
    PIE_SIZE,
    PROPOSER_ID,
    RESPONDER_ID,
    BargainingState,
)

EXPERIMENT_ID = "bargaining_v1"
RISK_EXPERIMENT_ID = "bargaining_risk_v1"
MAX_ROUNDS = 15
REPLICATIONS = 5
COST_CAP_USD = 3.00
MICRO_COST_CAP_USD = 1.00

# Empirical ballpark: 2 agents × 15 rounds vs CPR 5×15 → ~40% of ~$0.12 ≈ $0.05
ESTIMATED_COST_PER_RUN_USD = 0.05

COOP_LEVELS: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
}
# Full grid: proposer risk ∈ {0.2, 0.8} only (no medium)
PROPOSER_RISK_LEVELS: dict[str, float] = {
    "low": 0.2,
    "high": 0.8,
}

# Micro-pilot: proposer_coop ∈ {0.2, 0.8}, proposer_risk=0.2,
# responder fixed (coop=0.5, risk=0.5)
MICRO_PROPOSER_COOP = ("low", "high")
MICRO_PROPOSER_RISK = "low"
FIXED_RESPONDER_COOP = "medium"
FIXED_RESPONDER_RISK = "medium"  # 0.5
FIXED_RESPONDER_RISK_VALUE = 0.5

# Risk micro-pilot: proposer_coop=0.5 fixed, proposer_risk ∈ {0.2, 0.8}
RISK_MICRO_PROPOSER_COOP = "medium"
RISK_MICRO_PROPOSER_RISK = ("low", "high")


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    proposer_coop_level: str
    proposer_risk_level: str
    responder_coop_level: str
    responder_risk_level: str
    proposer_coop: float
    proposer_risk: float
    responder_coop: float
    responder_risk: float
    replication: int


@dataclass
class RunSummary:
    run_id: str
    rounds_played: int
    termination_reason: str | None
    cost_usd: float
    mean_keep_fraction: float
    accept_rate: float


def _run_id(
    p_coop: str,
    p_risk: str,
    r_coop: str,
    rep: int,
) -> str:
    return f"pcoop_{p_coop}_prisk_{p_risk}_rcoop_{r_coop}_rep{rep}"


def build_run_plan(
    *,
    micro_pilot: bool = False,
    risk_micro_pilot: bool = False,
) -> list[RunSpec]:
    if risk_micro_pilot:
        p_coop_levels = (RISK_MICRO_PROPOSER_COOP,)
        p_risk_levels = RISK_MICRO_PROPOSER_RISK
        r_coop_levels = (FIXED_RESPONDER_COOP,)
    elif micro_pilot:
        p_coop_levels = MICRO_PROPOSER_COOP
        p_risk_levels = (MICRO_PROPOSER_RISK,)
        r_coop_levels = (FIXED_RESPONDER_COOP,)
    else:
        p_coop_levels = tuple(COOP_LEVELS)
        p_risk_levels = tuple(PROPOSER_RISK_LEVELS)
        r_coop_levels = tuple(COOP_LEVELS)

    specs: list[RunSpec] = []
    for p_coop, p_risk, r_coop in product(p_coop_levels, p_risk_levels, r_coop_levels):
        for rep in range(1, REPLICATIONS + 1):
            specs.append(
                RunSpec(
                    run_id=_run_id(p_coop, p_risk, r_coop, rep),
                    proposer_coop_level=p_coop,
                    proposer_risk_level=p_risk,
                    responder_coop_level=r_coop,
                    responder_risk_level=FIXED_RESPONDER_RISK,
                    proposer_coop=COOP_LEVELS[p_coop],
                    proposer_risk=PROPOSER_RISK_LEVELS[p_risk],
                    responder_coop=COOP_LEVELS[r_coop],
                    responder_risk=FIXED_RESPONDER_RISK_VALUE,
                    replication=rep,
                )
            )
    return specs


def _make_initial_state(spec: RunSpec, experiment_id: str) -> BargainingState:
    tp, rs = _env_params_for_round(0)
    pie = PIE_SIZE
    return BargainingState(
        experiment_id=experiment_id,
        run_id=spec.run_id,
        max_rounds=MAX_ROUNDS,
        proposer_trait=TraitProfile(
            agent_id=PROPOSER_ID,
            cooperation_assigned=spec.proposer_coop,
            risk_tolerance_assigned=spec.proposer_risk,
            profile_label=(
                f"P_{spec.proposer_coop_level}Coop_{spec.proposer_risk_level}Risk"
            ),
        ),
        responder_trait=TraitProfile(
            agent_id=RESPONDER_ID,
            cooperation_assigned=spec.responder_coop,
            risk_tolerance_assigned=spec.responder_risk,
            profile_label=(
                f"R_{spec.responder_coop_level}Coop_{spec.responder_risk_level}Risk"
            ),
        ),
        env_snapshot=EnvironmentSnapshot(
            pool_current=pie,
            pool_capacity=pie,
            regen_rate=1.0,
            max_extractable_this_round=pie,
            round_number=0,
            is_collapsed=False,
        ),
        time_pressure=tp,
        resource_scarcity=rs,
    )


def _condition_key(spec: RunSpec) -> str:
    return (
        f"pcoop_{spec.proposer_coop_level}_"
        f"prisk_{spec.proposer_risk_level}_"
        f"rcoop_{spec.responder_coop_level}"
    )


def _register_plan(specs: list[RunSpec], experiment_id: str) -> None:
    rows = [
        (
            s.run_id,
            s.proposer_coop_level,
            s.proposer_risk_level,
            s.responder_coop_level,
            s.responder_risk_level,
            s.proposer_coop,
            s.proposer_risk,
            s.responder_coop,
            s.responder_risk,
            s.replication,
        )
        for s in specs
    ]
    register_bargaining_conditions(experiment_id, rows, BARGAINING_DB_PATH)


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
    n_cells = len(specs) // REPLICATIONS
    est_total = len(specs) * ESTIMATED_COST_PER_RUN_USD
    cost_cap = (
        MICRO_COST_CAP_USD if (micro_pilot or risk_micro_pilot) else COST_CAP_USD
    )

    print("=" * 72)
    print(f"BARGAINING — DENEY PLANI ({mode}, çalıştırılmadı)")
    print("=" * 72)
    print(f"  experiment_id          : {experiment_id}")
    print(f"  game                   : ultimatum (2-player sequential)")
    print(f"  database               : {BARGAINING_DB_PATH}")
    print(f"  prompt style           : abstract/symbolic (full_experiment_v1)")
    print(f"  toplam run             : {len(specs)}")
    print(f"  koşul hücresi          : {n_cells}")
    print(f"  tekrar/koşul (N)       : {REPLICATIONS}")
    print(f"  max_rounds             : {MAX_ROUNDS}")
    print(f"  pie_size               : {PIE_SIZE:.0f}")
    print(f"  agents/round           : 2 (proposer → responder)")
    print(f"  LLM calls/run (üst)    : {2 * MAX_ROUNDS}")
    print(f"  model                  : {settings.ANTHROPIC_MODEL}")
    print(f"  temperature            : {settings.TEMPERATURE}")
    print(f"  tahmini maliyet/run    : ~${ESTIMATED_COST_PER_RUN_USD:.2f}")
    print(f"  tahmini toplam maliyet : ~${est_total:.2f}")
    print(f"  maliyet güvenlik sınırı: ${cost_cap:.2f}")
    if risk_micro_pilot:
        print(
            "  filtre                 : proposer_coop=0.5, "
            "proposer_risk ∈ {0.2, 0.8}, responder=(0.5, 0.5)"
        )
    elif micro_pilot:
        print(
            "  filtre                 : proposer_coop ∈ {0.2, 0.8}, "
            "proposer_risk=0.2, responder=(0.5, 0.5)"
        )
    else:
        print(
            "  grid                   : proposer_coop∈{0.2,0.5,0.8} × "
            "proposer_risk∈{0.2,0.8} × responder_coop∈{0.2,0.5,0.8} "
            f"(responder_risk={FIXED_RESPONDER_RISK_VALUE}) → "
            f"{len(COOP_LEVELS) * len(PROPOSER_RISK_LEVELS) * len(COOP_LEVELS)} hücre"
        )

    print(f"\n  {'#':>3}  {'run_id':<48} {'p_coop':>6} {'p_risk':>6} {'r_coop':>6}  rep")
    print("  " + "-" * 78)
    for i, spec in enumerate(specs, 1):
        print(
            f"  {i:3d}  {spec.run_id:<48} "
            f"{spec.proposer_coop:6.1f} {spec.proposer_risk:6.1f} "
            f"{spec.responder_coop:6.1f}  {spec.replication}"
        )

    print(
        f"\n  (fiyatlandırma: ${_INPUT_COST_PER_M:.2f}/M input, "
        f"${_OUTPUT_COST_PER_M:.2f}/M output — Claude Haiku 4.5)"
    )


async def _run_single(spec: RunSpec, experiment_id: str) -> RunSummary:
    reset_token_usage()
    cost_before = token_usage.estimated_cost_usd()

    print(
        f"\n  → {spec.run_id}  "
        f"(p_coop={spec.proposer_coop}, p_risk={spec.proposer_risk}, "
        f"r_coop={spec.responder_coop}, rep={spec.replication})"
    )

    final_state = await app.ainvoke(_make_initial_state(spec, experiment_id))
    cost_usd = token_usage.estimated_cost_usd() - cost_before
    metrics = final_state["metrics_history"]
    n = len(metrics)
    mean_keep = (
        sum(m.keep_fraction for m in metrics) / n if n else 0.0
    )
    accept_rate = (
        sum(1 for m in metrics if m.accepted) / n if n else 0.0
    )

    return RunSummary(
        run_id=spec.run_id,
        rounds_played=n,
        termination_reason=final_state.get("termination_reason"),
        cost_usd=cost_usd,
        mean_keep_fraction=mean_keep,
        accept_rate=accept_rate,
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
        f"  traits                 : p_coop={spec.proposer_coop}, "
        f"p_risk={spec.proposer_risk}, r_coop={spec.responder_coop}"
    )
    print(f"  bu koşul maliyeti      : ${cond_cost:.4f}")
    print(f"  ilerleme               : {runs_completed}/{total_runs} run")
    print(f"  kümülatif maliyet      : ${total_cost:.4f} / ${cost_cap:.2f}")
    for s in condition_summaries:
        print(
            f"    {s.run_id}: {s.rounds_played} round, "
            f"keep_frac={s.mean_keep_fraction:.3f}, "
            f"accept={s.accept_rate:.2f}, ${s.cost_usd:.4f}"
        )


def _print_final_summary(
    summaries: list[RunSummary],
    stopped_early: bool,
    cost_cap: float,
    experiment_id: str,
) -> None:
    print(f"\n{'=' * 72}")
    print(f"DENEY ÖZET — {experiment_id}")
    print(f"{'=' * 72}")
    total_cost = sum(s.cost_usd for s in summaries)
    print(f"  tamamlanan run         : {len(summaries)}")
    print(f"  toplam maliyet         : ${total_cost:.4f}")
    if stopped_early:
        print(f"  ⚠ Erken durduruldu (maliyet sınırı ${cost_cap:.2f})")

    init_bargaining_db(BARGAINING_DB_PATH)
    with sqlite3.connect(BARGAINING_DB_PATH) as conn:
        rounds = conn.execute(
            "SELECT COUNT(*) FROM bargaining_rounds WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]
        metrics = conn.execute(
            "SELECT COUNT(*) FROM bargaining_metrics WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]
        conditions = conn.execute(
            "SELECT COUNT(*) FROM bargaining_conditions WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]
    print(f"  DB — bargaining_rounds  : {rounds}")
    print(f"  DB — bargaining_metrics : {metrics}")
    print(f"  DB — bargaining_conditions: {conditions}")
    print(
        f"\n  (fiyatlandırma: ${_INPUT_COST_PER_M:.2f}/M input, "
        f"${_OUTPUT_COST_PER_M:.2f}/M output — Claude Haiku 4.5)"
    )


async def run_experiment(
    *,
    dry_run: bool = False,
    micro_pilot: bool = False,
    risk_micro_pilot: bool = False,
) -> None:
    if micro_pilot and risk_micro_pilot:
        raise SystemExit("Use either --micro-pilot or --risk-micro-pilot, not both.")

    experiment_id = RISK_EXPERIMENT_ID if risk_micro_pilot else EXPERIMENT_ID
    specs = build_run_plan(
        micro_pilot=micro_pilot, risk_micro_pilot=risk_micro_pilot
    )
    cost_cap = (
        MICRO_COST_CAP_USD if (micro_pilot or risk_micro_pilot) else COST_CAP_USD
    )

    if dry_run:
        _print_plan(
            specs,
            micro_pilot=micro_pilot,
            risk_micro_pilot=risk_micro_pilot,
            experiment_id=experiment_id,
        )
        print("\n--plan modu: hiçbir koşu çalıştırılmadı (API çağrısı yok).")
        return

    if risk_micro_pilot:
        mode = "RISK-MICRO-PILOT"
    elif micro_pilot:
        mode = "MICRO-PILOT"
    else:
        mode = "FULL"
    print("=" * 72)
    print(f"BARGAINING — {mode} BAŞLIYOR")
    print("=" * 72)
    print(f"  experiment_id : {experiment_id}")
    print(f"  toplam run    : {len(specs)}")
    print(f"  maliyet sınırı: ${cost_cap:.2f}")

    _register_plan(specs, experiment_id)
    print(f"\n  bargaining_conditions tablosuna {len(specs)} satır yazıldı/güncellendi.")

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
        summary = await _run_single(spec, experiment_id)
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

    _print_final_summary(summaries, stopped_early, cost_cap, experiment_id)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bargaining ultimatum game runner "
            "(coop micro / risk micro / full)."
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
        help=(
            "proposer_coop∈{0.2,0.8}, proposer_risk=0.2, "
            "responder=(0.5,0.5), 5 rep/hücre = 10 run"
        ),
    )
    parser.add_argument(
        "--risk-micro-pilot",
        action="store_true",
        help=(
            "bargaining_risk_v1: proposer_coop=0.5, "
            "proposer_risk∈{0.2,0.8}, responder=(0.5,0.5), 10 run"
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(
        run_experiment(
            dry_run=args.plan,
            micro_pilot=args.micro_pilot,
            risk_micro_pilot=args.risk_micro_pilot,
        )
    )


if __name__ == "__main__":
    main()
