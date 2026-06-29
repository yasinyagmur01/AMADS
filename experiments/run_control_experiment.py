"""
Kontrol grubu deneyi — LLM yok, deterministik control_agent.

9 koşul (coop × risk ∈ {0.2, 0.5, 0.8}²) × 3 tekrar = 27 run.
Formül: extraction = declared_max × (1 − cooperation_assigned); risk kullanılmaz.

Kullanım (repo kökünden):
    python experiments/run_control_experiment.py --plan
    python experiments/run_control_experiment.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

os.environ.setdefault("LANGSMITH_TRACING", "false")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langgraph.graph import END, StateGraph

from agents.control_agent import run_control_agent_fanout
from analysis.trait_fidelity import fetch_run_fidelity
from core.config import settings
from core.database import RESULTS_DB_PATH, register_experiment_conditions
from core.state import EnvironmentSnapshot, SimulationState, TraitProfile
from environment.shocks import build_mock_dev_shock_schedule
from referee.referee_node import run_referee

try:
    from scipy.stats import pearsonr
except ImportError as exc:
    raise RuntimeError(f"scipy gerekli: pip install scipy ({exc})") from exc

EXPERIMENT_ID = "control_group_v1"
LLM_EXPERIMENT_ID = "full_experiment_v1"
LLM_COOP_FRACTION_R = 0.456
MAX_ROUNDS = 15
REPLICATIONS = 3
CORRELATION_MAX_ROUND = 0

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


def _build_graph():
    workflow = StateGraph(SimulationState)
    workflow.add_node("agent_fanout", run_control_agent_fanout)
    workflow.add_node("referee", run_referee)
    workflow.set_entry_point("agent_fanout")
    workflow.add_edge("agent_fanout", "referee")
    workflow.add_conditional_edges(
        "referee",
        lambda state: END if state.is_terminated else "agent_fanout",
    )
    return workflow.compile()


def _print_plan(specs: list[RunSpec]) -> None:
    print("=" * 72)
    print("KONTROL GRUBU DENEY PLANI (çalıştırılmadı)")
    print("=" * 72)
    print(f"  experiment_id          : {EXPERIMENT_ID}")
    print(f"  database               : {RESULTS_DB_PATH}")
    print(f"  koşul sayısı           : {len(TRAIT_LEVELS) ** 2} (3×3 kartezyen)")
    print(f"  tekrar/koşul (N)       : {REPLICATIONS}")
    print(f"  toplam run             : {len(specs)}")
    print(f"  max_rounds             : {MAX_ROUNDS}")
    print(f"  agent                  : control_agent (deterministik, LLM yok)")
    print(f"  formül                 : extraction = declared_max × (1 − coop)")
    print(f"  maliyet                : $0.00")
    print()


async def _run_single(spec: RunSpec, app) -> RunSummary:
    print(
        f"\n  → {spec.run_id}  "
        f"(coop={spec.coop_value}, risk={spec.risk_value}, rep={spec.replication})"
    )
    final_state = await app.ainvoke(_make_initial_state(spec))
    return RunSummary(
        run_id=spec.run_id,
        coop_level=spec.coop_level,
        risk_level=spec.risk_level,
        rounds_played=len(final_state["metrics_history"]),
        termination_reason=final_state["termination_reason"],
    )


def _print_correlation_comparison(control_r: float, control_p: float, n_runs: int) -> None:
    print(f"\n{'=' * 72}")
    print("COOP → EXTRACTION_FRACTION KORELASYON KARŞILAŞTIRMASI")
    print(f"(round penceresi: 0–{CORRELATION_MAX_ROUND}, extraction / declared_max)")
    print("=" * 72)
    print(f"  {'Grup':<22} {'experiment_id':<22} {'r':>8} {'p':>10} {'n':>5}")
    print("  " + "-" * 70)
    print(
        f"  {'Kontrol (kural tabanlı)':<22} {EXPERIMENT_ID:<22} "
        f"{control_r:8.3f} {control_p:10.4f} {n_runs:5d}"
    )
    print(
        f"  {'LLM (Haiku 4.5)':<22} {LLM_EXPERIMENT_ID:<22} "
        f"{LLM_COOP_FRACTION_R:8.3f} {'0.0016':>10} {'45':>5}"
    )
    print()
    if control_r < 0 and LLM_COOP_FRACTION_R > 0:
        print(
            "  Kontrol grubu beklenen negatif yönü doğruluyor (coop↑ → fraction↓). "
            "LLM grubu ters fidelity gösteriyor (coop↑ → fraction↑)."
        )
    elif control_r < 0:
        print("  Kontrol grubu: coop↑ → fraction↓ (beklenen yön).")
    print()


def _compute_control_correlation() -> tuple[float, float, int]:
    with sqlite3.connect(RESULTS_DB_PATH) as conn:
        rows = fetch_run_fidelity(
            conn, EXPERIMENT_ID, max_round=CORRELATION_MAX_ROUND
        )
    if not rows:
        raise RuntimeError(f"'{EXPERIMENT_ID}' için analiz verisi bulunamadı.")
    coop_x = [r.coop_assigned for r in rows]
    frac_y = [r.extraction_fraction for r in rows]
    r, p = pearsonr(coop_x, frac_y)
    return r, p, len(rows)


async def run_experiment(*, dry_run: bool = False) -> None:
    specs = build_run_plan()

    if dry_run:
        _print_plan(specs)
        print("--plan modu: hiçbir koşu çalıştırılmadı.")
        return

    print("=" * 72)
    print("KONTROL GRUBU DENEYİ BAŞLIYOR")
    print("=" * 72)
    print(f"  experiment_id : {EXPERIMENT_ID}")
    print(f"  toplam run    : {len(specs)}")
    print(f"  maliyet       : $0.00 (LLM yok)")

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
    print(f"\n  experiment_conditions tablosuna {len(specs)} satır yazıldı.")

    app = _build_graph()
    summaries: list[RunSummary] = []

    for spec in specs:
        summaries.append(await _run_single(spec, app))

    print(f"\n{'=' * 72}")
    print(f"DENEY ÖZET — {EXPERIMENT_ID}")
    print("=" * 72)
    print(f"  tamamlanan run : {len(summaries)}")
    print(f"  toplam maliyet : $0.00")

    with sqlite3.connect(RESULTS_DB_PATH) as conn:
        metrics = conn.execute(
            "SELECT COUNT(*) FROM metrics_snapshots WHERE experiment_id = ?",
            (EXPERIMENT_ID,),
        ).fetchone()[0]
        decisions = conn.execute(
            "SELECT COUNT(*) FROM agent_decisions WHERE experiment_id = ?",
            (EXPERIMENT_ID,),
        ).fetchone()[0]
    print(f"  DB — metrics_snapshots : {metrics}")
    print(f"  DB — agent_decisions   : {decisions}")

    control_r, control_p, n_runs = _compute_control_correlation()
    _print_correlation_comparison(control_r, control_p, n_runs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kontrol grubu deneyi (9 koşul × 3 tekrar, LLM yok).",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="27 run planını yazdır ve çık",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(run_experiment(dry_run=args.plan))


if __name__ == "__main__":
    main()
