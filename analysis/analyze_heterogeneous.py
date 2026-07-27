"""
Analyze heterogeneous_v1 results — produce Tables 1–4.

Usage (repo root):
    python analysis/analyze_heterogeneous.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import settings
from core.database import RESULTS_DB_PATH
from experiments.heterogeneous_conditions import (
    CONDITION_ORDER,
    CONDITIONS,
    EXPERIMENT_ID,
    load_composition_from_db,
    parse_condition_id,
)

BASELINE_ID = "full_experiment_v1"
MEDIUM_COOP = 0.5
MEDIUM_RISK = 0.5
FREE_RIDER_AGENT = "agent_5"


@dataclass(frozen=True)
class RunGroupStats:
    run_id: str
    condition_id: str
    total_extraction: float
    gini_mean: float
    rounds_played: int
    collapse_round: int | None


@dataclass(frozen=True)
class AgentRunExtraction:
    run_id: str
    condition_id: str
    agent_id: str
    coop: float
    risk: float
    total_extraction: float


def _collapse_round(rounds_played: int, last_round: int) -> int | None:
    if rounds_played < settings.MAX_ROUNDS:
        return int(last_round)
    return None


def _collapse_numeric(row: RunGroupStats) -> float:
    """Survivors counted as MAX_ROUNDS for mean collapse/survival."""
    if row.collapse_round is not None:
        return float(row.collapse_round)
    return float(settings.MAX_ROUNDS)


def _trait_band(coop: float) -> str:
    if coop >= 0.7:
        return "high_coop"
    if coop <= 0.3:
        return "low_coop"
    return "medium_coop"


def fetch_group_stats(
    conn: sqlite3.Connection, experiment_id: str
) -> list[RunGroupStats]:
    rows = conn.execute(
        """
        SELECT
            c.run_id,
            c.coop_level AS condition_id,
            COALESCE(SUM(m.total_extraction), 0) AS total_extraction,
            AVG(m.gini_coefficient) AS gini_mean,
            COUNT(m.round_number) AS rounds_played,
            MAX(m.round_number) AS last_round
        FROM experiment_conditions c
        LEFT JOIN metrics_snapshots m
          ON m.experiment_id = c.experiment_id AND m.run_id = c.run_id
        WHERE c.experiment_id = ?
        GROUP BY c.run_id, c.coop_level
        ORDER BY c.coop_level, c.replication
        """,
        (experiment_id,),
    ).fetchall()
    result: list[RunGroupStats] = []
    for run_id, condition_id, total_ext, gini_mean, rounds_played, last_round in rows:
        if rounds_played == 0:
            continue
        result.append(
            RunGroupStats(
                run_id=run_id,
                condition_id=condition_id,
                total_extraction=float(total_ext),
                gini_mean=float(gini_mean),
                rounds_played=int(rounds_played),
                collapse_round=_collapse_round(int(rounds_played), int(last_round)),
            )
        )
    return result


def fetch_agent_extractions(
    conn: sqlite3.Connection, experiment_id: str
) -> list[AgentRunExtraction]:
    rows = conn.execute(
        """
        SELECT
            d.run_id,
            c.coop_level AS condition_id,
            d.agent_id,
            SUM(d.extraction_amount) AS total_extraction
        FROM agent_decisions d
        JOIN experiment_conditions c
          ON c.experiment_id = d.experiment_id AND c.run_id = d.run_id
        WHERE d.experiment_id = ?
        GROUP BY d.run_id, c.coop_level, d.agent_id
        ORDER BY c.coop_level, d.run_id, d.agent_id
        """,
        (experiment_id,),
    ).fetchall()

    result: list[AgentRunExtraction] = []
    for run_id, condition_id, agent_id, total_ext in rows:
        cid = condition_id or parse_condition_id(run_id)
        if cid is None or cid not in CONDITIONS:
            continue
        composition = load_composition_from_db(conn, cid, experiment_id)
        if composition is None:
            composition = CONDITIONS[cid].composition
        pair = composition.get(agent_id)
        if pair is None:
            continue
        result.append(
            AgentRunExtraction(
                run_id=run_id,
                condition_id=cid,
                agent_id=agent_id,
                coop=float(pair["coop"]),
                risk=float(pair["risk"]),
                total_extraction=float(total_ext),
            )
        )
    return result


def fetch_baseline_medium(conn: sqlite3.Connection) -> list[RunGroupStats]:
    rows = conn.execute(
        """
        SELECT
            c.run_id,
            COALESCE(SUM(m.total_extraction), 0) AS total_extraction,
            AVG(m.gini_coefficient) AS gini_mean,
            COUNT(m.round_number) AS rounds_played,
            MAX(m.round_number) AS last_round
        FROM experiment_conditions c
        LEFT JOIN metrics_snapshots m
          ON m.experiment_id = c.experiment_id AND m.run_id = c.run_id
        WHERE c.experiment_id = ?
          AND c.coop_value = ?
          AND c.risk_value = ?
        GROUP BY c.run_id
        ORDER BY c.replication
        """,
        (BASELINE_ID, MEDIUM_COOP, MEDIUM_RISK),
    ).fetchall()
    result: list[RunGroupStats] = []
    for run_id, total_ext, gini_mean, rounds_played, last_round in rows:
        if rounds_played == 0:
            continue
        result.append(
            RunGroupStats(
                run_id=run_id,
                condition_id="medium_medium",
                total_extraction=float(total_ext),
                gini_mean=float(gini_mean),
                rounds_played=int(rounds_played),
                collapse_round=_collapse_round(int(rounds_played), int(last_round)),
            )
        )
    return result


def _mean_or_none(vals: Iterable[float]) -> float | None:
    vals = list(vals)
    return mean(vals) if vals else None


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def print_table1(stats: list[RunGroupStats]) -> None:
    print("\nTABLO 1 — Group-level sonuçlar:")
    print("| Condition | Total Extract (mean) | Collapse Round (mean) | Gini (mean) |")
    print("|-----------|---------------------|----------------------|-------------|")
    by_cond: dict[str, list[RunGroupStats]] = defaultdict(list)
    for row in stats:
        by_cond[row.condition_id].append(row)
    for cid in CONDITION_ORDER:
        label = CONDITIONS[cid].label
        rows = by_cond.get(cid, [])
        if not rows:
            print(f"| {label} | — | — | — |")
            continue
        ext = _mean_or_none(r.total_extraction for r in rows)
        col = _mean_or_none(_collapse_numeric(r) for r in rows)
        gini = _mean_or_none(r.gini_mean for r in rows)
        print(f"| {label} | {_fmt(ext, 2)} | {_fmt(col, 2)} | {_fmt(gini, 4)} |")


def print_table2(agents: list[AgentRunExtraction]) -> None:
    print("\nTABLO 2 — Dominant agent etkisi (COND_H1 ve COND_H2):")
    print("| Condition | Agent type    | Mean extraction | % of group total |")
    print("|-----------|---------------|-----------------|------------------|")

    # Display order matches requested table
    specs = [
        ("h1", "high_coop", "High-coop (3)"),
        ("h1", "medium_coop", "Medium (1)"),
        ("h1", "low_coop", "Low-coop (1)"),
        ("h2", "low_coop", "Low-coop (3)"),
        ("h2", "medium_coop", "Medium (1)"),
        ("h2", "high_coop", "High-coop (1)"),
    ]

    by_cond: dict[str, list[AgentRunExtraction]] = defaultdict(list)
    for row in agents:
        by_cond[row.condition_id].append(row)

    for cid, band, type_label in specs:
        rows = by_cond.get(cid, [])
        label = CONDITIONS[cid].label
        if not rows:
            print(f"| {label} | {type_label} | — | — |")
            continue
        group_total = sum(r.total_extraction for r in rows)
        band_rows = [r for r in rows if _trait_band(r.coop) == band]
        if not band_rows:
            print(f"| {label} | {type_label} | — | — |")
            continue
        mean_ext = mean(r.total_extraction for r in band_rows)
        band_sum = sum(r.total_extraction for r in band_rows)
        pct = (100.0 * band_sum / group_total) if group_total > 0 else None
        print(
            f"| {label} | {type_label} | {_fmt(mean_ext, 2)} | {_fmt(pct, 1)}% |"
        )


def print_table3(
    hetero_stats: list[RunGroupStats], baseline: list[RunGroupStats]
) -> None:
    print("\nTABLO 3 — COND_H4 vs full_experiment_v1 medium baseline:")
    print("| Metric              | full_exp_v1 medium | COND_H4 |")
    print("|---------------------|-------------------|---------|")
    h4 = [r for r in hetero_stats if r.condition_id == "h4"]

    def _row(metric: str, base_v: float | None, h4_v: float | None, digits: int) -> None:
        print(f"| {metric:<19} | {_fmt(base_v, digits):>17} | {_fmt(h4_v, digits):>7} |")

    _row(
        "Total extract mean",
        _mean_or_none(r.total_extraction for r in baseline),
        _mean_or_none(r.total_extraction for r in h4),
        2,
    )
    _row(
        "Collapse round mean",
        _mean_or_none(_collapse_numeric(r) for r in baseline),
        _mean_or_none(_collapse_numeric(r) for r in h4),
        2,
    )
    _row(
        "Gini mean",
        _mean_or_none(r.gini_mean for r in baseline),
        _mean_or_none(r.gini_mean for r in h4),
        4,
    )


def print_table4(agents: list[AgentRunExtraction]) -> None:
    print("\nTABLO 4 — Free rider tespiti (COND_H1):")
    h1 = [r for r in agents if r.condition_id == "h1"]
    if not h1:
        print("Agent 5 vs others: —")
        return
    rider = [r for r in h1 if r.agent_id == FREE_RIDER_AGENT]
    others = [r for r in h1 if r.agent_id != FREE_RIDER_AGENT]
    rider_ext = _mean_or_none(r.total_extraction for r in rider)
    others_ext = _mean_or_none(r.total_extraction for r in others)
    if rider_ext is None or others_ext is None or others_ext == 0:
        print("Agent 5 vs others: —")
        return
    pct_more = 100.0 * (rider_ext - others_ext) / others_ext
    print(f"Agent 5 mean extraction : {_fmt(rider_ext, 2)}")
    print(f"Other 4 agents mean     : {_fmt(others_ext, 2)}")
    print(f"Agent 5 fazla (%)       : {pct_more:+.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Analyze {EXPERIMENT_ID} heterogeneous population results."
    )
    parser.add_argument(
        "--db",
        default=RESULTS_DB_PATH,
        help=f"SQLite path (default: {RESULTS_DB_PATH})",
    )
    args = parser.parse_args()

    print("=" * 72)
    print(f"HETEROGENEOUS ANALYSIS — {EXPERIMENT_ID}")
    print("=" * 72)
    print(f"  database : {args.db}")

    if not Path(args.db).exists():
        print("  DB missing — nothing to analyze.")
        raise SystemExit(1)

    with sqlite3.connect(args.db) as conn:
        hetero_stats = fetch_group_stats(conn, EXPERIMENT_ID)
        agents = fetch_agent_extractions(conn, EXPERIMENT_ID)
        baseline = fetch_baseline_medium(conn)

        print_table1(hetero_stats)
        print_table2(agents)
        print_table3(hetero_stats, baseline)
        print_table4(agents)

    print("\nDone.")


if __name__ == "__main__":
    main()
