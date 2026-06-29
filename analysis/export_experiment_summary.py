"""
Export per-run and aggregate experiment summaries from results.db.

Phase 0 deliverable: per-run round-0 metrics exported from results.db.

Usage (repo root):
    python analysis/export_experiment_summary.py
    python analysis/export_experiment_summary.py --experiment full_experiment_v1
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.config import settings
from core.database import RESULTS_DB_PATH

INITIAL_POOL_CAPACITY = 100.0
DEFAULT_EXPERIMENTS = ("full_experiment_v1", "control_group_v1")
ROUND0_CSV_PATH = _ROOT / "data" / "amads_round0_summary.csv"
PHASE0_REPORT_PATH = _ROOT / "data" / "amads_phase0_report.md"

_PER_RUN_SQL = """
SELECT
    c.experiment_id,
    c.run_id,
    c.coop_level,
    c.risk_level,
    c.coop_value,
    c.risk_value,
    c.replication,
    frac.extraction_fraction_r0,
    m.gini_r0,
    m.cooperation_score_r0,
    m.total_extraction_r0,
    m.pool_after_r0,
    met.rounds_played,
    met.last_round
FROM experiment_conditions c
JOIN (
    SELECT
        experiment_id,
        run_id,
        AVG(
            CASE WHEN declared_max > 0
                 THEN extraction_amount / declared_max
                 ELSE 0 END
        ) AS extraction_fraction_r0
    FROM agent_decisions
    WHERE round_number = 0
    GROUP BY experiment_id, run_id
) frac ON c.experiment_id = frac.experiment_id AND c.run_id = frac.run_id
JOIN (
    SELECT
        experiment_id,
        run_id,
        gini_coefficient AS gini_r0,
        cooperation_score_avg AS cooperation_score_r0,
        total_extraction AS total_extraction_r0,
        pool_after AS pool_after_r0
    FROM metrics_snapshots
    WHERE round_number = 0
) m ON c.experiment_id = m.experiment_id AND c.run_id = m.run_id
JOIN (
    SELECT
        experiment_id,
        run_id,
        COUNT(round_number) AS rounds_played,
        MAX(round_number) AS last_round
    FROM metrics_snapshots
    GROUP BY experiment_id, run_id
) met ON c.experiment_id = met.experiment_id AND c.run_id = met.run_id
WHERE c.experiment_id = ?
ORDER BY c.coop_level, c.risk_level, c.replication
"""


@dataclass(frozen=True)
class PerRunRow:
    experiment_id: str
    run_id: str
    coop_level: str
    risk_level: str
    coop_value: float
    risk_value: float
    replication: int
    extraction_fraction_r0: float
    gini_r0: float
    cooperation_score_r0: float
    total_extraction_r0: float
    pool_after_r0: float
    extraction_over_capacity_r0: float
    rounds_played: int
    collapse_round: int | None
    terminated_early: bool


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


def fetch_per_run_rows(
    conn: sqlite3.Connection, experiment_id: str
) -> list[PerRunRow]:
    rows = conn.execute(_PER_RUN_SQL, (experiment_id,)).fetchall()
    result: list[PerRunRow] = []
    for row in rows:
        rounds_played = int(row[12])
        last_round = int(row[13])
        collapse = last_round if rounds_played < settings.MAX_ROUNDS else None
        total_ext = float(row[10])
        result.append(
            PerRunRow(
                experiment_id=row[0],
                run_id=row[1],
                coop_level=row[2],
                risk_level=row[3],
                coop_value=float(row[4]),
                risk_value=float(row[5]),
                replication=int(row[6]),
                extraction_fraction_r0=float(row[7]),
                gini_r0=float(row[8]),
                cooperation_score_r0=float(row[9]),
                total_extraction_r0=total_ext,
                pool_after_r0=float(row[11]),
                extraction_over_capacity_r0=total_ext / INITIAL_POOL_CAPACITY,
                rounds_played=rounds_played,
                collapse_round=collapse,
                terminated_early=collapse is not None,
            )
        )
    return result


def write_csv(path: Path, rows: list[PerRunRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "run_id",
        "coop_level",
        "risk_level",
        "coop_value",
        "risk_value",
        "replication",
        "extraction_fraction_r0",
        "gini_r0",
        "cooperation_score_r0",
        "total_extraction_r0",
        "pool_after_r0",
        "extraction_over_capacity_r0",
        "rounds_played",
        "collapse_round",
        "terminated_early",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "experiment_id": row.experiment_id,
                    "run_id": row.run_id,
                    "coop_level": row.coop_level,
                    "risk_level": row.risk_level,
                    "coop_value": row.coop_value,
                    "risk_value": row.risk_value,
                    "replication": row.replication,
                    "extraction_fraction_r0": f"{row.extraction_fraction_r0:.6f}",
                    "gini_r0": f"{row.gini_r0:.6f}",
                    "cooperation_score_r0": f"{row.cooperation_score_r0:.6f}",
                    "total_extraction_r0": f"{row.total_extraction_r0:.6f}",
                    "pool_after_r0": f"{row.pool_after_r0:.6f}",
                    "extraction_over_capacity_r0": f"{row.extraction_over_capacity_r0:.6f}",
                    "rounds_played": row.rounds_played,
                    "collapse_round": "" if row.collapse_round is None else row.collapse_round,
                    "terminated_early": row.terminated_early,
                }
            )


def _aggregate_stats(rows: list[PerRunRow]) -> dict[str, float | int]:
    fracs = [r.extraction_fraction_r0 for r in rows]
    ginis = [r.gini_r0 for r in rows]
    coops = [r.cooperation_score_r0 for r in rows]
    cap_ratios = [r.extraction_over_capacity_r0 for r in rows]
    early = sum(1 for r in rows if r.terminated_early)
    return {
        "n_runs": len(rows),
        "extraction_fraction_mean": statistics.mean(fracs),
        "extraction_fraction_stdev": statistics.stdev(fracs) if len(fracs) > 1 else 0.0,
        "gini_mean": statistics.mean(ginis),
        "gini_stdev": statistics.stdev(ginis) if len(ginis) > 1 else 0.0,
        "cooperation_score_mean": statistics.mean(coops),
        "cooperation_score_stdev": statistics.stdev(coops) if len(coops) > 1 else 0.0,
        "extraction_over_capacity_mean": statistics.mean(cap_ratios),
        "extraction_over_capacity_stdev": (
            statistics.stdev(cap_ratios) if len(cap_ratios) > 1 else 0.0
        ),
        "rounds_played_mean": statistics.mean([r.rounds_played for r in rows]),
        "early_collapse_count": early,
        "coop_to_fraction_r": _pearson_r(
            [r.coop_value for r in rows], fracs
        ),
        "risk_to_fraction_r": _pearson_r(
            [r.risk_value for r in rows], fracs
        ),
    }


def _read_logged_cost(experiment_id: str) -> float | None:
    if experiment_id != "full_experiment_v1":
        return 0.0
    log_path = _ROOT / "data" / "full_experiment_v1.log"
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        if "toplam maliyet" in line.lower():
            part = line.split(":")[-1].strip().replace("$", "")
            return float(part)
    return None


def write_phase0_report(
    path: Path,
    all_rows: dict[str, list[PerRunRow]],
    db_path: str,
) -> None:
    today = date.today().isoformat()
    lines = [
        "# AMADS Phase 0 — Veri Özeti",
        "",
        f"Oluşturulma: {today}",
        f"Kaynak DB: `{db_path}`",
        f"Round penceresi: **round 0** (confound'suz trait fidelity analizi)",
        f"Başlangıç havuz kapasitesi: **{INITIAL_POOL_CAPACITY}**",
        "",
        "## Deney envanteri",
        "",
    ]

    with sqlite3.connect(db_path) as conn:
        for exp_id in all_rows:
            n_cond = conn.execute(
                "SELECT COUNT(*) FROM experiment_conditions WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()[0]
            n_metrics = conn.execute(
                "SELECT COUNT(*) FROM metrics_snapshots WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()[0]
            n_decisions = conn.execute(
                "SELECT COUNT(*) FROM agent_decisions WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()[0]
            lines.append(
                f"- **{exp_id}**: {n_cond} run, {n_metrics} metrics satırı, "
                f"{n_decisions} agent_decisions satırı"
            )

    lines.extend(["", "## Round-0 özet (deney bazında)", ""])

    for exp_id, rows in all_rows.items():
        stats = _aggregate_stats(rows)
        lines.extend(
            [
                f"### {exp_id} (n={stats['n_runs']})",
                "",
                "| Metrik | Ortalama | SD |",
                "|---|---|---|",
                f"| extraction_fraction | {stats['extraction_fraction_mean']:.4f} | "
                f"{stats['extraction_fraction_stdev']:.4f} |",
                f"| cooperation_score_r0 | {stats['cooperation_score_mean']:.4f} | "
                f"{stats['cooperation_score_stdev']:.4f} |",
                f"| gini_r0 | {stats['gini_mean']:.4f} | {stats['gini_stdev']:.4f} |",
                f"| total_extraction / pool_capacity | "
                f"{stats['extraction_over_capacity_mean']:.4f} | "
                f"{stats['extraction_over_capacity_stdev']:.4f} |",
                f"| rounds_played (tam run) | {stats['rounds_played_mean']:.2f} | — |",
                f"| erken collapse (round < {settings.MAX_ROUNDS}) | "
                f"{stats['early_collapse_count']}/{stats['n_runs']} | — |",
                "",
                "**Trait fidelity (round 0, Pearson r):**",
                f"- cooperation → extraction_fraction: r = {stats['coop_to_fraction_r']:.4f}",
                f"- risk_tolerance → extraction_fraction: r = {stats['risk_to_fraction_r']:.4f}",
                "",
            ]
        )

    logged_cost = _read_logged_cost("full_experiment_v1")
    per_run_cost = (
        logged_cost / 45 if logged_cost is not None else None
    )
    lines.extend(
        [
            "## Maliyet kalibrasyonu (full_experiment_v1)",
            "",
            "| Kaynak | Değer |",
            "|---|---|",
            f"| Model | `{settings.ANTHROPIC_MODEL}` |",
            f"| Logged toplam maliyet | "
            f"{'$' + f'{logged_cost:.4f}' if logged_cost is not None else 'log bulunamadı'} |",
            f"| Run başına (45 run) | "
            f"{'$' + f'{per_run_cost:.4f}' if per_run_cost is not None else '—'} |",
            "| Master ref tahmini (Haiku tam run) | ~$0.17 |",
            "| Master ref tahmini (Sonnet tam run) | ~$0.50 |",
            "",
            "## Export CSV sütunları",
            "",
            "`amads_round0_summary.csv` alanları:",
            "",
            "- `extraction_fraction_r0` — ortalama(extraction / declared_max), round 0",
            "- `cooperation_score_r0`, `gini_r0` — Referee round-0 metrikleri",
            "- `extraction_over_capacity_r0` — total_extraction_r0 / başlangıç kapasitesi",
            "- `collapse_round`, `terminated_early` — erken sonlanma (round < max_rounds)",
            "",
            "## Çıktı dosyaları",
            "",
            f"- `{ROUND0_CSV_PATH.relative_to(_ROOT)}` — run başına round-0 satırları",
            f"- `{PHASE0_REPORT_PATH.relative_to(_ROOT)}` — bu rapor",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AMADS experiment summaries")
    parser.add_argument(
        "--db",
        default=RESULTS_DB_PATH,
        help=f"SQLite path (default: {RESULTS_DB_PATH})",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help="Experiment id (repeatable; default: full + control)",
    )
    parser.add_argument(
        "--csv",
        default=str(ROUND0_CSV_PATH),
        help="Per-run CSV output path",
    )
    parser.add_argument(
        "--report",
        default=str(PHASE0_REPORT_PATH),
        help="Phase 0 markdown report path",
    )
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"HATA: DB bulunamadı: {db_path}", file=sys.stderr)
        sys.exit(1)

    experiments = tuple(args.experiments) if args.experiments else DEFAULT_EXPERIMENTS
    all_rows: dict[str, list[PerRunRow]] = {}

    with sqlite3.connect(db_path) as conn:
        for exp_id in experiments:
            rows = fetch_per_run_rows(conn, exp_id)
            if not rows:
                print(f"UYARI: {exp_id} için veri yok", file=sys.stderr)
            all_rows[exp_id] = rows

    combined = [row for rows in all_rows.values() for row in rows]
    write_csv(Path(args.csv), combined)
    write_phase0_report(Path(args.report), all_rows, db_path)

    print(f"Yazıldı: {args.csv} ({len(combined)} satır)")
    print(f"Yazıldı: {args.report}")
    for exp_id, rows in all_rows.items():
        stats = _aggregate_stats(rows)
        print(
            f"  {exp_id}: n={stats['n_runs']}, "
            f"fraction_mean={stats['extraction_fraction_mean']:.4f}, "
            f"collapse={stats['early_collapse_count']}/{stats['n_runs']}"
        )


if __name__ == "__main__":
    main()
