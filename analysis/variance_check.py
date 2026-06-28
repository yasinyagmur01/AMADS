"""
Run-to-run variance analysis for a single experiment_id in results.db.

Compares extraction_amount across all runs (match rate, per-cell stdev) and
round-level variance for gini_coefficient / cooperation_score_avg. Reports mean
round variance and a two-sample t-test power analysis (required N per group).

Usage (from repo root):
    python analysis/variance_check.py ratio_fine_tune
    python analysis/variance_check.py ratio_fine_tune --effect-size 0.5
    python analysis/variance_check.py ratio_fine_tune --effect-delta 0.05 --verbose

Requires: statsmodels (pip install statsmodels)
"""

from __future__ import annotations

import argparse
import statistics
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.database import RESULTS_DB_PATH

try:
    from statsmodels.stats.power import TTestIndPower
except ImportError as exc:
    TTestIndPower = None  # type: ignore[misc, assignment]
    _STATSMODELS_IMPORT_ERROR = exc
else:
    _STATSMODELS_IMPORT_ERROR = None


@dataclass(frozen=True)
class ExtractionCell:
    round_number: int
    agent_id: str
    values: dict[str, float]


@dataclass
class ExtractionReport:
    runs: list[str]
    rounds: list[int]
    agents: list[str]
    cells: list[ExtractionCell]
    total_cells: int
    exact_match_count: int
    mismatch_stdevs: list[float]

    @property
    def match_rate(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return self.exact_match_count / self.total_cells


@dataclass(frozen=True)
class RoundMetricVariance:
    round_number: int
    values: dict[str, float]
    variance: float
    stdev: float


@dataclass
class MetricVarianceReport:
    metric_name: str
    rounds: list[RoundMetricVariance]

    @property
    def mean_variance(self) -> float:
        if not self.rounds:
            return 0.0
        return statistics.mean(r.variance for r in self.rounds)

    @property
    def mean_stdev(self) -> float:
        if self.mean_variance <= 0:
            return 0.0
        return self.mean_variance**0.5


@dataclass(frozen=True)
class PowerEstimate:
    metric_name: str
    mean_round_variance: float
    mean_round_stdev: float
    effect_size_cohens_d: float
    alpha: float
    power: float
    required_n_per_group: float


def _discover_runs(conn: sqlite3.Connection, experiment_id: str) -> list[str]:
    cur = conn.cursor()
    runs = {
        row[0]
        for row in cur.execute(
            """
            SELECT DISTINCT run_id FROM agent_decisions WHERE experiment_id = ?
            UNION
            SELECT DISTINCT run_id FROM metrics_snapshots WHERE experiment_id = ?
            """,
            (experiment_id, experiment_id),
        )
    }
    return sorted(runs)


def _fetch_extraction_rows(
    conn: sqlite3.Connection, experiment_id: str, runs: list[str]
) -> list[tuple[str, int, str, float]]:
    placeholders = ",".join("?" * len(runs))
    cur = conn.cursor()
    return cur.execute(
        f"""
        SELECT run_id, round_number, agent_id, extraction_amount
        FROM agent_decisions
        WHERE experiment_id = ? AND run_id IN ({placeholders})
        ORDER BY round_number, agent_id, run_id
        """,
        (experiment_id, *runs),
    ).fetchall()


def _fetch_metric_rows(
    conn: sqlite3.Connection,
    experiment_id: str,
    runs: list[str],
    column: str,
) -> list[tuple[str, int, float]]:
    if column not in {"gini_coefficient", "cooperation_score_avg"}:
        raise ValueError(f"Unsupported metric column: {column}")
    placeholders = ",".join("?" * len(runs))
    cur = conn.cursor()
    return cur.execute(
        f"""
        SELECT run_id, round_number, {column}
        FROM metrics_snapshots
        WHERE experiment_id = ? AND run_id IN ({placeholders})
        ORDER BY round_number, run_id
        """,
        (experiment_id, *runs),
    ).fetchall()


def analyze_extraction(
    conn: sqlite3.Connection, experiment_id: str, runs: list[str]
) -> ExtractionReport:
    pivot: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    for run_id, rnd, agent_id, amount in _fetch_extraction_rows(
        conn, experiment_id, runs
    ):
        pivot[(rnd, agent_id)][run_id] = amount

    rounds = sorted({k[0] for k in pivot})
    agents = sorted({k[1] for k in pivot})
    cells: list[ExtractionCell] = []
    exact_match_count = 0
    mismatch_stdevs: list[float] = []

    for rnd in rounds:
        for agent_id in agents:
            values = pivot.get((rnd, agent_id), {})
            if not values:
                continue
            cell = ExtractionCell(rnd, agent_id, dict(values))
            cells.append(cell)
            amounts = [values[r] for r in runs if r in values]
            if len(amounts) < len(runs):
                continue
            if len(set(amounts)) == 1:
                exact_match_count += 1
            elif len(amounts) >= 2:
                mismatch_stdevs.append(statistics.stdev(amounts))

    complete_cells = sum(
        1 for c in cells if all(r in c.values for r in runs)
    )

    return ExtractionReport(
        runs=runs,
        rounds=rounds,
        agents=agents,
        cells=cells,
        total_cells=complete_cells,
        exact_match_count=exact_match_count,
        mismatch_stdevs=mismatch_stdevs,
    )


def analyze_metric_variance(
    conn: sqlite3.Connection,
    experiment_id: str,
    runs: list[str],
    column: str,
    metric_name: str,
) -> MetricVarianceReport:
    pivot: dict[int, dict[str, float]] = defaultdict(dict)
    for run_id, rnd, value in _fetch_metric_rows(conn, experiment_id, runs, column):
        pivot[rnd][run_id] = value

    round_rows: list[RoundMetricVariance] = []
    for rnd in sorted(pivot):
        values = pivot[rnd]
        present = [values[r] for r in runs if r in values]
        if len(present) < len(runs):
            continue
        if len(present) >= 2:
            var = statistics.variance(present)
            std = statistics.stdev(present)
        else:
            var = 0.0
            std = 0.0
        round_rows.append(
            RoundMetricVariance(
                round_number=rnd,
                values={r: values[r] for r in runs if r in values},
                variance=var,
                stdev=std,
            )
        )

    return MetricVarianceReport(metric_name=metric_name, rounds=round_rows)


def _resolve_cohens_d(
    mean_stdev: float,
    effect_size: float | None,
    effect_delta: float | None,
) -> float:
    if effect_size is not None and effect_delta is not None:
        raise ValueError("Specify only one of --effect-size or --effect-delta.")
    if effect_delta is not None:
        if mean_stdev <= 0:
            raise ValueError(
                "Cannot derive Cohen's d from --effect-delta: mean round stdev is zero."
            )
        return effect_delta / mean_stdev
    if effect_size is not None:
        return effect_size
    return 0.5


def estimate_required_n(
    metric_report: MetricVarianceReport,
    *,
    effect_size: float | None,
    effect_delta: float | None,
    alpha: float,
    power: float,
) -> PowerEstimate:
    if TTestIndPower is None:
        raise RuntimeError(
            "statsmodels is required for power analysis. "
            f"Install with: pip install statsmodels ({_STATSMODELS_IMPORT_ERROR})"
        )

    mean_var = metric_report.mean_variance
    mean_std = metric_report.mean_stdev
    cohens_d = _resolve_cohens_d(mean_std, effect_size, effect_delta)

    analysis = TTestIndPower()
    required_n = analysis.solve_power(
        effect_size=cohens_d,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )

    return PowerEstimate(
        metric_name=metric_report.metric_name,
        mean_round_variance=mean_var,
        mean_round_stdev=mean_std,
        effect_size_cohens_d=cohens_d,
        alpha=alpha,
        power=power,
        required_n_per_group=float(required_n),
    )


def _short_run_label(run_id: str) -> str:
    if len(run_id) <= 12:
        return run_id
    return run_id[-8:]


def _print_extraction_report(report: ExtractionReport, *, verbose: bool) -> None:
    n_runs = len(report.runs)
    print("=" * 90)
    print("AGENT_DECISIONS — extraction_amount")
    print("=" * 90)
    print(f"Runs ({n_runs}): {', '.join(report.runs)}")
    if report.rounds:
        print(
            f"Rounds: {report.rounds[0]}–{report.rounds[-1]}  |  "
            f"Agents: {', '.join(report.agents)}"
        )
    print()
    print(f"Tam eşleşen hücre (round × agent): {report.exact_match_count}/{report.total_cells}")
    print(f"Eşleşme oranı: {100 * report.match_rate:.1f}%")
    if report.mismatch_stdevs:
        print(
            "StdDev (farklı hücreler): "
            f"min={min(report.mismatch_stdevs):.6f}, "
            f"max={max(report.mismatch_stdevs):.6f}, "
            f"mean={statistics.mean(report.mismatch_stdevs):.6f}"
        )

    print("\nRound bazında tam eşleşen agent sayısı:")
    for rnd in report.rounds:
        agents_in_round = [
            c for c in report.cells if c.round_number == rnd and len(c.values) >= n_runs
        ]
        n_match = sum(
            1
            for c in agents_in_round
            if len({c.values[r] for r in report.runs if r in c.values}) == 1
        )
        print(f"  round {rnd:2d}: {n_match}/{len(agents_in_round)} agent")

    if not verbose or n_runs > 6:
        if n_runs > 6 and verbose:
            print("\n(--verbose: >6 run olduğu için hücre tablosu atlandı)")
        return

    labels = [_short_run_label(r) for r in report.runs]
    col_w = max(10, max(len(l) for l in labels) + 2)
    header = f"{'Round':>5} {'Agent':>8} | " + " ".join(f"{l:>{col_w}}" for l in labels)
    header += f" | {'Match':>5} {'StdDev':>10}"
    print()
    print(header)
    print("-" * len(header))

    for rnd in report.rounds:
        for agent_id in report.agents:
            cell = next(
                (c for c in report.cells if c.round_number == rnd and c.agent_id == agent_id),
                None,
            )
            if cell is None or not all(r in cell.values for r in report.runs):
                continue
            amounts = [cell.values[r] for r in report.runs]
            is_exact = len(set(amounts)) == 1
            std = 0.0 if is_exact else statistics.stdev(amounts)
            vals = " ".join(f"{cell.values[r]:{col_w}.6f}" for r in report.runs)
            match_str = "YES" if is_exact else "NO"
            std_str = "—" if is_exact else f"{std:.6f}"
            print(f"{rnd:5d} {agent_id:>8} | {vals} | {match_str:>5} {std_str:>10}")


def _print_metric_report(report: MetricVarianceReport) -> None:
    print()
    print("=" * 90)
    print(f"METRICS_SNAPSHOTS — {report.metric_name}")
    print("=" * 90)
    if not report.rounds:
        print("(veri yok)")
        return

    runs = sorted({r for row in report.rounds for r in row.values})
    labels = [_short_run_label(r) for r in runs]
    col_w = max(10, max(len(l) for l in labels) + 2)
    header = f"{'Round':>5} | " + " ".join(f"{l:>{col_w}}" for l in labels)
    header += f" | {'variance':>12} {'stdev':>10}"
    print(header)
    print("-" * len(header))

    exact_rounds = 0
    for row in report.rounds:
        if row.variance == 0:
            exact_rounds += 1
        vals = " ".join(f"{row.values[r]:{col_w}.6f}" for r in runs)
        var_str = "0" if row.variance == 0 else f"{row.variance:.2e}"
        std_str = "—" if row.stdev == 0 else f"{row.stdev:.6f}"
        print(f"{row.round_number:5d} | {vals} | {var_str:>12} {std_str:>10}")

    variances = [r.variance for r in report.rounds]
    print()
    print(f"Toplam round: {len(report.rounds)}")
    print(f"Tam eşleşen round: {exact_rounds}/{len(report.rounds)}")
    print(
        f"Round varyans — min={min(variances):.2e}, "
        f"max={max(variances):.2e}, mean={report.mean_variance:.2e}"
    )
    print(f"Round varyans ortalaması (σ²): {report.mean_variance:.6e}")
    print(f"√(ortalama varyans) (σ):       {report.mean_stdev:.6f}")


def _print_power_section(
    estimates: list[PowerEstimate],
    current_n: int,
    *,
    effect_size: float | None,
    effect_delta: float | None,
) -> None:
    print()
    print("=" * 90)
    print("POWER ANALYSIS (iki bağımsız grup, two-sided t-test)")
    print("=" * 90)
    if effect_delta is not None:
        print(f"Efekt büyüklüğü: mutlak fark (delta) = {effect_delta}")
    elif effect_size is not None:
        print(f"Efekt büyüklüğü: Cohen's d = {effect_size}")
    else:
        print("Efekt büyüklüğü: Cohen's d = 0.5 (varsayılan, orta efekt)")

    for est in estimates:
        print()
        print(f"--- {est.metric_name} ---")
        print(f"  Ortalama round varyansı (σ²): {est.mean_round_variance:.6e}")
        print(f"  Ortalama round stdev (σ):     {est.mean_round_stdev:.6f}")
        print(f"  Kullanılan Cohen's d:         {est.effect_size_cohens_d:.4f}")
        print(f"  α = {est.alpha}, power = {est.power}")
        print(f"  Gerekli N (grup başına):      {est.required_n_per_group:.1f}")
        print(f"  Mevcut run sayısı:            {current_n}")
        if current_n >= est.required_n_per_group:
            print("  → Mevcut run sayısı yeterli (her koşul için).")
        else:
            deficit = est.required_n_per_group - current_n
            print(f"  → En az {deficit:.0f} ek run/koşul gerekli.")


def run_analysis(
    experiment_id: str,
    db_path: str = RESULTS_DB_PATH,
    *,
    effect_size: float | None = None,
    effect_delta: float | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
    verbose: bool = False,
) -> None:
    path = Path(db_path)
    if not path.exists():
        print(f"Hata: veritabanı bulunamadı: {path}", file=sys.stderr)
        sys.exit(1)

    with sqlite3.connect(path) as conn:
        runs = _discover_runs(conn, experiment_id)
        if not runs:
            print(
                f"Hata: '{experiment_id}' için run bulunamadı ({path}).",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"Experiment: {experiment_id}")
        print(f"Database:   {path}")
        print(f"Runs:       {len(runs)} → {', '.join(runs)}")
        print()

        extraction = analyze_extraction(conn, experiment_id, runs)
        gini = analyze_metric_variance(
            conn, experiment_id, runs, "gini_coefficient", "gini_coefficient"
        )
        coop = analyze_metric_variance(
            conn,
            experiment_id,
            runs,
            "cooperation_score_avg",
            "cooperation_score_avg",
        )

    _print_extraction_report(extraction, verbose=verbose)
    _print_metric_report(gini)
    _print_metric_report(coop)

    power_estimates = [
        estimate_required_n(
            gini,
            effect_size=effect_size,
            effect_delta=effect_delta,
            alpha=alpha,
            power=power,
        ),
        estimate_required_n(
            coop,
            effect_size=effect_size,
            effect_delta=effect_delta,
            alpha=alpha,
            power=power,
        ),
    ]
    _print_power_section(
        power_estimates,
        current_n=len(runs),
        effect_size=effect_size,
        effect_delta=effect_delta,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run-to-run variance analysis for all runs under an experiment_id."
        ),
    )
    parser.add_argument(
        "experiment_id",
        help="Experiment identifier (e.g. ratio_fine_tune)",
    )
    parser.add_argument(
        "--db",
        default=RESULTS_DB_PATH,
        help=f"SQLite results path (default: {RESULTS_DB_PATH})",
    )
    parser.add_argument(
        "--effect-size",
        type=float,
        default=None,
        metavar="D",
        help="Target Cohen's d for power analysis (default: 0.5 if --effect-delta omitted)",
    )
    parser.add_argument(
        "--effect-delta",
        type=float,
        default=None,
        metavar="DELTA",
        help="Absolute metric difference to detect; converted to Cohen's d via observed σ",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Desired statistical power (default: 0.80)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-cell extraction_amount table (skipped when >6 runs)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run_analysis(
        args.experiment_id,
        args.db,
        effect_size=args.effect_size,
        effect_delta=args.effect_delta,
        alpha=args.alpha,
        power=args.power,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
