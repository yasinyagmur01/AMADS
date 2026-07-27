"""
Compare full_experiment_v1 vs prompt_revision_v1 cooperation → extraction fidelity.

Success criterion (micro-pilot):
    mean_diff = mean(coop=0.2) − mean(coop=0.8) > 0.30
    (positive = expected direction: high coop extracts less)

Reference: full_experiment_v1 micro A/B gap was −3.60 (inverse).

Usage (repo root):
    python analysis/compare_prompt_revision.py
    python analysis/compare_prompt_revision.py --max-round 2
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.trait_fidelity import RunFidelityRow, fetch_run_fidelity
from core.database import RESULTS_DB_PATH

BASELINE_ID = "full_experiment_v1"
REVISION_ID = "prompt_revision_v1"
MICRO_RISK = 0.2
MICRO_COOP_LOW = 0.2
MICRO_COOP_HIGH = 0.8
SUCCESS_MEAN_DIFF = 0.30

try:
    from scipy.stats import pearsonr
except ImportError as exc:
    pearsonr = None  # type: ignore[misc, assignment]
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class CoopCorr:
    experiment_id: str
    n: int
    r: float | None
    p: float | None
    note: str = ""


@dataclass(frozen=True)
class MicroPilotGap:
    experiment_id: str
    n_low: int
    n_high: int
    mean_low: float | None
    mean_high: float | None
    mean_diff: float | None  # low − high (positive = expected fidelity)
    passes: bool | None
    note: str = ""


def _require_scipy() -> None:
    if pearsonr is None:
        raise RuntimeError(
            f"scipy gerekli. Kurulum: pip install scipy ({_SCIPY_IMPORT_ERROR})"
        )


def cooperation_extraction_corr(rows: list[RunFidelityRow]) -> CoopCorr:
    if len(rows) < 3:
        return CoopCorr(
            experiment_id="",
            n=len(rows),
            r=None,
            p=None,
            note="yetersiz n",
        )
    _require_scipy()
    x = [r.coop_assigned for r in rows]
    y = [r.extraction_fraction for r in rows]
    r, p = pearsonr(x, y)
    return CoopCorr(experiment_id="", n=len(rows), r=float(r), p=float(p))


def fetch_mean_extraction_by_condition(
    conn: sqlite3.Connection,
    experiment_id: str,
    *,
    coop_value: float,
    risk_value: float,
    max_round: int | None,
) -> tuple[list[float], str]:
    """Per-run mean extraction_amount for a trait cell; empty if no decisions."""
    round_filter = ""
    params: list = [experiment_id, coop_value, risk_value]
    if max_round is not None:
        round_filter = "AND d.round_number <= ?"
        params.append(max_round)

    sql = f"""
        SELECT
            c.run_id,
            AVG(d.extraction_amount) AS mean_extraction
        FROM experiment_conditions c
        JOIN agent_decisions d
          ON d.experiment_id = c.experiment_id AND d.run_id = c.run_id
        WHERE c.experiment_id = ?
          AND c.coop_value = ?
          AND c.risk_value = ?
          {round_filter}
        GROUP BY c.run_id
        ORDER BY c.replication
    """
    fetched = conn.execute(sql, params).fetchall()
    if not fetched:
        return [], "no agent_decisions for this cell"
    return [float(row[1]) for row in fetched], ""


def micro_pilot_gap(
    conn: sqlite3.Connection,
    experiment_id: str,
    *,
    max_round: int | None,
) -> MicroPilotGap:
    low_vals, low_note = fetch_mean_extraction_by_condition(
        conn,
        experiment_id,
        coop_value=MICRO_COOP_LOW,
        risk_value=MICRO_RISK,
        max_round=max_round,
    )
    high_vals, high_note = fetch_mean_extraction_by_condition(
        conn,
        experiment_id,
        coop_value=MICRO_COOP_HIGH,
        risk_value=MICRO_RISK,
        max_round=max_round,
    )
    note = low_note or high_note
    if not low_vals or not high_vals:
        return MicroPilotGap(
            experiment_id=experiment_id,
            n_low=len(low_vals),
            n_high=len(high_vals),
            mean_low=None,
            mean_high=None,
            mean_diff=None,
            passes=None,
            note=note or "eksik hücre",
        )
    mean_low = mean(low_vals)
    mean_high = mean(high_vals)
    mean_diff = mean_low - mean_high  # positive = expected direction
    return MicroPilotGap(
        experiment_id=experiment_id,
        n_low=len(low_vals),
        n_high=len(high_vals),
        mean_low=mean_low,
        mean_high=mean_high,
        mean_diff=mean_diff,
        passes=mean_diff > SUCCESS_MEAN_DIFF,
        note="",
    )


def _direction_label(mean_diff: float | None, r: float | None) -> str:
    # Prefer mean_diff sign when available (micro-pilot decision metric)
    if mean_diff is not None:
        if mean_diff > SUCCESS_MEAN_DIFF:
            return "EXPECTED"
        if mean_diff < -SUCCESS_MEAN_DIFF:
            return "INVERSE"
        return "WEAK / NULL"
    if r is None:
        return "n/a"
    if r < -0.2:
        return "EXPECTED"
    if r > 0.2:
        return "INVERSE"
    return "WEAK / NULL"


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    if signed:
        return f"{value:+.3f}"
    return f"{value:.3f}"


def run_comparison(*, max_round: int | None, db_path: str) -> dict:
    print("=" * 78)
    print("PROMPT REVISION COMPARISON — full_experiment_v1 vs prompt_revision_v1")
    print("=" * 78)
    print(f"  database   : {db_path}")
    window = f"round ≤ {max_round}" if max_round is not None else "all rounds"
    print(f"  window     : {window}")
    print(
        f"  micro cell : coop∈{{{MICRO_COOP_LOW}, {MICRO_COOP_HIGH}}}, "
        f"risk={MICRO_RISK}"
    )
    print(
        f"  success    : mean_diff = mean(0.2) − mean(0.8) > {SUCCESS_MEAN_DIFF}"
    )
    print()

    with sqlite3.connect(db_path) as conn:
        corr_by_id: dict[str, CoopCorr] = {}
        gap_by_id: dict[str, MicroPilotGap] = {}

        for experiment_id in (BASELINE_ID, REVISION_ID):
            fidelity = fetch_run_fidelity(conn, experiment_id, max_round=max_round)
            corr = cooperation_extraction_corr(fidelity)
            corr_by_id[experiment_id] = CoopCorr(
                experiment_id=experiment_id,
                n=corr.n,
                r=corr.r,
                p=corr.p,
                note=corr.note or ("no fidelity rows" if not fidelity else ""),
            )
            gap_by_id[experiment_id] = micro_pilot_gap(
                conn, experiment_id, max_round=max_round
            )

    base_gap = gap_by_id[BASELINE_ID]
    rev_gap = gap_by_id[REVISION_ID]
    base_corr = corr_by_id[BASELINE_ID]
    rev_corr = corr_by_id[REVISION_ID]

    base_dir = _direction_label(base_gap.mean_diff, base_corr.r)
    rev_dir = _direction_label(rev_gap.mean_diff, rev_corr.r)
    base_pass = "HAYIR" if base_gap.passes is not True else "EVET"
    rev_pass = (
        "EVET"
        if rev_gap.passes is True
        else ("HAYIR" if rev_gap.passes is False else "n/a")
    )

    # Decision table requested by Phase-1 micro-pilot prompt
    print(
        f"| {'Metric':<28} | {'full_experiment_v1':^22} | "
        f"{'prompt_revision_v1 (micro)':^26} |"
    )
    print(f"|{'-' * 30}|{'-' * 24}|{'-' * 28}|")
    print(
        f"| {'coop=0.2 mean extract':<28} | {_fmt(base_gap.mean_low):^22} | "
        f"{_fmt(rev_gap.mean_low):^26} |"
    )
    print(
        f"| {'coop=0.8 mean extract':<28} | {_fmt(base_gap.mean_high):^22} | "
        f"{_fmt(rev_gap.mean_high):^26} |"
    )
    print(
        f"| {'mean_diff (0.2 - 0.8)':<28} | {_fmt(base_gap.mean_diff, signed=True):^22} | "
        f"{_fmt(rev_gap.mean_diff, signed=True):^26} |"
    )
    print(
        f"| {'Pearson r (coop→extract)':<28} | {_fmt(base_corr.r, signed=True):^22} | "
        f"{_fmt(rev_corr.r, signed=True):^26} |"
    )
    print(
        f"| {'Yön':<28} | {base_dir:^22} | {rev_dir:^26} |"
    )
    print(
        f"| {'Başarı kriteri karşılandı mı?':<28} | {base_pass:^22} | {rev_pass:^26} |"
    )
    print()

    print("Details:")
    print(
        f"  {BASELINE_ID}: n_low={base_gap.n_low}, n_high={base_gap.n_high}, "
        f"Pearson n={base_corr.n}, p={_fmt(base_corr.p)}"
    )
    print(
        f"  {REVISION_ID}: n_low={rev_gap.n_low}, n_high={rev_gap.n_high}, "
        f"Pearson n={rev_corr.n}, p={_fmt(rev_corr.p)}"
        + (f" ({rev_gap.note})" if rev_gap.note else "")
    )
    print()

    if rev_gap.passes is True:
        print(
            f"Micro-pilot SUCCESS: mean_diff={rev_gap.mean_diff:+.3f} > "
            f"{SUCCESS_MEAN_DIFF} (expected direction)."
        )
    elif rev_gap.passes is False:
        print(
            f"Micro-pilot FAIL: mean_diff={rev_gap.mean_diff:+.3f} ≤ "
            f"{SUCCESS_MEAN_DIFF}."
        )
    else:
        print(
            f"Micro-pilot: {REVISION_ID} results not yet in DB "
            f"({rev_gap.note or 'no data'})."
        )

    return {
        "baseline_gap": base_gap,
        "revision_gap": rev_gap,
        "baseline_corr": base_corr,
        "revision_corr": rev_corr,
        "revision_passes": rev_gap.passes,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare full_experiment_v1 vs prompt_revision_v1 fidelity.",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=None,
        help="Optional round window (e.g. 2) to reduce collapse confound",
    )
    parser.add_argument(
        "--db",
        default=RESULTS_DB_PATH,
        help=f"SQLite path (default: {RESULTS_DB_PATH})",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_comparison(max_round=args.max_round, db_path=args.db)


if __name__ == "__main__":
    main()
