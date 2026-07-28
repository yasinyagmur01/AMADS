"""
Analyze bargaining experiments vs CPR locked references.

Usage (repo root):
    python analysis/analyze_bargaining.py --scope micro
    python analysis/analyze_bargaining.py --scope risk-micro
    python analysis/analyze_bargaining.py --scope full
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

from scenarios.bargaining.persistence import BARGAINING_DB_PATH

BARGAINING_ID = "bargaining_v1"
RISK_EXPERIMENT_ID = "bargaining_risk_v1"
CPR_ID = "full_experiment_v1"

MICRO_COOP_LOW = 0.2
MICRO_COOP_HIGH = 0.8
MICRO_RISK_LOW = 0.2
MICRO_RISK_HIGH = 0.8

# Locked full_experiment_v1 coop reference
CPR_COOP_LOCKED = {
    "mean_low": 5.040,
    "mean_high": 5.745,
    "mean_diff": -0.704,
    "r": 0.456,
    "direction": "INVERSE",
}

# Locked full_experiment_v1 risk reference (user-specified)
CPR_RISK_LOCKED = {
    "r": 0.678,
    "direction": "EXPECTED",
    "p": "<0.0001",
}

try:
    from scipy.stats import pearsonr
except ImportError as exc:
    pearsonr = None  # type: ignore[misc, assignment]
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None


@dataclass(frozen=True)
class BargainingRunRow:
    run_id: str
    proposer_coop: float
    proposer_risk: float
    responder_coop: float
    responder_risk: float
    mean_keep_amount: float
    mean_keep_fraction: float
    accept_rate: float
    reject_rate: float
    mean_accepted_keep: float | None
    rounds_played: int


def _require_scipy() -> None:
    if pearsonr is None:
        raise RuntimeError(
            f"scipy gerekli. Kurulum: pip install scipy ({_SCIPY_IMPORT_ERROR})"
        )


def _direction_coop(r: float) -> str:
    """Expected: negative r (high coop → lower keep)."""
    if abs(r) < 0.2:
        return "TRAIT-BLIND / NEGLIGIBLE"
    if r < 0:
        return "EXPECTED (negative r)"
    return "INVERSE (positive r)"


def _direction_risk(r: float) -> str:
    """Expected: positive r (high risk → higher keep)."""
    if abs(r) < 0.2:
        return "TRAIT-BLIND / NEGLIGIBLE"
    if r > 0:
        return "EXPECTED (positive r)"
    return "INVERSE (negative r)"


def fetch_bargaining_runs(
    conn: sqlite3.Connection,
    experiment_id: str,
    *,
    scope: str,
) -> list[BargainingRunRow]:
    risk_filter = ""
    params: list = [experiment_id, experiment_id]
    if scope == "micro":
        risk_filter = """
            AND c.proposer_risk = 0.2
            AND c.proposer_coop IN (0.2, 0.8)
            AND c.responder_coop = 0.5
            AND c.responder_risk = 0.5
        """
    elif scope == "risk-micro":
        risk_filter = """
            AND c.proposer_coop = 0.5
            AND c.proposer_risk IN (0.2, 0.8)
            AND c.responder_coop = 0.5
            AND c.responder_risk = 0.5
        """

    sql = f"""
        SELECT
            c.run_id,
            c.proposer_coop,
            c.proposer_risk,
            c.responder_coop,
            c.responder_risk,
            agg.mean_keep_amount,
            agg.mean_keep_fraction,
            agg.accept_rate,
            agg.reject_rate,
            agg.mean_accepted_keep,
            agg.rounds_played
        FROM bargaining_conditions c
        JOIN (
            SELECT
                m.run_id,
                AVG(m.keep_amount) AS mean_keep_amount,
                AVG(m.keep_fraction) AS mean_keep_fraction,
                AVG(CAST(m.accepted AS REAL)) AS accept_rate,
                AVG(1.0 - CAST(m.accepted AS REAL)) AS reject_rate,
                AVG(CASE WHEN m.accepted = 1 THEN m.keep_amount END)
                    AS mean_accepted_keep,
                COUNT(m.round_number) AS rounds_played
            FROM bargaining_metrics m
            WHERE m.experiment_id = ?
            GROUP BY m.run_id
        ) agg ON c.run_id = agg.run_id
        WHERE c.experiment_id = ?
        {risk_filter}
        ORDER BY c.proposer_coop, c.proposer_risk, c.replication
    """
    rows = conn.execute(sql, params).fetchall()
    return [
        BargainingRunRow(
            run_id=r[0],
            proposer_coop=r[1],
            proposer_risk=r[2],
            responder_coop=r[3],
            responder_risk=r[4],
            mean_keep_amount=r[5],
            mean_keep_fraction=r[6],
            accept_rate=r[7],
            reject_rate=r[8],
            mean_accepted_keep=r[9],
            rounds_played=r[10],
        )
        for r in rows
    ]


def integrity_check(conn: sqlite3.Connection, experiment_id: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT keep_amount, offer_to_responder
        FROM bargaining_rounds
        WHERE experiment_id = ?
        """,
        (experiment_id,),
    ).fetchall()
    null_keep = sum(1 for k, _ in rows if k is None)
    out_of_range = sum(
        1 for k, _ in rows if k is not None and not (0.0 <= float(k) <= 100.0)
    )
    pie_mismatch = sum(
        1
        for k, o in rows
        if k is not None and o is not None and abs(float(k) + float(o) - 100.0) > 1e-6
    )
    return {
        "n_rounds": len(rows),
        "keep_null": null_keep,
        "keep_out_of_range": out_of_range,
        "pie_mismatch": pie_mismatch,
    }


def _print_integrity(checks: dict[str, int]) -> None:
    print("=" * 72)
    print("INTEGRITY CHECK (keep_amount)")
    print("=" * 72)
    print(f"  rounds checked              : {checks['n_rounds']}")
    print(f"  keep_amount null            : {checks['keep_null']}")
    print(f"  keep_amount outside [0,100] : {checks['keep_out_of_range']}")
    print(f"  keep + offer ≠ 100          : {checks['pie_mismatch']}")
    ok = (
        checks["keep_null"] == 0
        and checks["keep_out_of_range"] == 0
        and checks["pie_mismatch"] == 0
    )
    print(f"  RESULT                      : {'OK' if ok else 'FAIL'}")
    print()


def run_coop_micro_analysis(rows: list[BargainingRunRow]) -> None:
    keep_by_coop = {
        MICRO_COOP_LOW: [
            r.mean_keep_amount
            for r in rows
            if abs(r.proposer_coop - MICRO_COOP_LOW) < 1e-9
        ],
        MICRO_COOP_HIGH: [
            r.mean_keep_amount
            for r in rows
            if abs(r.proposer_coop - MICRO_COOP_HIGH) < 1e-9
        ],
    }
    mean_low = mean(keep_by_coop[MICRO_COOP_LOW]) if keep_by_coop[MICRO_COOP_LOW] else None
    mean_high = (
        mean(keep_by_coop[MICRO_COOP_HIGH]) if keep_by_coop[MICRO_COOP_HIGH] else None
    )
    mean_diff = (
        mean_low - mean_high
        if mean_low is not None and mean_high is not None
        else None
    )

    coop_x = [r.proposer_coop for r in rows]
    keep_y = [r.mean_keep_amount for r in rows]
    _require_scipy()
    r_val, p_val = pearsonr(coop_x, keep_y)
    direction = _direction_coop(float(r_val))

    print("| Metric                              | full_experiment_v1 (CPR) | bargaining_v1 (micro) |")
    print("|--------------------------------------|---------------------------|------------------------|")
    cpr = CPR_COOP_LOCKED
    barg_l = f"{mean_low:.3f}" if mean_low is not None else "—"
    barg_h = f"{mean_high:.3f}" if mean_high is not None else "—"
    barg_d = f"{mean_diff:.3f}" if mean_diff is not None else "—"
    barg_r = f"{float(r_val):+.3f}"
    print(
        f"| coop=0.2 mean (kendine ayırdığı pay) | {cpr['mean_low']:.3f} (extraction)        | {barg_l:<22} |"
    )
    print(
        f"| coop=0.8 mean (kendine ayırdığı pay) | {cpr['mean_high']:.3f} (extraction)        | {barg_h:<22} |"
    )
    print(
        f"| mean_diff (0.2 − 0.8)                | {cpr['mean_diff']:+.3f}                    | {barg_d:<22} |"
    )
    print(
        f"| Pearson r (coop→kendine_pay)         | {cpr['r']:+.3f}                    | {barg_r:<22} |"
    )
    print(
        f"| Yön                                   | {cpr['direction']:<25} | {direction:<22} |"
    )
    print()
    print(f"Pearson r = {float(r_val):+.4f},  p = {float(p_val):.4g},  n = {len(rows)}")
    print(
        f"Anlamlılık (α=0.05): "
        f"{'evet' if float(p_val) < 0.05 else 'hayır'} "
        f"(p={'<' if float(p_val) < 0.05 else '≥'} 0.05)"
    )
    print()

    print("Run-level mean keep_amount:")
    print(f"  {'run_id':<48} {'coop':>5} {'keep_X':>8} {'accept':>7} {'reject':>7}")
    print("  " + "-" * 80)
    for row in rows:
        print(
            f"  {row.run_id:<48} {row.proposer_coop:5.1f} "
            f"{row.mean_keep_amount:8.3f} {row.accept_rate:7.3f} "
            f"{row.reject_rate:7.3f}"
        )
    print()


def run_risk_micro_analysis(rows: list[BargainingRunRow]) -> None:
    keep_by_risk = {
        MICRO_RISK_LOW: [
            r.mean_keep_amount
            for r in rows
            if abs(r.proposer_risk - MICRO_RISK_LOW) < 1e-9
        ],
        MICRO_RISK_HIGH: [
            r.mean_keep_amount
            for r in rows
            if abs(r.proposer_risk - MICRO_RISK_HIGH) < 1e-9
        ],
    }
    mean_low = mean(keep_by_risk[MICRO_RISK_LOW]) if keep_by_risk[MICRO_RISK_LOW] else None
    mean_high = (
        mean(keep_by_risk[MICRO_RISK_HIGH]) if keep_by_risk[MICRO_RISK_HIGH] else None
    )
    # User-specified: mean_diff = 0.8 − 0.2
    mean_diff = (
        mean_high - mean_low
        if mean_low is not None and mean_high is not None
        else None
    )

    risk_x = [r.proposer_risk for r in rows]
    keep_y = [r.mean_keep_amount for r in rows]
    _require_scipy()
    r_val, p_val = pearsonr(risk_x, keep_y)
    direction = _direction_risk(float(r_val))

    print(
        "| Metric                         | full_experiment_v1 (CPR, risk) | bargaining_risk_v1 (micro) |"
    )
    print(
        "|----------------------------------|-----------------------------------|-------------------------------|"
    )
    cpr = CPR_RISK_LOCKED
    barg_l = f"{mean_low:.3f}" if mean_low is not None else "—"
    barg_h = f"{mean_high:.3f}" if mean_high is not None else "—"
    barg_d = f"{mean_diff:.3f}" if mean_diff is not None else "—"
    barg_r = f"{float(r_val):+.3f}"
    barg_p = f"{float(p_val):.4g}"
    print(
        f"| risk=0.2 mean (kendine ayırdığı pay) | —                              | {barg_l:<29} |"
    )
    print(
        f"| risk=0.8 mean (kendine ayırdığı pay) | —                              | {barg_h:<29} |"
    )
    print(
        f"| mean_diff (0.8 − 0.2)            | —                                  | {barg_d:<29} |"
    )
    print(
        f"| Pearson r (risk→kendine_pay)     | {cpr['r']:+.3f} (CPR)                      | {barg_r:<29} |"
    )
    print(
        f"| Yön                               | {cpr['direction']:<33} | {direction:<29} |"
    )
    print(
        f"| p-value                          | {cpr['p']:<33} | {barg_p:<29} |"
    )
    print()
    print(f"Pearson r = {float(r_val):+.4f},  p = {float(p_val):.4g},  n = {len(rows)}")
    print(
        f"Anlamlılık (α=0.05): "
        f"{'evet' if float(p_val) < 0.05 else 'hayır'} "
        f"(p={'<' if float(p_val) < 0.05 else '≥'} 0.05)"
    )
    print()

    print("Run-level mean keep_amount:")
    print(f"  {'run_id':<48} {'risk':>5} {'keep_X':>8} {'accept':>7} {'reject':>7}")
    print("  " + "-" * 80)
    for row in rows:
        print(
            f"  {row.run_id:<48} {row.proposer_risk:5.1f} "
            f"{row.mean_keep_amount:8.3f} {row.accept_rate:7.3f} "
            f"{row.reject_rate:7.3f}"
        )
    print()

    # Cell reject rates
    print("Responder reject_rate by proposer_risk cell:")
    for risk in (MICRO_RISK_LOW, MICRO_RISK_HIGH):
        members = [r for r in rows if abs(r.proposer_risk - risk) < 1e-9]
        if not members:
            continue
        print(
            f"  risk={risk:.1f}: n={len(members)}, "
            f"mean_keep={mean(m.mean_keep_amount for m in members):.3f}, "
            f"reject_rate={mean(m.reject_rate for m in members):.3f}, "
            f"accept_rate={mean(m.accept_rate for m in members):.3f}"
        )
    print()


def run_analysis(
    *,
    bargaining_db: str,
    scope: str,
    experiment_id: str | None = None,
) -> None:
    if experiment_id:
        eid = experiment_id
    elif scope == "risk-micro":
        eid = RISK_EXPERIMENT_ID
    else:
        eid = BARGAINING_ID

    path = Path(bargaining_db)
    print("=" * 72)
    print(f"BARGAINING ANALYSIS (scope={scope})")
    print("=" * 72)
    print(f"  bargaining experiment : {eid}")
    print(f"  CPR reference         : {CPR_ID} (locked constants)")
    print(f"  database              : {bargaining_db}")
    print()

    if not path.exists():
        print(f"Error: bargaining DB missing: {path}", file=sys.stderr)
        raise SystemExit(1)

    with sqlite3.connect(path) as conn:
        rows = fetch_bargaining_runs(conn, eid, scope=scope)
        checks = integrity_check(conn, eid)

    if not rows:
        print(
            f"Error: no metrics for '{eid}' — run may not have completed.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"  bargaining runs       : {len(rows)}")
    print(f"  rounds/run (mean)     : {mean(r.rounds_played for r in rows):.1f}")
    print()

    _print_integrity(checks)

    if scope == "risk-micro":
        run_risk_micro_analysis(rows)
    elif scope in ("micro", "full"):
        run_coop_micro_analysis(rows)
    else:
        raise SystemExit(f"Unknown scope: {scope}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bargaining vs CPR trait fidelity comparison.",
    )
    parser.add_argument(
        "--scope",
        choices=("micro", "risk-micro", "full"),
        default="micro",
        help="micro | risk-micro | full",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Override experiment_id (default: bargaining_v1 / bargaining_risk_v1)",
    )
    parser.add_argument(
        "--bargaining-db",
        default=BARGAINING_DB_PATH,
        help=f"Bargaining SQLite (default: {BARGAINING_DB_PATH})",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_analysis(
        bargaining_db=args.bargaining_db,
        scope=args.scope,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
