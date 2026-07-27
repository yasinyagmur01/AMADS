"""
Seed bargaining_conditions for bargaining_v1 / bargaining_risk_v1.

Usage (repo root):
    python scripts/seed_bargaining_conditions.py --micro-only
    python scripts/seed_bargaining_conditions.py --risk-micro
    python scripts/seed_bargaining_conditions.py --risk-micro --check
    python scripts/seed_bargaining_conditions.py --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from itertools import product
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scenarios.bargaining.persistence import (
    BARGAINING_DB_PATH,
    init_bargaining_db,
    register_bargaining_conditions,
)

EXPERIMENT_ID = "bargaining_v1"
RISK_EXPERIMENT_ID = "bargaining_risk_v1"
REPLICATIONS = 5

COOP_LEVELS: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
}
PROPOSER_RISK_LEVELS: dict[str, float] = {
    "low": 0.2,
    "high": 0.8,
}

MICRO_PROPOSER_COOP = ("low", "high")
MICRO_PROPOSER_RISK = "low"
FIXED_RESPONDER_COOP = "medium"
FIXED_RESPONDER_RISK_LEVEL = "medium"
FIXED_RESPONDER_RISK_VALUE = 0.5

# Risk micro-pilot: coop fixed 0.5, risk ∈ {0.2, 0.8}
RISK_MICRO_PROPOSER_COOP = "medium"
RISK_MICRO_PROPOSER_RISK = ("low", "high")


def _run_id(p_coop: str, p_risk: str, r_coop: str, rep: int) -> str:
    return f"pcoop_{p_coop}_prisk_{p_risk}_rcoop_{r_coop}_rep{rep}"


def build_condition_rows(*, micro_only: bool = False) -> list[tuple]:
    if micro_only:
        p_coop_levels = MICRO_PROPOSER_COOP
        p_risk_levels = (MICRO_PROPOSER_RISK,)
        r_coop_levels = (FIXED_RESPONDER_COOP,)
    else:
        p_coop_levels = tuple(COOP_LEVELS)
        p_risk_levels = tuple(PROPOSER_RISK_LEVELS)
        r_coop_levels = tuple(COOP_LEVELS)

    rows: list[tuple] = []
    for p_coop, p_risk, r_coop in product(p_coop_levels, p_risk_levels, r_coop_levels):
        for rep in range(1, REPLICATIONS + 1):
            rows.append(
                (
                    _run_id(p_coop, p_risk, r_coop, rep),
                    p_coop,
                    p_risk,
                    r_coop,
                    FIXED_RESPONDER_RISK_LEVEL,
                    COOP_LEVELS[p_coop],
                    PROPOSER_RISK_LEVELS[p_risk],
                    COOP_LEVELS[r_coop],
                    FIXED_RESPONDER_RISK_VALUE,
                    rep,
                )
            )
    return rows


def build_risk_micro_rows() -> list[tuple]:
    """proposer_coop=0.5 fixed, proposer_risk∈{0.2,0.8}, responder=(0.5,0.5)."""
    rows: list[tuple] = []
    p_coop = RISK_MICRO_PROPOSER_COOP
    for p_risk in RISK_MICRO_PROPOSER_RISK:
        for rep in range(1, REPLICATIONS + 1):
            rows.append(
                (
                    _run_id(p_coop, p_risk, FIXED_RESPONDER_COOP, rep),
                    p_coop,
                    p_risk,
                    FIXED_RESPONDER_COOP,
                    FIXED_RESPONDER_RISK_LEVEL,
                    COOP_LEVELS[p_coop],
                    PROPOSER_RISK_LEVELS[p_risk],
                    COOP_LEVELS[FIXED_RESPONDER_COOP],
                    FIXED_RESPONDER_RISK_VALUE,
                    rep,
                )
            )
    return rows


def micro_pilot_run_ids() -> list[str]:
    ids: list[str] = []
    for p_coop in MICRO_PROPOSER_COOP:
        for rep in range(1, REPLICATIONS + 1):
            ids.append(
                _run_id(p_coop, MICRO_PROPOSER_RISK, FIXED_RESPONDER_COOP, rep)
            )
    return ids


def risk_micro_run_ids() -> list[str]:
    ids: list[str] = []
    for p_risk in RISK_MICRO_PROPOSER_RISK:
        for rep in range(1, REPLICATIONS + 1):
            ids.append(
                _run_id(
                    RISK_MICRO_PROPOSER_COOP, p_risk, FIXED_RESPONDER_COOP, rep
                )
            )
    return ids


def check_micro_pilot_conditions(db_path: str) -> tuple[bool, list[str], list[str]]:
    expected = micro_pilot_run_ids()
    path = Path(db_path)
    if not path.exists():
        return False, [], expected

    init_bargaining_db(db_path)
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "bargaining_conditions" not in tables:
            return False, [], expected

        rows = conn.execute(
            """
            SELECT run_id FROM bargaining_conditions
            WHERE experiment_id = ?
              AND proposer_coop IN (0.2, 0.8)
              AND proposer_risk = 0.2
              AND responder_coop = 0.5
              AND responder_risk = 0.5
            ORDER BY run_id
            """,
            (EXPERIMENT_ID,),
        ).fetchall()

    present = [row[0] for row in rows]
    present_set = set(present)
    missing = [run_id for run_id in expected if run_id not in present_set]
    ok = len(missing) == 0 and len(present_set & set(expected)) == len(expected)
    return ok, present, missing


def check_risk_micro_conditions(db_path: str) -> tuple[bool, list[str], list[str]]:
    expected = risk_micro_run_ids()
    path = Path(db_path)
    if not path.exists():
        return False, [], expected

    init_bargaining_db(db_path)
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "bargaining_conditions" not in tables:
            return False, [], expected

        rows = conn.execute(
            """
            SELECT run_id FROM bargaining_conditions
            WHERE experiment_id = ?
              AND proposer_coop = 0.5
              AND proposer_risk IN (0.2, 0.8)
              AND responder_coop = 0.5
              AND responder_risk = 0.5
            ORDER BY run_id
            """,
            (RISK_EXPERIMENT_ID,),
        ).fetchall()

    present = [row[0] for row in rows]
    present_set = set(present)
    missing = [run_id for run_id in expected if run_id not in present_set]
    ok = len(missing) == 0 and len(present_set & set(expected)) == len(expected)
    return ok, present, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed bargaining_conditions (coop micro / risk micro / full)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned rows without writing to the database",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify seeded conditions exist; exit 1 if missing",
    )
    parser.add_argument(
        "--micro-only",
        action="store_true",
        help="Seed only the 10 coop micro-pilot rows (bargaining_v1)",
    )
    parser.add_argument(
        "--risk-micro",
        action="store_true",
        help=(
            "Seed risk micro-pilot (bargaining_risk_v1): "
            "p_coop=0.5, p_risk∈{0.2,0.8}, responder=(0.5,0.5), 10 rows"
        ),
    )
    parser.add_argument(
        "--db",
        default=BARGAINING_DB_PATH,
        help=f"SQLite path (default: {BARGAINING_DB_PATH})",
    )
    args = parser.parse_args()

    if args.micro_only and args.risk_micro:
        raise SystemExit("Use either --micro-only or --risk-micro, not both.")

    if args.check:
        if args.risk_micro:
            ok, present, missing = check_risk_micro_conditions(args.db)
            exp_id = RISK_EXPERIMENT_ID
            label = "risk-micro (p_coop=0.5, p_risk∈{0.2,0.8}, responder=0.5)"
        else:
            ok, present, missing = check_micro_pilot_conditions(args.db)
            exp_id = EXPERIMENT_ID
            label = "micro-pilot (p_coop∈{0.2,0.8}, p_risk=0.2, r=0.5)"
        print(f"experiment_id : {exp_id}")
        print(f"database      : {args.db}")
        print(f"check         : {label}")
        print(f"present       : {len(present)} / 10")
        if missing:
            print(f"missing       : {len(missing)}")
            for run_id in missing:
                print(f"  - {run_id}")
            print("\nRESULT: FAIL — seed required")
            raise SystemExit(1)
        print("\nRESULT: OK — conditions present")
        raise SystemExit(0)

    if args.risk_micro:
        rows = build_risk_micro_rows()
        experiment_id = RISK_EXPERIMENT_ID
        mode_label = "risk-micro"
        expected = 10
    else:
        rows = build_condition_rows(micro_only=args.micro_only)
        experiment_id = EXPERIMENT_ID
        mode_label = "micro-only" if args.micro_only else "full grid"
        expected = 10 if args.micro_only else 90

    n_cells = len(rows) // REPLICATIONS
    print(f"experiment_id : {experiment_id}")
    print(f"database      : {args.db}")
    print(
        f"rows to seed  : {len(rows)} "
        f"({n_cells} conditions × {REPLICATIONS} reps, {mode_label})"
    )

    if args.dry_run:
        print("\n--dry-run: no DB write.")
        for (
            run_id,
            p_coop_lvl,
            p_risk_lvl,
            r_coop_lvl,
            _r_risk_lvl,
            p_coop,
            p_risk,
            r_coop,
            r_risk,
            rep,
        ) in rows:
            print(
                f"  {run_id:<48} "
                f"p_coop={p_coop:.1f}/{p_coop_lvl:<6} "
                f"p_risk={p_risk:.1f}/{p_risk_lvl:<6} "
                f"r_coop={r_coop:.1f}/{r_coop_lvl:<6} "
                f"r_risk={r_risk:.1f} rep={rep}"
            )
        return

    register_bargaining_conditions(experiment_id, rows, args.db)

    with sqlite3.connect(args.db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM bargaining_conditions WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()[0]

    print(f"\nWrote/upserted {experiment_id}: {count} rows in {args.db}")
    print("CPR data/results.db untouched.")
    if count < expected:
        raise SystemExit(f"Expected >= {expected} rows for {experiment_id}, got {count}")


if __name__ == "__main__":
    main()
