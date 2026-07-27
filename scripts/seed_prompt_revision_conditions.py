"""
Seed experiment_conditions rows for prompt_revision_v1.

Same 3×3 × 5 replication grid as full_experiment_v1; only experiment_id differs.
Does not modify existing full_experiment_v1 rows (composite UNIQUE on
experiment_id + run_id).

Usage (repo root):
    python scripts/seed_prompt_revision_conditions.py
    python scripts/seed_prompt_revision_conditions.py --dry-run
    python scripts/seed_prompt_revision_conditions.py --check
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

from core.database import RESULTS_DB_PATH, register_experiment_conditions

EXPERIMENT_ID = "prompt_revision_v1"
REPLICATIONS = 5
TRAIT_LEVELS: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
}

# Micro-pilot cells: coop ∈ {0.2, 0.8}, risk = 0.2 → 10 runs
MICRO_COOP_LEVELS = ("low", "high")
MICRO_RISK_LEVEL = "low"


def build_condition_rows() -> list[tuple[str, str, str, float, float, int]]:
    rows: list[tuple[str, str, str, float, float, int]] = []
    for coop_level, risk_level in product(TRAIT_LEVELS, TRAIT_LEVELS):
        coop_value = TRAIT_LEVELS[coop_level]
        risk_value = TRAIT_LEVELS[risk_level]
        for rep in range(1, REPLICATIONS + 1):
            run_id = f"cond_{coop_level}_{risk_level}_rep{rep}"
            rows.append(
                (run_id, coop_level, risk_level, coop_value, risk_value, rep)
            )
    return rows


def micro_pilot_run_ids() -> list[str]:
    ids: list[str] = []
    for coop_level in MICRO_COOP_LEVELS:
        for rep in range(1, REPLICATIONS + 1):
            ids.append(f"cond_{coop_level}_{MICRO_RISK_LEVEL}_rep{rep}")
    return ids


def check_micro_pilot_conditions(db_path: str) -> tuple[bool, list[str], list[str]]:
    """Return (ok, present_run_ids, missing_run_ids) for the 10 micro-pilot cells."""
    expected = micro_pilot_run_ids()
    path = Path(db_path)
    if not path.exists():
        return False, [], expected

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "experiment_conditions" not in tables:
            return False, [], expected

        rows = conn.execute(
            """
            SELECT run_id FROM experiment_conditions
            WHERE experiment_id = ?
              AND coop_value IN (0.2, 0.8)
              AND risk_value = 0.2
            ORDER BY run_id
            """,
            (EXPERIMENT_ID,),
        ).fetchall()

    present = [row[0] for row in rows]
    present_set = set(present)
    missing = [run_id for run_id in expected if run_id not in present_set]
    ok = len(missing) == 0 and len(present_set & set(expected)) == len(expected)
    return ok, present, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Seed experiment_conditions for {EXPERIMENT_ID}."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned rows without writing to the database",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify micro-pilot 10 conditions exist "
            "(coop∈{0.2,0.8}, risk=0.2, 5 rep/cell); exit 1 if missing"
        ),
    )
    parser.add_argument(
        "--db",
        default=RESULTS_DB_PATH,
        help=f"SQLite path (default: {RESULTS_DB_PATH})",
    )
    args = parser.parse_args()

    if args.check:
        ok, present, missing = check_micro_pilot_conditions(args.db)
        print(f"experiment_id : {EXPERIMENT_ID}")
        print(f"database      : {args.db}")
        print("check         : micro-pilot conditions (coop∈{0.2,0.8}, risk=0.2)")
        print(f"present       : {len(present)} / 10")
        if missing:
            print(f"missing       : {len(missing)}")
            for run_id in missing:
                print(f"  - {run_id}")
            print("\nRESULT: FAIL — seed required")
            raise SystemExit(1)
        print("\nRESULT: OK — micro-pilot conditions present")
        raise SystemExit(0)

    rows = build_condition_rows()
    print(f"experiment_id : {EXPERIMENT_ID}")
    print(f"database      : {args.db}")
    print(f"rows to seed  : {len(rows)} (9 conditions × {REPLICATIONS} reps)")

    if args.dry_run:
        print("\n--dry-run: no DB write.")
        for run_id, coop_level, risk_level, coop_val, risk_val, rep in rows:
            print(
                f"  {run_id:<32} coop={coop_val:.1f}/{coop_level:<6} "
                f"risk={risk_val:.1f}/{risk_level:<6} rep={rep}"
            )
        return

    register_experiment_conditions(EXPERIMENT_ID, rows, args.db)

    with sqlite3.connect(args.db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM experiment_conditions WHERE experiment_id = ?",
            (EXPERIMENT_ID,),
        ).fetchone()[0]
        full_count = conn.execute(
            "SELECT COUNT(*) FROM experiment_conditions WHERE experiment_id = ?",
            ("full_experiment_v1",),
        ).fetchone()[0]

    print(f"\nWrote/upserted {EXPERIMENT_ID}: {count} rows")
    print(f"Unchanged full_experiment_v1 conditions: {full_count} rows")
    if count != 45:
        raise SystemExit(f"Expected 45 rows for {EXPERIMENT_ID}, got {count}")


if __name__ == "__main__":
    main()
