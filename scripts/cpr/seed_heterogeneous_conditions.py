"""
Seed heterogeneous_v1 conditions + per-condition composition JSON.

Writes:
  - experiment_conditions: 4 conditions × 5 reps = 20 rows
  - heterogeneous_compositions: 4 rows (composition JSON per condition)

Does not modify full_experiment_v1 / prompt_revision_v1 rows.

Usage (repo root):
    python scripts/cpr/seed_heterogeneous_conditions.py
    python scripts/cpr/seed_heterogeneous_conditions.py --dry-run
    python scripts/cpr/seed_heterogeneous_conditions.py --check
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.database import RESULTS_DB_PATH, register_experiment_conditions
from experiments.cpr.heterogeneous_conditions import (
    CONDITION_ORDER,
    CONDITIONS,
    EXPERIMENT_ID,
    REPLICATIONS,
    ensure_composition_table,
    experiment_condition_rows,
    upsert_compositions,
)


def check_seeded(db_path: str) -> tuple[bool, dict[str, int], list[str]]:
    """Return (ok, counts, missing_condition_ids)."""
    expected_runs = len(CONDITION_ORDER) * REPLICATIONS
    path = Path(db_path)
    if not path.exists():
        return False, {"conditions": 0, "compositions": 0}, list(CONDITION_ORDER)

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        cond_count = 0
        if "experiment_conditions" in tables:
            cond_count = conn.execute(
                "SELECT COUNT(*) FROM experiment_conditions WHERE experiment_id = ?",
                (EXPERIMENT_ID,),
            ).fetchone()[0]

        comp_count = 0
        present_ids: set[str] = set()
        if "heterogeneous_compositions" in tables:
            comp_count = conn.execute(
                "SELECT COUNT(*) FROM heterogeneous_compositions WHERE experiment_id = ?",
                (EXPERIMENT_ID,),
            ).fetchone()[0]
            present_ids = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT condition_id FROM heterogeneous_compositions
                    WHERE experiment_id = ?
                    """,
                    (EXPERIMENT_ID,),
                ).fetchall()
            }

    missing = [cid for cid in CONDITION_ORDER if cid not in present_ids]
    ok = cond_count == expected_runs and comp_count == len(CONDITION_ORDER) and not missing
    return ok, {"conditions": cond_count, "compositions": comp_count}, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Seed experiment_conditions + compositions for {EXPERIMENT_ID}."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned rows without writing to the database",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify 20 condition rows + 4 composition JSON rows exist",
    )
    parser.add_argument(
        "--db",
        default=RESULTS_DB_PATH,
        help=f"SQLite path (default: {RESULTS_DB_PATH})",
    )
    args = parser.parse_args()

    if args.check:
        ok, counts, missing = check_seeded(args.db)
        print(f"experiment_id : {EXPERIMENT_ID}")
        print(f"database      : {args.db}")
        print(f"conditions    : {counts['conditions']} / {len(CONDITION_ORDER) * REPLICATIONS}")
        print(f"compositions  : {counts['compositions']} / {len(CONDITION_ORDER)}")
        if missing:
            print(f"missing comps : {', '.join(missing)}")
            print("\nRESULT: FAIL — seed required")
            raise SystemExit(1)
        if not ok:
            print("\nRESULT: FAIL — unexpected row counts")
            raise SystemExit(1)
        print("\nRESULT: OK — heterogeneous conditions present")
        raise SystemExit(0)

    rows = experiment_condition_rows()
    print(f"experiment_id : {EXPERIMENT_ID}")
    print(f"database      : {args.db}")
    print(
        f"rows to seed  : {len(rows)} experiment_conditions "
        f"({len(CONDITION_ORDER)} × {REPLICATIONS}) + "
        f"{len(CONDITION_ORDER)} compositions"
    )

    if args.dry_run:
        print("\n--dry-run: no DB write.")
        print("\nCompositions:")
        for cid in CONDITION_ORDER:
            cond = CONDITIONS[cid]
            print(f"  {cond.label} ({cond.name})")
            print(f"    {cond.composition_json()}")
        print("\nexperiment_conditions rows:")
        for run_id, coop_level, risk_level, coop_val, risk_val, rep in rows:
            print(
                f"  {run_id:<20} condition={coop_level:<4} "
                f"risk_level={risk_level:<8} "
                f"mean_coop={coop_val:.3f} mean_risk={risk_val:.3f} rep={rep}"
            )
        return

    register_experiment_conditions(EXPERIMENT_ID, rows, args.db)

    with sqlite3.connect(args.db) as conn:
        ensure_composition_table(conn)
        n_comp = upsert_compositions(conn, EXPERIMENT_ID)
        conn.commit()
        cond_count = conn.execute(
            "SELECT COUNT(*) FROM experiment_conditions WHERE experiment_id = ?",
            (EXPERIMENT_ID,),
        ).fetchone()[0]
        full_count = conn.execute(
            "SELECT COUNT(*) FROM experiment_conditions WHERE experiment_id = ?",
            ("full_experiment_v1",),
        ).fetchone()[0]

    print(f"\nWrote/upserted {EXPERIMENT_ID} conditions : {cond_count} rows")
    print(f"Wrote/upserted compositions             : {n_comp} rows")
    print(f"Unchanged full_experiment_v1 conditions : {full_count} rows")
    expected = len(CONDITION_ORDER) * REPLICATIONS
    if cond_count != expected:
        raise SystemExit(f"Expected {expected} condition rows, got {cond_count}")


if __name__ == "__main__":
    main()
