"""
One-shot migration: move legacy experiment_ids from results.db → archive_results.db.

Archive IDs:
  mock_dev_collapse, mock_dev_complete, ratio_fine_tune, ratio_revalidation,
  ratio_validation, single_agent_test, full_real_test

Usage (repo root):
    python scripts/migrate_archive_results.py
    python scripts/migrate_archive_results.py --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
SRC_DB = _ROOT / "data" / "results.db"
DST_DB = _ROOT / "data" / "archive_results.db"

ARCHIVE_IDS = (
    "mock_dev_collapse",
    "mock_dev_complete",
    "ratio_fine_tune",
    "ratio_revalidation",
    "ratio_validation",
    "single_agent_test",
    "full_real_test",
)

# Tables that may contain experiment_id rows to move.
TABLES = (
    "agent_decisions",
    "metrics_snapshots",
    "experiment_conditions",
    "experiment_conditions_old",
    "heterogeneous_compositions",
)


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _count_by_id(conn: sqlite3.Connection, table: str) -> dict[str, int]:
    if not _table_exists(conn, table):
        return {}
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if "experiment_id" not in cols:
        return {}
    rows = conn.execute(
        f"""
        SELECT experiment_id, COUNT(*)
        FROM {table}
        WHERE experiment_id IN ({_placeholders(len(ARCHIVE_IDS))})
        GROUP BY experiment_id
        """,
        ARCHIVE_IDS,
    ).fetchall()
    return {eid: int(n) for eid, n in rows}


def _snapshot(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    return {t: _count_by_id(conn, t) for t in TABLES}


def _ensure_schema(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    for table in TABLES:
        if not _table_exists(src, table):
            continue
        create_sql = src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if create_sql and create_sql[0]:
            dst.execute(create_sql[0])
    dst.commit()


def _copy_rows(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    table: str,
) -> int:
    if not _table_exists(src, table):
        return 0
    cols = [r[1] for r in src.execute(f"PRAGMA table_info({table})")]
    if "experiment_id" not in cols:
        return 0
    col_list = ", ".join(cols)
    # Exclude autoincrement id so destination gets fresh PKs and avoids collisions.
    insert_cols = [c for c in cols if c != "id"]
    select_cols = ", ".join(insert_cols)
    insert_list = ", ".join(insert_cols)
    ph = _placeholders(len(insert_cols))
    rows = src.execute(
        f"""
        SELECT {select_cols}
        FROM {table}
        WHERE experiment_id IN ({_placeholders(len(ARCHIVE_IDS))})
        """,
        ARCHIVE_IDS,
    ).fetchall()
    if not rows:
        return 0
    dst.executemany(
        f"INSERT INTO {table} ({insert_list}) VALUES ({ph})",
        rows,
    )
    return len(rows)


def _delete_rows(src: sqlite3.Connection, table: str) -> int:
    if not _table_exists(src, table):
        return 0
    cols = {r[1] for r in src.execute(f"PRAGMA table_info({table})")}
    if "experiment_id" not in cols:
        return 0
    cur = src.execute(
        f"""
        DELETE FROM {table}
        WHERE experiment_id IN ({_placeholders(len(ARCHIVE_IDS))})
        """,
        ARCHIVE_IDS,
    )
    return cur.rowcount


def migrate(*, dry_run: bool) -> int:
    if not SRC_DB.exists():
        print(f"ERROR: source DB missing: {SRC_DB}", file=sys.stderr)
        return 1

    src = sqlite3.connect(SRC_DB)
    before = _snapshot(src)
    total_before = sum(sum(v.values()) for v in before.values())
    print(f"Source: {SRC_DB}")
    print(f"Dest  : {DST_DB}")
    print(f"Archive IDs: {', '.join(ARCHIVE_IDS)}")
    print("\n--- BEFORE (source counts for archive IDs) ---")
    for table, counts in before.items():
        print(f"  {table}: {counts if counts else '{}'}")
    print(f"  TOTAL rows to move: {total_before}")

    if dry_run:
        print("\nDry-run only — no writes.")
        src.close()
        return 0

    if DST_DB.exists():
        print(f"\nERROR: destination already exists: {DST_DB}", file=sys.stderr)
        print("Delete or rename it before re-running.", file=sys.stderr)
        src.close()
        return 1

    dst = sqlite3.connect(DST_DB)
    try:
        _ensure_schema(src, dst)
        copied: dict[str, int] = {}
        for table in TABLES:
            n = _copy_rows(src, dst, table)
            copied[table] = n
        dst.commit()

        deleted: dict[str, int] = {}
        for table in TABLES:
            deleted[table] = _delete_rows(src, table)
        src.commit()

        after_src = _snapshot(src)
        after_dst = _snapshot(dst)

        print("\n--- COPIED → archive_results.db ---")
        for table, n in copied.items():
            print(f"  {table}: {n}")
        print("\n--- DELETED from results.db ---")
        for table, n in deleted.items():
            print(f"  {table}: {n}")
        print("\n--- AFTER source (should be empty for archive IDs) ---")
        for table, counts in after_src.items():
            print(f"  {table}: {counts if counts else '{}'}")
        print("\n--- AFTER dest ---")
        for table, counts in after_dst.items():
            print(f"  {table}: {counts if counts else '{}'}")

        # Verify: copied == deleted == before; source residual 0; dest == before
        ok = True
        for table in TABLES:
            b = before[table]
            d = after_dst[table]
            s = after_src[table]
            if s:
                print(f"FAIL: source still has {table}={s}", file=sys.stderr)
                ok = False
            if b != d:
                print(
                    f"FAIL: {table} before={b} != dest={d}",
                    file=sys.stderr,
                )
                ok = False
            if copied[table] != deleted[table]:
                print(
                    f"FAIL: {table} copied={copied[table]} != deleted={deleted[table]}",
                    file=sys.stderr,
                )
                ok = False
            if copied[table] != sum(b.values()):
                print(
                    f"FAIL: {table} copied={copied[table]} != sum(before)={sum(b.values())}",
                    file=sys.stderr,
                )
                ok = False

        # Remaining active experiments still in source
        remaining = src.execute(
            "SELECT DISTINCT experiment_id FROM agent_decisions ORDER BY 1"
        ).fetchall()
        print("\n--- Remaining experiment_ids in results.db ---")
        for (eid,) in remaining:
            print(f"  - {eid}")

        if ok:
            print("\nOK: migration verified — row counts match.")
            return 0
        print("\nFAIL: verification errors (see above).", file=sys.stderr)
        return 1
    finally:
        dst.close()
        src.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print source counts only; do not write.",
    )
    args = parser.parse_args()
    raise SystemExit(migrate(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
