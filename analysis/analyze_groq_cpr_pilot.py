"""Integrity + fidelity summary for full_experiment_groq_v1 micro-pilot."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.trait_fidelity import fetch_run_fidelity
from core.database import RESULTS_DB_PATH
from scipy.stats import pearsonr

EXPERIMENT_ID = "full_experiment_groq_v1"


def integrity_check(conn: sqlite3.Connection, experiment_id: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT extraction_amount, declared_max, justification
        FROM agent_decisions
        WHERE experiment_id = ?
        """,
        (experiment_id,),
    ).fetchall()
    null_ext = sum(1 for r in rows if r[0] is None)
    null_dec = sum(1 for r in rows if r[1] is None)
    out_of_range = sum(
        1
        for r in rows
        if r[0] is not None and r[1] is not None and (r[0] < 0 or r[0] > r[1] + 1e-6)
    )
    null_just = sum(1 for r in rows if r[2] is None)
    return {
        "n_decisions": len(rows),
        "extraction_null": null_ext,
        "declared_max_null": null_dec,
        "extraction_gt_declared_max": out_of_range,
        "justification_null": null_just,
    }


def main() -> None:
    eid = sys.argv[1] if len(sys.argv) > 1 else EXPERIMENT_ID
    conn = sqlite3.connect(RESULTS_DB_PATH)
    integ = integrity_check(conn, eid)
    print("INTEGRITY:", integ)
    ok = (
        integ["extraction_null"] == 0
        and integ["declared_max_null"] == 0
        and integ["extraction_gt_declared_max"] == 0
        and integ["n_decisions"] > 0
    )
    print("INTEGRITY RESULT:", "OK" if ok else "FAIL")

    rows = fetch_run_fidelity(conn, eid, max_round=0)
    conn.close()
    if not rows:
        print("No fidelity rows")
        raise SystemExit(1)

    low = [r.extraction_fraction for r in rows if r.coop_assigned <= 0.25]
    high = [r.extraction_fraction for r in rows if r.coop_assigned >= 0.75]
    if low and high:
        mean_diff = sum(high) / len(high) - sum(low) / len(low)
        print(f"mean_extraction_fraction low_coop={sum(low)/len(low):.4f} (n={len(low)})")
        print(f"mean_extraction_fraction high_coop={sum(high)/len(high):.4f} (n={len(high)})")
        print(f"mean_diff (high-low)={mean_diff:.4f}")

    xs = [r.coop_assigned for r in rows]
    ys = [r.extraction_fraction for r in rows]
    r, p = pearsonr(xs, ys)
    if r > 0.1:
        direction = "inverse (Haiku-like +)"
    elif r < -0.1:
        direction = "expected (Sonnet-like −)"
    else:
        direction = "flat/ambiguous"
    print(f"coop→extraction_fraction: r={r:.3f} p={p:.4f} n={len(xs)}")
    print(f"direction: {direction}")
    print("Compare: Haiku r≈+0.456 (inverse); Sonnet r≈−0.84 (expected)")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
