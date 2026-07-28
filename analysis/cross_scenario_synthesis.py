"""Cross-scenario trait-fidelity synthesis table (Phase 9).

Pulls completed scenario × trait × model results and tags structural axes.
Patterns across few scenarios are labeled qualitative/suggestive, not statistically
testable as scenario-level effects.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scipy.stats import pearsonr

from analysis.trait_fidelity import fetch_run_fidelity
from core.database import RESULTS_DB_PATH
from scenarios.bargaining.persistence import BARGAINING_DB_PATH


@dataclass(frozen=True)
class FidelityRow:
    scenario: str
    trait: str
    model: str
    experiment_id: str
    n: int
    r: float
    p: float
    direction: str
    evidence: str  # statistically_supported | qualitative_suggestive | anomaly
    axes: str


def _dir(r: float, expected_sign: float) -> str:
    if abs(r) < 0.1:
        return "flat"
    observed = 1.0 if r > 0 else -1.0
    if observed == expected_sign:
        return "expected"
    return "inverse"


def _cpr_rows() -> list[FidelityRow]:
    out: list[FidelityRow] = []
    # Locked Haiku (published constants)
    out.append(
        FidelityRow(
            "CPR",
            "cooperation",
            "Haiku",
            "full_experiment_v1",
            45,
            0.456,
            0.0016,
            "inverse",
            "statistically_supported",
            "simultaneous;N-agent;repeated-no-partner-memory;conflict",
        )
    )
    out.append(
        FidelityRow(
            "CPR",
            "risk",
            "Haiku",
            "full_experiment_v1",
            45,
            0.678,
            0.0001,
            "expected",
            "statistically_supported",
            "simultaneous;N-agent;repeated-no-partner-memory;conflict",
        )
    )
    out.append(
        FidelityRow(
            "CPR",
            "cooperation",
            "Sonnet",
            "sonnet_crossmodel_v1",
            40,
            -0.84,
            0.001,
            "expected",
            "statistically_supported",
            "simultaneous;N-agent;repeated-no-partner-memory;conflict",
        )
    )
    # Groq micro-pilot from DB if present
    path = Path(RESULTS_DB_PATH)
    if path.exists():
        with sqlite3.connect(path) as conn:
            rows = fetch_run_fidelity(conn, "full_experiment_groq_v1", max_round=0)
        if len(rows) >= 4:
            xs = [r.coop_assigned for r in rows]
            ys = [r.extraction_fraction for r in rows]
            r, p = pearsonr(xs, ys)
            out.append(
                FidelityRow(
                    "CPR",
                    "cooperation",
                    "Groq-8b",
                    "full_experiment_groq_v1",
                    len(rows),
                    float(r),
                    float(p),
                    _dir(float(r), expected_sign=-1.0),
                    "statistically_supported" if p < 0.05 else "qualitative_suggestive",
                    "simultaneous;N-agent;repeated-no-partner-memory;conflict",
                )
            )
    return out


def _bargaining_rows() -> list[FidelityRow]:
    out: list[FidelityRow] = []
    out.append(
        FidelityRow(
            "Bargaining",
            "cooperation",
            "Haiku",
            "bargaining_v1",
            10,
            -0.981,
            0.001,
            "expected",
            "statistically_supported",
            "sequential;dyadic;repeated;conflict",
        )
    )
    out.append(
        FidelityRow(
            "Bargaining",
            "risk",
            "Haiku",
            "bargaining_risk_v1",
            10,
            0.548,
            0.10,
            "expected",
            "qualitative_suggestive",
            "sequential;dyadic;repeated;conflict",
        )
    )
    path = Path(BARGAINING_DB_PATH)
    if not path.exists():
        return out
    with sqlite3.connect(path) as conn:
        for eid, trait, model, expected_sign, sql_x, sql_y in (
            (
                "bargaining_groq_v1",
                "cooperation",
                "Groq-8b",
                -1.0,
                """
                SELECT c.proposer_coop, AVG(m.keep_amount)
                FROM bargaining_conditions c
                JOIN bargaining_metrics m
                  ON c.experiment_id=m.experiment_id AND c.run_id=m.run_id
                WHERE c.experiment_id=?
                GROUP BY c.run_id
                """,
                None,
            ),
            (
                "bargaining_risk_groq_v1",
                "risk",
                "Groq-8b",
                1.0,
                """
                SELECT c.proposer_risk, AVG(m.keep_amount)
                FROM bargaining_conditions c
                JOIN bargaining_metrics m
                  ON c.experiment_id=m.experiment_id AND c.run_id=m.run_id
                WHERE c.experiment_id=?
                GROUP BY c.run_id
                """,
                None,
            ),
        ):
            rows = conn.execute(sql_x, (eid,)).fetchall()
            if len(rows) < 4:
                continue
            xs = [float(r[0]) for r in rows]
            ys = [float(r[1]) for r in rows]
            r, p = pearsonr(xs, ys)
            out.append(
                FidelityRow(
                    "Bargaining",
                    trait,
                    model,
                    eid,
                    len(rows),
                    float(r),
                    float(p),
                    _dir(float(r), expected_sign=expected_sign),
                    "statistically_supported" if p < 0.05 else "qualitative_suggestive",
                    "sequential;dyadic;repeated;conflict",
                )
            )
    return out


def _optional_scenario_rows(db_path: str, eid: str, scenario: str, axes: str) -> list[FidelityRow]:
    path = Path(db_path)
    if not path.exists():
        return []
    out: list[FidelityRow] = []
    with sqlite3.connect(path) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        # Prefer summary table if present
        summary = None
        for cand in tables:
            if cand.endswith("_summary") or cand == "run_summary":
                summary = cand
                break
        if summary:
            # Best-effort: print presence only; detailed metrics filled by scenario scripts
            n = conn.execute(
                f"SELECT COUNT(*) FROM {summary} WHERE experiment_id=?",
                (eid,),
            ).fetchone()[0]
            if n:
                out.append(
                    FidelityRow(
                        scenario,
                        "cooperation",
                        "Groq-8b",
                        eid,
                        n,
                        float("nan"),
                        float("nan"),
                        "see scenario report",
                        "qualitative_suggestive",
                        axes,
                    )
                )
    return out


def main() -> None:
    rows = _cpr_rows() + _bargaining_rows()
    rows += _optional_scenario_rows(
        "data/ipd_results.db",
        "iterated_pd_groq_v1",
        "IteratedPD",
        "simultaneous;dyadic;repeated+memory;conflict",
    )
    rows += _optional_scenario_rows(
        "data/stag_hunt_results.db",
        "stag_hunt_groq_v1",
        "StagHunt",
        "simultaneous;dyadic;repeated;coordination",
    )

    print("=" * 100)
    print("CROSS-SCENARIO FIDELITY SYNTHESIS")
    print("=" * 100)
    print(
        f"{'scenario':<12} {'trait':<12} {'model':<10} {'n':>3} {'r':>7} {'p':>8} "
        f"{'dir':<10} {'evidence':<24} experiment_id"
    )
    print("-" * 100)
    for r in rows:
        r_s = f"{r.r:.3f}" if r.r == r.r else "  n/a"
        p_s = f"{r.p:.4f}" if r.p == r.p else "  n/a"
        print(
            f"{r.scenario:<12} {r.trait:<12} {r.model:<10} {r.n:3d} {r_s:>7} {p_s:>8} "
            f"{r.direction:<10} {r.evidence:<24} {r.experiment_id}"
        )
    print()
    print("Structural-axis note: with ≤4 scenarios, any axis↔direction pattern is")
    print("QUALITATIVE/SUGGESTIVE only — not statistically testable at the scenario level.")
    print()
    print("Observed pattern (suggestive):")
    print("  - CPR cooperation inverse on Haiku and Groq-8b; expected on Sonnet.")
    print("  - Bargaining cooperation expected on Haiku; inverse on Groq-8b.")
    print("  - Model identity appears at least as important as scenario structure.")


if __name__ == "__main__":
    main()
