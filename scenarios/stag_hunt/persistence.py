"""SQLite persistence for stag hunt runs (isolated DB)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from scenarios.stag_hunt.state import StagHuntRoundMetrics

STAG_HUNT_DB_PATH = "data/stag_hunt_results.db"

_CONDITIONS_DDL = """
CREATE TABLE IF NOT EXISTS stag_hunt_conditions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    coop_level TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    coop_value REAL NOT NULL,
    risk_value REAL NOT NULL,
    replication INTEGER NOT NULL,
    UNIQUE (experiment_id, run_id)
)
"""

_ROUNDS_DDL = """
CREATE TABLE IF NOT EXISTS stag_hunt_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    agent_a_choice TEXT NOT NULL,
    agent_b_choice TEXT NOT NULL,
    agent_a_payoff REAL NOT NULL,
    agent_b_payoff REAL NOT NULL,
    justification_a TEXT,
    justification_b TEXT
)
"""

_METRICS_DDL = """
CREATE TABLE IF NOT EXISTS stag_hunt_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    agent_a_choice TEXT NOT NULL,
    agent_b_choice TEXT NOT NULL,
    agent_a_payoff REAL NOT NULL,
    agent_b_payoff REAL NOT NULL,
    agent_a_score REAL NOT NULL,
    agent_b_score REAL NOT NULL,
    mutual_stag INTEGER NOT NULL,
    mutual_hare INTEGER NOT NULL,
    miscoordinated INTEGER NOT NULL
)
"""

_SUMMARY_DDL = """
CREATE TABLE IF NOT EXISTS stag_hunt_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    final_score_a REAL NOT NULL,
    final_score_b REAL NOT NULL,
    stag_rate_a REAL NOT NULL,
    stag_rate_b REAL NOT NULL,
    coordination_rate REAL NOT NULL,
    early_stag_rate_a REAL NOT NULL,
    late_stag_rate_a REAL NOT NULL,
    early_stag_rate_b REAL NOT NULL,
    late_stag_rate_b REAL NOT NULL,
    UNIQUE (experiment_id, run_id)
)
"""

_INSERT_CONDITION = """
INSERT INTO stag_hunt_conditions (
    experiment_id, run_id, coop_level, risk_level,
    coop_value, risk_value, replication
) VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (experiment_id, run_id) DO UPDATE SET
    coop_level = excluded.coop_level,
    risk_level = excluded.risk_level,
    coop_value = excluded.coop_value,
    risk_value = excluded.risk_value,
    replication = excluded.replication
"""

_INSERT_SUMMARY = """
INSERT INTO stag_hunt_summary (
    experiment_id, run_id, final_score_a, final_score_b,
    stag_rate_a, stag_rate_b, coordination_rate,
    early_stag_rate_a, late_stag_rate_a,
    early_stag_rate_b, late_stag_rate_b
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (experiment_id, run_id) DO UPDATE SET
    final_score_a = excluded.final_score_a,
    final_score_b = excluded.final_score_b,
    stag_rate_a = excluded.stag_rate_a,
    stag_rate_b = excluded.stag_rate_b,
    coordination_rate = excluded.coordination_rate,
    early_stag_rate_a = excluded.early_stag_rate_a,
    late_stag_rate_a = excluded.late_stag_rate_a,
    early_stag_rate_b = excluded.early_stag_rate_b,
    late_stag_rate_b = excluded.late_stag_rate_b
"""


def init_stag_hunt_db(db_path: str = STAG_HUNT_DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CONDITIONS_DDL)
        conn.execute(_ROUNDS_DDL)
        conn.execute(_METRICS_DDL)
        conn.execute(_SUMMARY_DDL)
        conn.commit()


def register_stag_hunt_conditions(
    experiment_id: str,
    conditions: list[tuple],
    db_path: str = STAG_HUNT_DB_PATH,
) -> None:
    """Each tuple: (run_id, coop_level, risk_level, coop_value, risk_value, replication)."""
    init_stag_hunt_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(_INSERT_CONDITION, [(experiment_id, *c) for c in conditions])
        conn.commit()


def save_stag_hunt_round(
    *,
    experiment_id: str,
    run_id: str,
    history_entry: dict,
    metrics: StagHuntRoundMetrics,
    db_path: str = STAG_HUNT_DB_PATH,
) -> None:
    """Persist one completed stag hunt round (raw log + referee metrics)."""
    init_stag_hunt_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO stag_hunt_rounds (
                experiment_id, run_id, round_number,
                agent_a_choice, agent_b_choice,
                agent_a_payoff, agent_b_payoff,
                justification_a, justification_b
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                run_id,
                history_entry["round_number"],
                history_entry["agent_a_choice"],
                history_entry["agent_b_choice"],
                history_entry["agent_a_payoff"],
                history_entry["agent_b_payoff"],
                history_entry.get("justification_a"),
                history_entry.get("justification_b"),
            ),
        )
        conn.execute(
            """
            INSERT INTO stag_hunt_metrics (
                experiment_id, run_id, round_number,
                agent_a_choice, agent_b_choice,
                agent_a_payoff, agent_b_payoff,
                agent_a_score, agent_b_score,
                mutual_stag, mutual_hare, miscoordinated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                run_id,
                metrics.round_number,
                metrics.agent_a_choice,
                metrics.agent_b_choice,
                metrics.agent_a_payoff,
                metrics.agent_b_payoff,
                metrics.agent_a_score,
                metrics.agent_b_score,
                1 if metrics.mutual_stag else 0,
                1 if metrics.mutual_hare else 0,
                1 if metrics.miscoordinated else 0,
            ),
        )
        conn.commit()


def save_stag_hunt_summary(
    *,
    experiment_id: str,
    run_id: str,
    final_score_a: float,
    final_score_b: float,
    stag_rate_a: float,
    stag_rate_b: float,
    coordination_rate: float,
    early_stag_rate_a: float,
    late_stag_rate_a: float,
    early_stag_rate_b: float,
    late_stag_rate_b: float,
    db_path: str = STAG_HUNT_DB_PATH,
) -> None:
    """Persist per-run summary metrics computed at termination (LLM-free)."""
    init_stag_hunt_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            _INSERT_SUMMARY,
            (
                experiment_id,
                run_id,
                final_score_a,
                final_score_b,
                stag_rate_a,
                stag_rate_b,
                coordination_rate,
                early_stag_rate_a,
                late_stag_rate_a,
                early_stag_rate_b,
                late_stag_rate_b,
            ),
        )
        conn.commit()
