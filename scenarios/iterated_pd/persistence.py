"""SQLite persistence for iterated prisoner's dilemma runs (isolated DB)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from scenarios.iterated_pd.state import IPDRoundMetrics

IPD_DB_PATH = "data/ipd_results.db"

_CONDITIONS_DDL = """
CREATE TABLE IF NOT EXISTS ipd_conditions (
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
CREATE TABLE IF NOT EXISTS ipd_rounds (
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
CREATE TABLE IF NOT EXISTS ipd_metrics (
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
    mutual_cooperation INTEGER NOT NULL,
    mutual_defection INTEGER NOT NULL
)
"""

_SUMMARY_DDL = """
CREATE TABLE IF NOT EXISTS ipd_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    final_score_a REAL NOT NULL,
    final_score_b REAL NOT NULL,
    forgiveness_rate_a REAL NOT NULL,
    forgiveness_rate_b REAL NOT NULL,
    early_coop_rate_a REAL NOT NULL,
    late_coop_rate_a REAL NOT NULL,
    early_coop_rate_b REAL NOT NULL,
    late_coop_rate_b REAL NOT NULL,
    UNIQUE (experiment_id, run_id)
)
"""

_INSERT_CONDITION = """
INSERT INTO ipd_conditions (
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
INSERT INTO ipd_summary (
    experiment_id, run_id, final_score_a, final_score_b,
    forgiveness_rate_a, forgiveness_rate_b,
    early_coop_rate_a, late_coop_rate_a,
    early_coop_rate_b, late_coop_rate_b
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (experiment_id, run_id) DO UPDATE SET
    final_score_a = excluded.final_score_a,
    final_score_b = excluded.final_score_b,
    forgiveness_rate_a = excluded.forgiveness_rate_a,
    forgiveness_rate_b = excluded.forgiveness_rate_b,
    early_coop_rate_a = excluded.early_coop_rate_a,
    late_coop_rate_a = excluded.late_coop_rate_a,
    early_coop_rate_b = excluded.early_coop_rate_b,
    late_coop_rate_b = excluded.late_coop_rate_b
"""


def init_ipd_db(db_path: str = IPD_DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CONDITIONS_DDL)
        conn.execute(_ROUNDS_DDL)
        conn.execute(_METRICS_DDL)
        conn.execute(_SUMMARY_DDL)
        conn.commit()


def register_ipd_conditions(
    experiment_id: str,
    conditions: list[tuple],
    db_path: str = IPD_DB_PATH,
) -> None:
    """Each tuple: (run_id, coop_level, risk_level, coop_value, risk_value, replication)."""
    init_ipd_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(_INSERT_CONDITION, [(experiment_id, *c) for c in conditions])
        conn.commit()


def save_ipd_round(
    *,
    experiment_id: str,
    run_id: str,
    history_entry: dict,
    metrics: IPDRoundMetrics,
    db_path: str = IPD_DB_PATH,
) -> None:
    """Persist one completed IPD round (raw log + referee metrics)."""
    init_ipd_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ipd_rounds (
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
            INSERT INTO ipd_metrics (
                experiment_id, run_id, round_number,
                agent_a_choice, agent_b_choice,
                agent_a_payoff, agent_b_payoff,
                agent_a_score, agent_b_score,
                mutual_cooperation, mutual_defection
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                1 if metrics.mutual_cooperation else 0,
                1 if metrics.mutual_defection else 0,
            ),
        )
        conn.commit()


def save_ipd_summary(
    *,
    experiment_id: str,
    run_id: str,
    final_score_a: float,
    final_score_b: float,
    forgiveness_rate_a: float,
    forgiveness_rate_b: float,
    early_coop_rate_a: float,
    late_coop_rate_a: float,
    early_coop_rate_b: float,
    late_coop_rate_b: float,
    db_path: str = IPD_DB_PATH,
) -> None:
    """Persist per-run summary metrics computed at termination (LLM-free)."""
    init_ipd_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            _INSERT_SUMMARY,
            (
                experiment_id,
                run_id,
                final_score_a,
                final_score_b,
                forgiveness_rate_a,
                forgiveness_rate_b,
                early_coop_rate_a,
                late_coop_rate_a,
                early_coop_rate_b,
                late_coop_rate_b,
            ),
        )
        conn.commit()
