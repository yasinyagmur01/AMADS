import sqlite3
from pathlib import Path

from core.state import SimulationState

RESULTS_DB_PATH = "data/results.db"

_AGENT_DECISIONS_DDL = """
CREATE TABLE IF NOT EXISTS agent_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    extraction_amount REAL NOT NULL,
    justification TEXT NOT NULL,
    declared_max REAL NOT NULL
)
"""

_METRICS_SNAPSHOTS_DDL = """
CREATE TABLE IF NOT EXISTS metrics_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    gini_coefficient REAL NOT NULL,
    cooperation_score_avg REAL NOT NULL,
    total_extraction REAL NOT NULL,
    pool_after REAL NOT NULL,
    constraint_violations INTEGER NOT NULL
)
"""

_EXPERIMENT_CONDITIONS_DDL = """
CREATE TABLE IF NOT EXISTS experiment_conditions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL UNIQUE,
    coop_level TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    coop_value REAL NOT NULL,
    risk_value REAL NOT NULL,
    replication INTEGER NOT NULL
)
"""


def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_AGENT_DECISIONS_DDL)
        conn.execute(_METRICS_SNAPSHOTS_DDL)
        conn.execute(_EXPERIMENT_CONDITIONS_DDL)
        conn.commit()


def register_experiment_conditions(
    experiment_id: str,
    conditions: list[tuple[str, str, str, float, float, int]],
    db_path: str = RESULTS_DB_PATH,
) -> None:
    """Persist run_id → condition mapping for downstream SQL joins.

    Each tuple: (run_id, coop_level, risk_level, coop_value, risk_value, replication)
    """
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO experiment_conditions (
                experiment_id, run_id, coop_level, risk_level,
                coop_value, risk_value, replication
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (experiment_id, run_id, coop_level, risk_level, coop_val, risk_val, rep)
                for run_id, coop_level, risk_level, coop_val, risk_val, rep in conditions
            ],
        )
        conn.commit()


def save_round_to_db(state: SimulationState, db_path: str) -> None:
    """Persist current-round decisions and the latest metrics snapshot."""
    init_db(db_path)

    current_round = state.round_number
    round_decisions = [
        d for d in state.round_decisions if d.round_number == current_round
    ]

    if not state.metrics_history:
        return

    metrics = state.metrics_history[-1]

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO agent_decisions (
                experiment_id, run_id, agent_id, round_number,
                extraction_amount, justification, declared_max
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    state.experiment_id,
                    state.run_id,
                    decision.agent_id,
                    decision.round_number,
                    decision.extraction_amount,
                    decision.justification,
                    decision.declared_max,
                )
                for decision in round_decisions
            ],
        )
        conn.execute(
            """
            INSERT INTO metrics_snapshots (
                experiment_id, run_id, round_number,
                gini_coefficient, cooperation_score_avg,
                total_extraction, pool_after, constraint_violations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.experiment_id,
                state.run_id,
                metrics.round_number,
                metrics.gini_coefficient,
                metrics.cooperation_score_avg,
                metrics.total_extraction,
                metrics.pool_after,
                metrics.constraint_violations,
            ),
        )
        conn.commit()
