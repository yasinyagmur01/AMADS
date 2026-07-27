"""SQLite persistence for bargaining runs (separate from CPR tables)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core.state import TraitProfile
from scenarios.bargaining.state import BargainingMetricsSnapshot

BARGAINING_DB_PATH = "data/bargaining_results.db"

_CONDITIONS_DDL = """
CREATE TABLE IF NOT EXISTS bargaining_conditions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    proposer_coop_level TEXT NOT NULL,
    proposer_risk_level TEXT NOT NULL,
    responder_coop_level TEXT NOT NULL,
    responder_risk_level TEXT NOT NULL,
    proposer_coop REAL NOT NULL,
    proposer_risk REAL NOT NULL,
    responder_coop REAL NOT NULL,
    responder_risk REAL NOT NULL,
    replication INTEGER NOT NULL,
    UNIQUE (experiment_id, run_id)
)
"""

_ROUNDS_DDL = """
CREATE TABLE IF NOT EXISTS bargaining_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    keep_amount REAL NOT NULL,
    offer_to_responder REAL NOT NULL,
    accepted INTEGER NOT NULL,
    proposer_payoff REAL NOT NULL,
    responder_payoff REAL NOT NULL,
    time_pressure REAL NOT NULL,
    resource_scarcity REAL NOT NULL,
    justification_proposer TEXT,
    justification_responder TEXT
)
"""

_METRICS_DDL = """
CREATE TABLE IF NOT EXISTS bargaining_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    keep_amount REAL NOT NULL,
    offer_to_responder REAL NOT NULL,
    accepted INTEGER NOT NULL,
    proposer_payoff REAL NOT NULL,
    responder_payoff REAL NOT NULL,
    keep_fraction REAL NOT NULL,
    proposer_coop_alignment REAL NOT NULL,
    responder_coop_alignment REAL NOT NULL
)
"""

_INSERT_CONDITION = """
INSERT INTO bargaining_conditions (
    experiment_id, run_id,
    proposer_coop_level, proposer_risk_level,
    responder_coop_level, responder_risk_level,
    proposer_coop, proposer_risk, responder_coop, responder_risk,
    replication
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (experiment_id, run_id) DO UPDATE SET
    proposer_coop_level = excluded.proposer_coop_level,
    proposer_risk_level = excluded.proposer_risk_level,
    responder_coop_level = excluded.responder_coop_level,
    responder_risk_level = excluded.responder_risk_level,
    proposer_coop = excluded.proposer_coop,
    proposer_risk = excluded.proposer_risk,
    responder_coop = excluded.responder_coop,
    responder_risk = excluded.responder_risk,
    replication = excluded.replication
"""


def init_bargaining_db(db_path: str = BARGAINING_DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CONDITIONS_DDL)
        conn.execute(_ROUNDS_DDL)
        conn.execute(_METRICS_DDL)
        conn.commit()


def register_bargaining_conditions(
    experiment_id: str,
    conditions: list[tuple],
    db_path: str = BARGAINING_DB_PATH,
) -> None:
    """
    Each tuple:
      (run_id, p_coop_level, p_risk_level, r_coop_level, r_risk_level,
       p_coop, p_risk, r_coop, r_risk, replication)
    """
    init_bargaining_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            _INSERT_CONDITION,
            [
                (
                    experiment_id,
                    run_id,
                    p_coop_lvl,
                    p_risk_lvl,
                    r_coop_lvl,
                    r_risk_lvl,
                    p_coop,
                    p_risk,
                    r_coop,
                    r_risk,
                    rep,
                )
                for (
                    run_id,
                    p_coop_lvl,
                    p_risk_lvl,
                    r_coop_lvl,
                    r_risk_lvl,
                    p_coop,
                    p_risk,
                    r_coop,
                    r_risk,
                    rep,
                ) in conditions
            ],
        )
        conn.commit()


def save_bargaining_round(
    *,
    experiment_id: str,
    run_id: str,
    history_entry: dict,
    metrics: BargainingMetricsSnapshot,
    proposer_trait: TraitProfile,
    responder_trait: TraitProfile,
    db_path: str = BARGAINING_DB_PATH,
) -> None:
    """Persist one completed bargaining round. Traits unused for now (audit hook)."""
    del proposer_trait, responder_trait  # reserved for future trait-log columns
    init_bargaining_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO bargaining_rounds (
                experiment_id, run_id, round_number,
                keep_amount, offer_to_responder, accepted,
                proposer_payoff, responder_payoff,
                time_pressure, resource_scarcity,
                justification_proposer, justification_responder
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                run_id,
                history_entry["round_number"],
                history_entry["keep_amount"],
                history_entry["offer_to_responder"],
                1 if history_entry["accepted"] else 0,
                history_entry["proposer_payoff"],
                history_entry["responder_payoff"],
                history_entry["time_pressure"],
                history_entry["resource_scarcity"],
                history_entry.get("justification_proposer"),
                history_entry.get("justification_responder"),
            ),
        )
        conn.execute(
            """
            INSERT INTO bargaining_metrics (
                experiment_id, run_id, round_number,
                keep_amount, offer_to_responder, accepted,
                proposer_payoff, responder_payoff, keep_fraction,
                proposer_coop_alignment, responder_coop_alignment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                run_id,
                metrics.round_number,
                metrics.keep_amount,
                metrics.offer_to_responder,
                1 if metrics.accepted else 0,
                metrics.proposer_payoff,
                metrics.responder_payoff,
                metrics.keep_fraction,
                metrics.proposer_coop_alignment,
                metrics.responder_coop_alignment,
            ),
        )
        conn.commit()
