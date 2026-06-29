import re
import sqlite3
from pathlib import Path

from core.state import SimulationState

RESULTS_DB_PATH = "data/results.db"

_TRAIT_LEVELS: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
}

_RUN_ID_PATTERN = re.compile(
    r"^cond_(?P<coop>low|medium|high)_(?P<risk>low|medium|high)_rep(?P<rep>\d+)$"
)

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
    run_id TEXT NOT NULL,
    coop_level TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    coop_value REAL NOT NULL,
    risk_value REAL NOT NULL,
    replication INTEGER NOT NULL,
    UNIQUE (experiment_id, run_id)
)
"""

_INSERT_EXPERIMENT_CONDITION = """
INSERT INTO experiment_conditions (
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


def _experiment_conditions_needs_migration(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='experiment_conditions'"
    ).fetchone()
    if row is None:
        return False
    ddl = row[0] or ""
    return "UNIQUE (experiment_id, run_id)" not in ddl.replace("\n", " ")


def parse_run_id(run_id: str) -> tuple[str, str, float, float, int] | None:
    """Parse cond_{coop}_{risk}_rep{N} into trait levels and replication."""
    match = _RUN_ID_PATTERN.match(run_id)
    if not match:
        return None
    coop_level = match.group("coop")
    risk_level = match.group("risk")
    return (
        coop_level,
        risk_level,
        _TRAIT_LEVELS[coop_level],
        _TRAIT_LEVELS[risk_level],
        int(match.group("rep")),
    )


def _distinct_run_ids(conn: sqlite3.Connection, experiment_id: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT run_id FROM (
            SELECT run_id FROM metrics_snapshots WHERE experiment_id = ?
            UNION
            SELECT run_id FROM agent_decisions WHERE experiment_id = ?
        )
        ORDER BY run_id
        """,
        (experiment_id, experiment_id),
    ).fetchall()
    return [row[0] for row in rows]


def _condition_row_from_old(
    conn: sqlite3.Connection,
    experiment_id: str,
    run_id: str,
) -> tuple[str, str, float, float, int] | None:
    row = conn.execute(
        """
        SELECT coop_level, risk_level, coop_value, risk_value, replication
        FROM experiment_conditions_old
        WHERE experiment_id = ? AND run_id = ?
        """,
        (experiment_id, run_id),
    ).fetchone()
    if row is None:
        return None
    return row[0], row[1], row[2], row[3], row[4]


def _insert_condition_row(
    conn: sqlite3.Connection,
    experiment_id: str,
    run_id: str,
    coop_level: str,
    risk_level: str,
    coop_value: float,
    risk_value: float,
    replication: int,
) -> None:
    conn.execute(
        _INSERT_EXPERIMENT_CONDITION,
        (
            experiment_id,
            run_id,
            coop_level,
            risk_level,
            coop_value,
            risk_value,
            replication,
        ),
    )


def _rebuild_experiment_conditions(
    conn: sqlite3.Connection,
    experiment_id: str,
) -> int:
    inserted = 0
    for run_id in _distinct_run_ids(conn, experiment_id):
        parsed = _condition_row_from_old(conn, experiment_id, run_id)
        if parsed is None:
            parsed = parse_run_id(run_id)
        if parsed is None:
            raise ValueError(
                f"Cannot resolve conditions for {experiment_id}/{run_id}"
            )
        coop_level, risk_level, coop_value, risk_value, replication = parsed
        _insert_condition_row(
            conn,
            experiment_id,
            run_id,
            coop_level,
            risk_level,
            coop_value,
            risk_value,
            replication,
        )
        inserted += 1
    return inserted


def migrate_experiment_conditions_schema(
    db_path: str = RESULTS_DB_PATH,
) -> dict[str, int]:
    """Migrate experiment_conditions to composite UNIQUE (experiment_id, run_id).

    Renames the legacy table to experiment_conditions_old (kept as backup),
    rebuilds rows for full_experiment_v1 and control_group_v1 from the backup
    plus run_id parsing where rows were overwritten by cross-experiment collisions.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    targets = ("full_experiment_v1", "control_group_v1")
    expected = {"full_experiment_v1": 45, "control_group_v1": 27}

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        if "experiment_conditions_old" in tables:
            counts = {
                exp: conn.execute(
                    "SELECT COUNT(*) FROM experiment_conditions WHERE experiment_id = ?",
                    (exp,),
                ).fetchone()[0]
                for exp in targets
            }
            for exp, want in expected.items():
                if counts[exp] != want:
                    raise RuntimeError(
                        f"Migration already applied but {exp} has "
                        f"{counts[exp]} rows (expected {want})"
                    )
            return counts

        if "experiment_conditions" not in tables:
            conn.execute(_EXPERIMENT_CONDITIONS_DDL)
            conn.commit()
            return {exp: 0 for exp in targets}

        if not _experiment_conditions_needs_migration(conn):
            counts = {
                exp: conn.execute(
                    "SELECT COUNT(*) FROM experiment_conditions WHERE experiment_id = ?",
                    (exp,),
                ).fetchone()[0]
                for exp in targets
            }
            return counts

        conn.execute("ALTER TABLE experiment_conditions RENAME TO experiment_conditions_old")
        conn.execute(_EXPERIMENT_CONDITIONS_DDL)

        rebuilt: dict[str, int] = {}
        for experiment_id in targets:
            rebuilt[experiment_id] = _rebuild_experiment_conditions(conn, experiment_id)

        for experiment_id, want in expected.items():
            got = conn.execute(
                "SELECT COUNT(*) FROM experiment_conditions WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()[0]
            if got != want:
                raise RuntimeError(
                    f"Post-migration validation failed for {experiment_id}: "
                    f"got {got}, expected {want}"
                )

        conn.commit()
        return rebuilt


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
    Upserts on (experiment_id, run_id) so different experiments may share run_id strings.
    """
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            _INSERT_EXPERIMENT_CONDITION,
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
