"""
Shared definitions for heterogeneous_v1 population compositions.

Four fixed within-run trait mixtures (COND_H1–H4). Used by the seed script,
experiment runner, and analysis — single source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypedDict

from core.state import TraitProfile

EXPERIMENT_ID = "heterogeneous_v1"
REPLICATIONS = 5
AGENT_IDS = [f"agent_{i}" for i in range(1, 6)]


class AgentTraitPair(TypedDict):
    coop: float
    risk: float


Composition = dict[str, AgentTraitPair]


@dataclass(frozen=True)
class HeteroCondition:
    condition_id: str  # h1, h2, h3, h4
    label: str  # COND_H1, …
    name: str
    composition: Composition

    def composition_json(self) -> str:
        return json.dumps(self.composition, sort_keys=True)

    def mean_coop(self) -> float:
        return sum(t["coop"] for t in self.composition.values()) / len(self.composition)

    def mean_risk(self) -> float:
        return sum(t["risk"] for t in self.composition.values()) / len(self.composition)


CONDITIONS: dict[str, HeteroCondition] = {
    "h1": HeteroCondition(
        condition_id="h1",
        label="COND_H1",
        name="High Cooperation Dominant",
        composition={
            "agent_1": {"coop": 0.8, "risk": 0.2},
            "agent_2": {"coop": 0.8, "risk": 0.2},
            "agent_3": {"coop": 0.8, "risk": 0.2},
            "agent_4": {"coop": 0.5, "risk": 0.5},
            "agent_5": {"coop": 0.2, "risk": 0.8},
        },
    ),
    "h2": HeteroCondition(
        condition_id="h2",
        label="COND_H2",
        name="Low Cooperation Dominant",
        composition={
            "agent_1": {"coop": 0.2, "risk": 0.8},
            "agent_2": {"coop": 0.2, "risk": 0.8},
            "agent_3": {"coop": 0.2, "risk": 0.8},
            "agent_4": {"coop": 0.5, "risk": 0.5},
            "agent_5": {"coop": 0.8, "risk": 0.2},
        },
    ),
    "h3": HeteroCondition(
        condition_id="h3",
        label="COND_H3",
        name="Mixed Equal",
        composition={
            "agent_1": {"coop": 0.8, "risk": 0.2},
            "agent_2": {"coop": 0.8, "risk": 0.2},
            "agent_3": {"coop": 0.5, "risk": 0.5},
            "agent_4": {"coop": 0.2, "risk": 0.8},
            "agent_5": {"coop": 0.2, "risk": 0.8},
        },
    ),
    "h4": HeteroCondition(
        condition_id="h4",
        label="COND_H4",
        name="Homogeneous Medium (control)",
        composition={
            "agent_1": {"coop": 0.5, "risk": 0.5},
            "agent_2": {"coop": 0.5, "risk": 0.5},
            "agent_3": {"coop": 0.5, "risk": 0.5},
            "agent_4": {"coop": 0.5, "risk": 0.5},
            "agent_5": {"coop": 0.5, "risk": 0.5},
        },
    ),
}

CONDITION_ORDER = ("h1", "h2", "h3", "h4")


def _level_for_value(value: float) -> str:
    if value <= 0.3:
        return "low"
    if value >= 0.7:
        return "high"
    return "medium"


def profile_label(coop: float, risk: float) -> str:
    return f"{_level_for_value(coop).capitalize()}Coop_{_level_for_value(risk).capitalize()}Risk"


def make_agent_traits(condition: HeteroCondition) -> dict[str, TraitProfile]:
    """Per-agent TraitProfile dict — heterogeneous assignment path."""
    traits: dict[str, TraitProfile] = {}
    for agent_id in AGENT_IDS:
        pair = condition.composition[agent_id]
        traits[agent_id] = TraitProfile(
            agent_id=agent_id,
            cooperation_assigned=pair["coop"],
            risk_tolerance_assigned=pair["risk"],
            profile_label=profile_label(pair["coop"], pair["risk"]),
        )
    return traits


def run_id_for(condition_id: str, replication: int) -> str:
    return f"cond_{condition_id}_rep{replication}"


def parse_condition_id(run_id: str) -> str | None:
    """Extract h1|h2|h3|h4 from cond_hN_repM."""
    if not run_id.startswith("cond_") or "_rep" not in run_id:
        return None
    mid = run_id[len("cond_") : run_id.rfind("_rep")]
    return mid if mid in CONDITIONS else None


# Side table for composition JSON (does not alter core experiment_conditions DDL).
COMPOSITION_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS heterogeneous_compositions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    condition_id TEXT NOT NULL,
    label TEXT NOT NULL,
    name TEXT NOT NULL,
    composition_json TEXT NOT NULL,
    UNIQUE (experiment_id, condition_id)
)
"""

_UPSERT_COMPOSITION = """
INSERT INTO heterogeneous_compositions (
    experiment_id, condition_id, label, name, composition_json
) VALUES (?, ?, ?, ?, ?)
ON CONFLICT (experiment_id, condition_id) DO UPDATE SET
    label = excluded.label,
    name = excluded.name,
    composition_json = excluded.composition_json
"""


def ensure_composition_table(conn) -> None:
    conn.execute(COMPOSITION_TABLE_DDL)


def upsert_compositions(conn, experiment_id: str = EXPERIMENT_ID) -> int:
    ensure_composition_table(conn)
    rows = [
        (
            experiment_id,
            cond.condition_id,
            cond.label,
            cond.name,
            cond.composition_json(),
        )
        for cond in (CONDITIONS[cid] for cid in CONDITION_ORDER)
    ]
    conn.executemany(_UPSERT_COMPOSITION, rows)
    return len(rows)


def load_composition_from_db(
    conn, condition_id: str, experiment_id: str = EXPERIMENT_ID
) -> Composition | None:
    ensure_composition_table(conn)
    row = conn.execute(
        """
        SELECT composition_json FROM heterogeneous_compositions
        WHERE experiment_id = ? AND condition_id = ?
        """,
        (experiment_id, condition_id),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def experiment_condition_rows(
    *,
    replications: int = REPLICATIONS,
) -> list[tuple[str, str, str, float, float, int]]:
    """Rows for register_experiment_conditions (flat join metadata).

    coop_level stores condition_id (h1–h4); risk_level is 'hetero' except
    COND_H4 which uses 'medium' to mirror the homogeneous medium control cell.
    coop_value / risk_value are population means (H4 → 0.5 / 0.5).
    """
    rows: list[tuple[str, str, str, float, float, int]] = []
    for cid in CONDITION_ORDER:
        cond = CONDITIONS[cid]
        risk_level = "medium" if cid == "h4" else "hetero"
        for rep in range(1, replications + 1):
            rows.append(
                (
                    run_id_for(cid, rep),
                    cid,
                    risk_level,
                    round(cond.mean_coop(), 4),
                    round(cond.mean_risk(), 4),
                    rep,
                )
            )
    return rows
