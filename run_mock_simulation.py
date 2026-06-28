import asyncio
import sqlite3
from pathlib import Path

from langgraph.graph import END, StateGraph

from environment.shocks import build_mock_dev_shock_schedule
from agents.mock_agent import run_aggressive_mock_agent_fanout, run_mock_agent_fanout
from core.config import settings
from core.database import RESULTS_DB_PATH
from core.state import (
    EnvironmentSnapshot,
    SimulationState,
    TraitProfile,
)
from referee.referee_node import run_referee

AGENT_PROFILES = [
    ("agent_1", 0.9, 0.1, "HighCoop"),
    ("agent_2", 0.7, 0.3, "ModerateCoop"),
    ("agent_3", 0.5, 0.5, "Balanced"),
    ("agent_4", 0.3, 0.7, "LowCoop"),
    ("agent_5", 0.1, 0.9, "Selfish"),
]

SHOCK_SCHEDULE = build_mock_dev_shock_schedule()


def _build_graph(agent_node):
    workflow = StateGraph(SimulationState)
    workflow.add_node("agent_fanout", agent_node)
    workflow.add_node("referee", run_referee)
    workflow.set_entry_point("agent_fanout")
    workflow.add_edge("agent_fanout", "referee")
    workflow.add_conditional_edges(
        "referee",
        lambda state: END if state.is_terminated else "agent_fanout",
    )
    return workflow.compile()


def _make_traits() -> dict[str, TraitProfile]:
    return {
        agent_id: TraitProfile(
            agent_id=agent_id,
            cooperation_assigned=coop,
            risk_tolerance_assigned=risk,
            profile_label=label,
        )
        for agent_id, coop, risk, label in AGENT_PROFILES
    }


def _make_initial_state(experiment_id: str, run_id: str) -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id=experiment_id,
        run_id=run_id,
        max_rounds=settings.MAX_ROUNDS,
        agent_traits=_make_traits(),
        shock_schedule=SHOCK_SCHEDULE,
        environment=EnvironmentSnapshot(
            pool_current=pool,
            pool_capacity=pool,
            regen_rate=1.15,
            max_extractable_this_round=pool * settings.EXTRACTION_LIMIT_RATIO,
            round_number=0,
            is_collapsed=False,
        ),
    )


def reset_results_db(db_path: str = RESULTS_DB_PATH) -> None:
    path = Path(db_path)
    if path.exists():
        path.unlink()
    print(f"Reset: {db_path} silindi (varsa).")


def _validate_db(run_id: str, expected_rounds: int, db_path: str = RESULTS_DB_PATH):
    conn = sqlite3.connect(db_path)
    metrics_rows = conn.execute(
        "SELECT round_number FROM metrics_snapshots WHERE run_id = ? ORDER BY round_number",
        (run_id,),
    ).fetchall()
    decision_count = conn.execute(
        "SELECT COUNT(*) FROM agent_decisions WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    conn.close()

    metric_rounds = [r[0] for r in metrics_rows]
    expected_metric_rounds = list(range(expected_rounds))
    missing = [r for r in expected_metric_rounds if r not in metric_rounds]

    return {
        "metrics_count": len(metric_rounds),
        "metrics_rounds": metric_rounds,
        "missing_rounds": missing,
        "decisions_count": decision_count,
    }


async def _run_scenario(
    label: str,
    experiment_id: str,
    run_id: str,
    agent_node,
    expected_rounds: int,
    expect_full_completion: bool = True,
):
    print(f"\n{'=' * 60}")
    print(f"RUN: {label}")
    print(f"  experiment_id = {experiment_id}")
    print(f"  run_id (LangSmith thread_id) = {run_id}")
    print(f"  LangSmith project = {settings.LANGSMITH_PROJECT}")
    print(f"{'=' * 60}")

    app = _build_graph(agent_node)
    initial_state = _make_initial_state(experiment_id, run_id)
    config = {"configurable": {"thread_id": run_id}}

    final_state = await app.ainvoke(initial_state, config=config)

    db_stats = _validate_db(run_id, expected_rounds)

    print(f"\n--- Sonuç: {label} ---")
    print(f"  round_number (final)     : {final_state['round_number']}")
    print(f"  termination_reason       : {final_state['termination_reason']}")
    print(f"  is_terminated            : {final_state['is_terminated']}")
    print(f"  pool_current (final)     : {final_state['environment'].pool_current:.4f}")
    print(f"  metrics_snapshots satır  : {db_stats['metrics_count']}", end="")
    if expect_full_completion:
        print(f" (beklenen: {expected_rounds})")
        if db_stats["missing_rounds"]:
            print(f"  EKSİK round'lar         : {db_stats['missing_rounds']}")
        else:
            print(f"  metrics round'ları       : {db_stats['metrics_rounds']}")
    else:
        print(f" (erken bitiş — max {expected_rounds} değil)")
        print(f"  oynanan round'lar         : {db_stats['metrics_rounds']}")
    print(
        f"  agent_decisions satır    : {db_stats['decisions_count']}"
        f" (beklenen: {db_stats['metrics_count'] * settings.AGENT_COUNT})"
    )

    return final_state, db_stats


async def main():
    reset_results_db()

    print("LangSmith tracing env:")
    print(f"  LANGSMITH_TRACING = {settings.LANGSMITH_TRACING}")
    print(f"  LANGSMITH_PROJECT = {settings.LANGSMITH_PROJECT}")
    print(f"  LANGSMITH_API_KEY = {'set' if settings.LANGSMITH_API_KEY else 'NOT SET'}")

    completed_state, completed_db = await _run_scenario(
        label="Tam mock koşusu (moderate extraction)",
        experiment_id="mock_dev_complete",
        run_id="mock_complete_001",
        agent_node=run_mock_agent_fanout,
        expected_rounds=settings.MAX_ROUNDS,
    )

    ok_completed = (
        completed_state["round_number"] == settings.MAX_ROUNDS
        and completed_state["termination_reason"] == "completed"
        and completed_db["metrics_count"] == settings.MAX_ROUNDS
        and completed_db["decisions_count"] == settings.MAX_ROUNDS * settings.AGENT_COUNT
    )
    print(f"\n  [DOĞRULAMA] completed run OK: {ok_completed}")

    collapse_state, collapse_db = await _run_scenario(
        label="Collapse mock koşusu (aggressive extraction)",
        experiment_id="mock_dev_collapse",
        run_id="mock_collapse_001",
        agent_node=run_aggressive_mock_agent_fanout,
        expected_rounds=settings.MAX_ROUNDS,
        expect_full_completion=False,
    )

    collapse_rounds_played = collapse_db["metrics_count"]
    ok_collapse = (
        collapse_state["termination_reason"] == "collapse"
        and collapse_state["round_number"] < settings.MAX_ROUNDS
        and collapse_state["environment"].pool_current <= 0
    )
    print(f"\n  [DOĞRULAMA] collapse run OK: {ok_collapse}")
    print(f"  Collapse round (0-indexed son oynanan): {collapse_rounds_played - 1 if collapse_rounds_played else 'N/A'}")
    print(f"  round_number after collapse: {collapse_state['round_number']}")

    print(f"\n{'=' * 60}")
    print("LangSmith'te kontrol için run_id (thread_id) değerleri:")
    print("  mock_complete_001  → tam 15 round koşusu")
    print("  mock_collapse_001  → erken collapse koşusu")
    print(f"  Proje: {settings.LANGSMITH_PROJECT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
