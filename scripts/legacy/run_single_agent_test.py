"""Tek gerçek agent (agent_1) + 4 mock agent ile kısa test koşusu."""

import asyncio
import sqlite3

from langgraph.graph import END, StateGraph

from agents.decision_agent import (
    REAL_AGENT_ID,
    _INPUT_COST_PER_M,
    _OUTPUT_COST_PER_M,
    reset_token_usage,
    run_hybrid_agent_fanout,
    token_usage,
)
from core.config import settings
from core.database import RESULTS_DB_PATH
from core.state import EnvironmentSnapshot, SimulationState, TraitProfile
from environment.shocks import build_mock_dev_shock_schedule
from referee.referee_node import run_referee

EXPERIMENT_ID = "single_agent_test"
RUN_ID = "single_agent_001"
MAX_ROUNDS = 4  # 3-5 round aralığı; maliyet kontrolü

AGENT_PROFILES = [
    ("agent_1", 0.9, 0.1, "HighCoop"),
    ("agent_2", 0.7, 0.3, "ModerateCoop"),
    ("agent_3", 0.5, 0.5, "Balanced"),
    ("agent_4", 0.3, 0.7, "LowCoop"),
    ("agent_5", 0.1, 0.9, "Selfish"),
]


def _build_graph():
    workflow = StateGraph(SimulationState)
    workflow.add_node("agent_fanout", run_hybrid_agent_fanout)
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


def _make_initial_state() -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id=EXPERIMENT_ID,
        run_id=RUN_ID,
        max_rounds=MAX_ROUNDS,
        agent_traits=_make_traits(),
        shock_schedule=build_mock_dev_shock_schedule(),
        environment=EnvironmentSnapshot(
            pool_current=pool,
            pool_capacity=pool,
            regen_rate=1.15,
            max_extractable_this_round=pool * settings.EXTRACTION_LIMIT_RATIO,
            round_number=0,
            is_collapsed=False,
        ),
    )


def _print_db_summary():
    conn = sqlite3.connect(RESULTS_DB_PATH)
    metrics = conn.execute(
        "SELECT COUNT(*) FROM metrics_snapshots WHERE experiment_id = ?",
        (EXPERIMENT_ID,),
    ).fetchone()[0]
    decisions = conn.execute(
        "SELECT COUNT(*) FROM agent_decisions WHERE experiment_id = ?",
        (EXPERIMENT_ID,),
    ).fetchone()[0]
    real_decisions = conn.execute(
        "SELECT COUNT(*) FROM agent_decisions WHERE experiment_id = ? AND agent_id = ?",
        (EXPERIMENT_ID, REAL_AGENT_ID),
    ).fetchone()[0]
    conn.close()
    print(f"\n--- DB özeti ({RESULTS_DB_PATH}) ---")
    print(f"  experiment_id          : {EXPERIMENT_ID}")
    print(f"  metrics_snapshots satır: {metrics}")
    print(f"  agent_decisions satır  : {decisions}")
    print(f"  {REAL_AGENT_ID} kararları (LLM): {real_decisions}")


async def main():
    reset_token_usage()

    print("Tek-agent Anthropic test koşusu")
    print(f"  experiment_id = {EXPERIMENT_ID}")
    print(f"  run_id        = {RUN_ID}")
    print(f"  max_rounds    = {MAX_ROUNDS}")
    print(f"  gerçek agent  = {REAL_AGENT_ID} ({settings.ANTHROPIC_MODEL})")
    print(f"  temperature   = {settings.TEMPERATURE}")
    print(f"  mock agentlar = agent_2 … agent_5")

    app = _build_graph()
    final_state = await app.ainvoke(_make_initial_state())

    print("\n--- Simülasyon sonucu ---")
    print(f"  round_number       : {final_state['round_number']}")
    print(f"  termination_reason : {final_state['termination_reason']}")
    print(f"  pool_current       : {final_state['environment'].pool_current:.4f}")

    print(f"\n--- {REAL_AGENT_ID} LLM kararları ---")
    for d in final_state["round_decisions"]:
        if d.agent_id == REAL_AGENT_ID:
            print(
                f"  round {d.round_number}: "
                f"extract={d.extraction_amount:.4f}, "
                f"declared_max={d.declared_max:.4f}, "
                f"justification={d.justification[:80]}…"
            )

    _print_db_summary()

    print("\n--- Token kullanımı ve tahmini maliyet ---")
    print(f"  input_tokens  : {token_usage.input_tokens}")
    print(f"  output_tokens : {token_usage.output_tokens}")
    print(f"  toplam token  : {token_usage.input_tokens + token_usage.output_tokens}")
    print(f"  tahmini maliyet: ${token_usage.estimated_cost_usd():.6f} USD")
    print(
        f"  (fiyatlandırma: ${_INPUT_COST_PER_M:.2f}/M input, "
        f"${_OUTPUT_COST_PER_M:.2f}/M output — Claude Haiku 4.5)"
    )


if __name__ == "__main__":
    asyncio.run(main())
