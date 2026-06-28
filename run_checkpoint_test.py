"""
LangGraph SqliteSaver checkpoint doğrulama testi.

Senaryo:
  1. thread_id=checkpoint_test_001 ile mock graph'ı 5 round (0–4) çalıştır, bilerek dur.
  2. Aynı thread_id ile yeni oturumda max_rounds=15'e kadar devam et.
  3. State'in checkpoint'ten geri yüklendiğini doğrula (round 5'ten devam, metrics korunmuş).

Not: Bu test data/results.db'ye yazmaz; ayrı checkpoint ve results dosyaları kullanır.
Gereksinim: pip install langgraph-checkpoint-sqlite aiosqlite
"""

import asyncio
import warnings
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

import core.database as db
import referee.referee_node as referee_node
from agents.mock_agent import run_mock_agent_fanout
from core.config import settings
from core.state import EnvironmentSnapshot, SimulationState, TraitProfile
from environment.shocks import build_mock_dev_shock_schedule

# --- Sabitler ---
THREAD_ID = "checkpoint_test_001"
EXPERIMENT_ID = "checkpoint_test"
MAX_ROUNDS = settings.MAX_ROUNDS  # 15
FIRST_PHASE_ROUNDS = 5  # round 0–4 oynanır; round_number 5'e ulaşınca dur
CHECKPOINT_DB = "data/checkpoint_test.db"
RESULTS_DB = "data/checkpoint_test_results.db"

AGENT_PROFILES = [
    ("agent_1", 0.9, 0.1, "HighCoop"),
    ("agent_2", 0.7, 0.3, "ModerateCoop"),
    ("agent_3", 0.5, 0.5, "Balanced"),
    ("agent_4", 0.3, 0.7, "LowCoop"),
    ("agent_5", 0.1, 0.9, "Selfish"),
]

SHOCK_SCHEDULE = build_mock_dev_shock_schedule()

# results.db'ye yazmayı engelle — test DB'sine yönlendir
db.RESULTS_DB_PATH = RESULTS_DB
referee_node.RESULTS_DB_PATH = RESULTS_DB


def _build_graph(checkpointer):
    workflow = StateGraph(SimulationState)
    workflow.add_node("agent_fanout", run_mock_agent_fanout)
    workflow.add_node("referee", referee_node.run_referee)
    workflow.set_entry_point("agent_fanout")
    workflow.add_edge("agent_fanout", "referee")
    workflow.add_conditional_edges(
        "referee",
        lambda state: END if state.is_terminated else "agent_fanout",
    )
    return workflow.compile(checkpointer=checkpointer)


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
        run_id=THREAD_ID,
        max_rounds=MAX_ROUNDS,
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


def _pool_current(state: dict) -> float:
    env = state["environment"]
    return env.pool_current if hasattr(env, "pool_current") else env["pool_current"]


def _metrics_to_dicts(state: dict) -> list[dict]:
    return [
        m.model_dump() if hasattr(m, "model_dump") else dict(m)
        for m in state["metrics_history"]
    ]


def _print_phase_header(label: str) -> None:
    print(f"\n{'=' * 60}")
    print(label)
    print(f"  thread_id      = {THREAD_ID}")
    print(f"  checkpoint db  = {CHECKPOINT_DB}")
    print(f"  results db     = {RESULTS_DB} (NOT data/results.db)")
    print(f"{'=' * 60}")


async def run_first_half() -> dict:
    """Round 0–4 oynat, round_number==5 olunca recursion_limit ile dur."""
    _print_phase_header("FAZ 1 — İlk yarı (5 round, bilerek durdurulacak)")

    Path(CHECKPOINT_DB).unlink(missing_ok=True)
    Path(RESULTS_DB).unlink(missing_ok=True)

    config = {
        "configurable": {"thread_id": THREAD_ID},
        # Her round = agent_fanout + referee = 2 adım; 5 round = 10 adım
        "recursion_limit": FIRST_PHASE_ROUNDS * 2,
    }

    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app = _build_graph(checkpointer)
        initial_state = _make_initial_state()

        try:
            await app.ainvoke(initial_state, config=config)
        except GraphRecursionError:
            print("  [Beklenen] recursion_limit'e ulaşıldı — graph bilerek durduruldu.")

        snapshot = await app.aget_state(config)
        state = snapshot.values

    print("\n--- Faz 1 sonu durumu (checkpoint'e yazıldı) ---")
    print(f"  round_number           : {state['round_number']}")
    print(f"  pool_current           : {_pool_current(state):.4f}")
    print(f"  metrics_history uzunluk: {len(state['metrics_history'])}")
    print(f"  is_terminated          : {state['is_terminated']}")
    print(f"  next node (devam noktası): {snapshot.next}")

    return state


async def run_second_half() -> dict:
    """Aynı thread_id ile checkpoint'ten devam et, max_rounds=15'e kadar."""
    _print_phase_header("FAZ 2 — İkinci yarı (checkpoint'ten devam, max_rounds=15)")

    config = {"configurable": {"thread_id": THREAD_ID}}

    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        app = _build_graph(checkpointer)

        resume_snapshot = await app.aget_state(config)
        print(f"\n  Checkpoint'ten okunan round_number: {resume_snapshot.values['round_number']}")
        print(f"  Checkpoint metrics_history uzunluk : {len(resume_snapshot.values['metrics_history'])}")
        print(f"  Devam edilecek node                : {resume_snapshot.next}")

        final_state = await app.ainvoke(None, config=config)

    print("\n--- Faz 2 sonu durumu ---")
    print(f"  round_number           : {final_state['round_number']}")
    print(f"  pool_current           : {_pool_current(final_state):.4f}")
    print(f"  metrics_history uzunluk: {len(final_state['metrics_history'])}")
    print(f"  termination_reason     : {final_state['termination_reason']}")
    print(f"  is_terminated          : {final_state['is_terminated']}")

    return final_state


def verify_checkpoint(
    first_state: dict,
    second_state: dict,
) -> None:
    """Checkpoint'in gerçekten çalışıp çalışmadığını raporla."""
    print(f"\n{'=' * 60}")
    print("DOĞRULAMA RAPORU")
    print(f"{'=' * 60}")

    first_metrics = _metrics_to_dicts(first_state)
    second_metrics = _metrics_to_dicts(second_state)

    # (a) round_number 5'ten başlayıp 15'e ulaştı mı?
    resumed_from_5 = first_state["round_number"] == FIRST_PHASE_ROUNDS
    reached_15 = second_state["round_number"] == MAX_ROUNDS
    not_restarted = second_state["round_number"] != 0

    print("\n(a) round_number akışı:")
    print(f"    Faz 1 sonu round_number     : {first_state['round_number']} (beklenen: {FIRST_PHASE_ROUNDS})")
    print(f"    Faz 2 sonu round_number     : {second_state['round_number']} (beklenen: {MAX_ROUNDS})")
    print(f"    0'dan yeniden başlamadı mı? : {not_restarted}")
    print(f"    5'ten devam etti mi?        : {resumed_from_5}")
    print(f"    15'e ulaştı mı?             : {reached_15}")

    # (b) metrics_history round 0–4 aynı mı?
    overlap = min(len(first_metrics), FIRST_PHASE_ROUNDS)
    metrics_match = first_metrics[:overlap] == second_metrics[:overlap]

    print("\n(b) metrics_history round 0–4 karşılaştırması:")
    print(f"    Faz 1 metrics (ilk {overlap} round): {len(first_metrics)} kayıt")
    print(f"    Faz 2 metrics (ilk {overlap} round): aynı mı? {metrics_match}")
    if not metrics_match:
        for i in range(overlap):
            if first_metrics[i] != second_metrics[i]:
                print(f"    FARK round {i}:")
                print(f"      faz1: {first_metrics[i]}")
                print(f"      faz2: {second_metrics[i]}")

    # (c) final durum
    completed = (
        second_state["round_number"] == MAX_ROUNDS
        and second_state["termination_reason"] == "completed"
    )

    print("\n(c) final durum:")
    print(f"    round_number == {MAX_ROUNDS}        : {second_state['round_number'] == MAX_ROUNDS}")
    print(f"    termination_reason == 'completed' : {second_state['termination_reason'] == 'completed'}")

    all_ok = resumed_from_5 and reached_15 and not_restarted and metrics_match and completed

    print(f"\n{'=' * 60}")
    if all_ok:
        print("SONUÇ: CHECKPOINT ÇALIŞIYOR — state kalıcı olarak korundu ve devam edildi.")
    else:
        print("SONUÇ: CHECKPOINT ÇALIŞMIYOR veya BEKLENMEYEN DURUM — yukarıdaki maddeleri inceleyin.")
        if not resumed_from_5 or not not_restarted:
            print("  → round_number 0'dan başladı veya 5'te durmadı; checkpoint resume başarısız.")
        if not metrics_match:
            print("  → metrics_history round 0–4 farklı; state checkpoint'ten gelmemiş olabilir.")
        if not completed:
            print("  → Simülasyon 15 round'ta 'completed' ile bitmedi.")
    print(f"{'=' * 60}\n")


async def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Deserializing unregistered type",
        category=UserWarning,
    )

    print("LangGraph SqliteSaver Checkpoint Testi")
    print(f"  max_rounds (hedef) = {MAX_ROUNDS}")
    print(f"  faz 1 durma noktası = round_number {FIRST_PHASE_ROUNDS} (round 0–{FIRST_PHASE_ROUNDS - 1} oynandıktan sonra)")

    first_state = await run_first_half()
    second_state = await run_second_half()
    verify_checkpoint(first_state, second_state)


if __name__ == "__main__":
    asyncio.run(main())
