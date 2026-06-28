"""
EXTRACTION_LIMIT_RATIO kalibrasyon testi.

Tüm ajanlar mock (LLM yok). Her ratio için 10 rastgele koşu; sonuçlar
yalnızca terminale ve data/calibration_results.csv'ye yazılır.
data/results.db'ye dokunulmaz.
"""

import asyncio
import csv
import os
from dataclasses import dataclass
from pathlib import Path

from langgraph.graph import END, StateGraph

import referee.referee_node as referee_node
from agents.mock_agent import run_random_mock_agent_fanout
from core.config import settings
from core.state import EnvironmentSnapshot, SimulationState, TraitProfile
from environment.shocks import build_mock_dev_shock_schedule

EXPERIMENT_ID = "calibration"
MAX_ROUNDS = 15
SHOCK_ROUND = 7
RUNS_PER_RATIO = 40
RATIOS = [0.30, 0.32, 0.34]
CALIBRATION_CSV = Path("data/calibration_results.csv")

AGENT_PROFILES = [
    ("agent_1", 0.9, 0.1, "HighCoop"),
    ("agent_2", 0.7, 0.3, "ModerateCoop"),
    ("agent_3", 0.5, 0.5, "Balanced"),
    ("agent_4", 0.3, 0.7, "LowCoop"),
    ("agent_5", 0.1, 0.9, "Selfish"),
]

SHOCK_SCHEDULE = build_mock_dev_shock_schedule()


def _noop_save_round_to_db(_state, _db_path) -> None:
    pass


referee_node.save_round_to_db = _noop_save_round_to_db


def _build_graph():
    workflow = StateGraph(SimulationState)
    workflow.add_node("agent_fanout", run_random_mock_agent_fanout)
    workflow.add_node("referee", referee_node.run_referee)
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


def _make_initial_state(ratio: float, run_id: str) -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
        max_rounds=MAX_ROUNDS,
        agent_traits=_make_traits(),
        shock_schedule=SHOCK_SCHEDULE,
        environment=EnvironmentSnapshot(
            pool_current=pool,
            pool_capacity=pool,
            regen_rate=1.15,
            max_extractable_this_round=pool * ratio,
            round_number=0,
            is_collapsed=False,
        ),
    )


def _saw_shock_round(metrics_history: list) -> bool:
    return any(m.round_number >= SHOCK_ROUND for m in metrics_history)


@dataclass
class RunResult:
    ratio: float
    run_index: int
    rounds_played: int
    saw_shock: bool
    completed: bool
    termination_reason: str | None


@dataclass
class RatioSummary:
    ratio: float
    avg_rounds: float
    shock_hits: int
    shock_rate_pct: float
    completed: int
    completed_rate_pct: float


async def _run_single(app, ratio: float, run_index: int) -> RunResult:
    settings.EXTRACTION_LIMIT_RATIO = ratio
    run_id = f"calib_{ratio:.2f}_{run_index:02d}"
    final_state = await app.ainvoke(
        _make_initial_state(ratio, run_id),
        config={"configurable": {"thread_id": run_id}},
    )
    metrics = final_state["metrics_history"]
    return RunResult(
        ratio=ratio,
        run_index=run_index,
        rounds_played=len(metrics),
        saw_shock=_saw_shock_round(metrics),
        completed=final_state["termination_reason"] == "completed",
        termination_reason=final_state["termination_reason"],
    )


def _summarize_ratio(ratio: float, results: list[RunResult]) -> RatioSummary:
    n = len(results)
    shock_hits = sum(1 for r in results if r.saw_shock)
    completed = sum(1 for r in results if r.completed)
    avg_rounds = sum(r.rounds_played for r in results) / n
    return RatioSummary(
        ratio=ratio,
        avg_rounds=avg_rounds,
        shock_hits=shock_hits,
        shock_rate_pct=100.0 * shock_hits / n,
        completed=completed,
        completed_rate_pct=100.0 * completed / n,
    )


def _pick_recommendation(summaries: list[RatioSummary]) -> RatioSummary | None:
    """Round 7 görme %50–70, tamamlama %100 değil."""
    candidates = [
        s
        for s in summaries
        if 50.0 <= s.shock_rate_pct <= 70.0 and s.completed < RUNS_PER_RATIO
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda s: abs(s.shock_rate_pct - 60.0),
    )


def _print_table(summaries: list[RatioSummary]) -> None:
    header = (
        f"{'Ratio':>6} | {'Avg Rounds':>10} | "
        f"{'Saw R7':>8} | {'R7 %':>6} | "
        f"{'Done 15':>8} | {'Done %':>7}"
    )
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}")
    print("EXTRACTION_LIMIT_RATIO KALİBRASYON ÖZETİ")
    print(f"  Koşu sayısı/ratio: {RUNS_PER_RATIO}  |  max_rounds: {MAX_ROUNDS}  |  şok round: {SHOCK_ROUND}")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for s in summaries:
        print(
            f"{s.ratio:6.2f} | {s.avg_rounds:10.2f} | "
            f"{s.shock_hits:>3}/{RUNS_PER_RATIO:<4} | {s.shock_rate_pct:5.1f}% | "
            f"{s.completed:>3}/{RUNS_PER_RATIO:<4} | {s.completed_rate_pct:5.1f}%"
        )
    print(sep)

    rec = _pick_recommendation(summaries)
    print("\nÖneri (R7 %50–70, tamamlama < %100):")
    if rec:
        print(
            f"  EXTRACTION_LIMIT_RATIO={rec.ratio:.2f} "
            f"(R7={rec.shock_rate_pct:.0f}%, tamamlama={rec.completed_rate_pct:.0f}%)"
        )
    else:
        print("  Hedef aralıkta ratio bulunamadı — tabloya göre manuel seçim gerekir.")


def _write_csv(summaries: list[RatioSummary]) -> None:
    CALIBRATION_CSV.parent.mkdir(parents=True, exist_ok=True)
    with CALIBRATION_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "extraction_limit_ratio",
                "avg_rounds_played",
                "saw_shock_round_7",
                "shock_rate_pct",
                "completed_15",
                "completed_rate_pct",
                "runs_per_ratio",
                "max_rounds",
                "shock_round",
            ],
        )
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "extraction_limit_ratio": s.ratio,
                    "avg_rounds_played": round(s.avg_rounds, 2),
                    "saw_shock_round_7": s.shock_hits,
                    "shock_rate_pct": round(s.shock_rate_pct, 1),
                    "completed_15": s.completed,
                    "completed_rate_pct": round(s.completed_rate_pct, 1),
                    "runs_per_ratio": RUNS_PER_RATIO,
                    "max_rounds": MAX_ROUNDS,
                    "shock_round": SHOCK_ROUND,
                }
            )
    print(f"\nCSV yazıldı: {CALIBRATION_CSV}")


async def main() -> None:
    settings.LANGSMITH_TRACING = False
    os.environ["LANGSMITH_TRACING"] = "false"

    print("EXTRACTION_LIMIT_RATIO kalibrasyon testi")
    print(f"  Ratios: {RATIOS}")
    print(f"  Runs per ratio: {RUNS_PER_RATIO}")
    print("  DB yazımı: KAPALI (data/results.db dokunulmaz)")

    all_summaries: list[RatioSummary] = []
    app = _build_graph()

    for ratio in RATIOS:
        print(f"\n--- ratio={ratio:.2f} ---")
        results: list[RunResult] = []
        for i in range(RUNS_PER_RATIO):
            result = await _run_single(app, ratio, i)
            results.append(result)
            print(
                f"  run {i + 1:2d}: rounds={result.rounds_played:2d}  "
                f"R7={'yes' if result.saw_shock else 'no ':3s}  "
                f"done={'yes' if result.completed else 'no ':3s}  "
                f"reason={result.termination_reason}"
            )
        all_summaries.append(_summarize_ratio(ratio, results))

    _print_table(all_summaries)
    _write_csv(all_summaries)


if __name__ == "__main__":
    asyncio.run(main())
