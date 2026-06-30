"""5 agent'ın tamamı gerçek Anthropic LLM ile ratio doğrulama koşusu (Bölüm 10.3)."""

import asyncio
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

from langgraph.graph import END, StateGraph

from agents.decision_agent import (
    _INPUT_COST_PER_M,
    _OUTPUT_COST_PER_M,
    reset_token_usage,
    run_agent_fanout,
    token_usage,
)
from core.config import settings
from core.database import RESULTS_DB_PATH
from core.state import AgentDecision, EnvironmentSnapshot, SimulationState, TraitProfile
from environment.shocks import build_mock_dev_shock_schedule
from referee.referee_node import run_referee

EXPERIMENT_ID = "ratio_fine_tune"
RUN_IDS = [f"ratio_fine_{i:03d}" for i in range(1, 6)]
MAX_ROUNDS = 15
COST_CAP_USD = 0.30

AGENT_PROFILES = [
    ("agent_1", 0.9, 0.1, "HighCoop"),
    ("agent_2", 0.7, 0.3, "ModerateCoop"),
    ("agent_3", 0.5, 0.5, "Balanced"),
    ("agent_4", 0.3, 0.7, "LowCoop"),
    ("agent_5", 0.1, 0.9, "Selfish"),
]


@dataclass
class RunSummary:
    run_id: str
    rounds_played: int
    termination_reason: str | None
    saw_round_7: bool
    cost_usd: float
    collapse_round: int | None = None


def _build_graph():
    workflow = StateGraph(SimulationState)
    workflow.add_node("agent_fanout", run_agent_fanout)
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


def _make_initial_state(run_id: str) -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id=EXPERIMENT_ID,
        run_id=run_id,
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


def _find_constraint_violations(decisions: list[AgentDecision]) -> list[tuple]:
    violations = []
    for d in decisions:
        if d.extraction_amount > d.declared_max:
            violations.append(
                (
                    d.round_number,
                    d.agent_id,
                    d.extraction_amount,
                    d.declared_max,
                )
            )
    return violations


def _print_agent_summaries(decisions: list[AgentDecision], run_id: str) -> None:
    by_agent: dict[str, list[AgentDecision]] = defaultdict(list)
    for d in decisions:
        by_agent[d.agent_id].append(d)

    print(f"\n--- [{run_id}] Agent bazlı karar özeti ---")
    for agent_id in sorted(by_agent):
        print(f"\n  [{agent_id}]")
        for d in sorted(by_agent[agent_id], key=lambda x: x.round_number):
            justification = d.justification[:80]
            suffix = "…" if len(d.justification) > 80 else ""
            print(
                f"    round {d.round_number}: "
                f"extract={d.extraction_amount:.4f}, "
                f"declared_max={d.declared_max:.4f}, "
                f"justification={justification}{suffix}"
            )


def _print_constraint_warnings(decisions: list[AgentDecision], final_state: dict) -> None:
    violations = _find_constraint_violations(decisions)
    print("\n--- Constraint violation kontrolü ---")
    if not violations:
        print("  UYARI YOK — tüm kararlarda extraction_amount <= declared_max")
        return

    print(f"  ⚠ {len(violations)} ihlal tespit edildi (extraction_amount > declared_max):")
    for rnd, agent_id, extract, declared in violations:
        print(
            f"    round {rnd} | {agent_id}: "
            f"extraction={extract:.4f} > declared_max={declared:.4f}"
        )

    print("\n  Referee constraint_violations metriği (round bazlı):")
    for m in final_state["metrics_history"]:
        if m.constraint_violations > 0:
            print(
                f"    round {m.round_number}: "
                f"constraint_violations={m.constraint_violations}"
            )


def _saw_round_7(final_state: dict) -> bool:
    metrics = final_state["metrics_history"]
    return any(m.round_number >= 7 for m in metrics)


def _rounds_played(final_state: dict) -> int:
    return len(final_state["metrics_history"])


async def _run_single(run_id: str, app) -> RunSummary:
    reset_token_usage()
    cost_before = token_usage.estimated_cost_usd()

    print(f"\n{'=' * 60}")
    print(f"Koşu: {run_id}")
    print(f"  experiment_id = {EXPERIMENT_ID}")
    print(f"  max_rounds    = {MAX_ROUNDS}")
    print(f"  agent sayısı  = {settings.AGENT_COUNT} (hepsi gerçek LLM)")
    print(f"  model         = {settings.ANTHROPIC_MODEL}")
    print(f"  ratio         = {settings.EXTRACTION_LIMIT_RATIO}")
    print(f"{'=' * 60}")

    final_state = await app.ainvoke(_make_initial_state(run_id))
    decisions = final_state["round_decisions"]

    print("\n--- Simülasyon sonucu ---")
    print(f"  round_number       : {final_state['round_number']}")
    print(f"  rounds_played      : {_rounds_played(final_state)}")
    print(f"  termination_reason : {final_state['termination_reason']}")
    print(f"  pool_current       : {final_state['environment'].pool_current:.4f}")
    print(f"  saw_round_7        : {_saw_round_7(final_state)}")

    _print_agent_summaries(decisions, run_id)
    _print_constraint_warnings(decisions, final_state)

    cost_usd = token_usage.estimated_cost_usd() - cost_before
    print("\n--- Token kullanımı ve tahmini maliyet ---")
    print(f"  input_tokens  : {token_usage.input_tokens}")
    print(f"  output_tokens : {token_usage.output_tokens}")
    print(f"  tahmini maliyet: ${cost_usd:.6f} USD")

    return RunSummary(
        run_id=run_id,
        rounds_played=_rounds_played(final_state),
        termination_reason=final_state["termination_reason"],
        saw_round_7=_saw_round_7(final_state),
        cost_usd=cost_usd,
        collapse_round=(
            _rounds_played(final_state)
            if final_state["termination_reason"] == "collapse"
            else None
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
    conn.close()
    print(f"\n--- DB özeti ({RESULTS_DB_PATH}) ---")
    print(f"  experiment_id          : {EXPERIMENT_ID}")
    print(f"  metrics_snapshots satır: {metrics}")
    print(f"  agent_decisions satır  : {decisions}")


def _print_summary_table(summaries: list[RunSummary]) -> None:
    print(f"\n{'=' * 80}")
    print(f"ÖZET TABLO — {EXPERIMENT_ID} (EXTRACTION_LIMIT_RATIO={settings.EXTRACTION_LIMIT_RATIO})")
    print(f"{'=' * 80}")
    header = (
        f"{'run_id':<20} {'rounds':>6} {'termination':<12} "
        f"{'R7?':>5} {'collapse_rnd':>12} {'maliyet (USD)':>14}"
    )
    print(header)
    print("-" * len(header))
    total_cost = 0.0
    collapse_rounds: list[int] = []
    for s in summaries:
        total_cost += s.cost_usd
        if s.collapse_round is not None:
            collapse_rounds.append(s.collapse_round)
        collapse_str = str(s.collapse_round) if s.collapse_round is not None else "—"
        print(
            f"{s.run_id:<20} {s.rounds_played:>6} "
            f"{s.termination_reason or '—':<12} "
            f"{'evet' if s.saw_round_7 else 'hayır':>5} "
            f"{collapse_str:>12} "
            f"${s.cost_usd:>13.6f}"
        )
    print("-" * len(header))
    print(f"{'TOPLAM':<20} {'':>6} {'':<12} {'':>5} {'':>12} ${total_cost:>13.6f}")
    if collapse_rounds:
        avg = sum(collapse_rounds) / len(collapse_rounds)
        print(
            f"\n  collapse_round (collapse olanlar): "
            f"min={min(collapse_rounds)}, max={max(collapse_rounds)}, "
            f"ort={avg:.2f} (n={len(collapse_rounds)})"
        )
    print(
        f"\n  (fiyatlandırma: ${_INPUT_COST_PER_M:.2f}/M input, "
        f"${_OUTPUT_COST_PER_M:.2f}/M output — Claude Haiku 4.5)"
    )


async def main():
    print("Ratio fine-tune — tam gerçek-agent Anthropic test koşusu")
    print(f"  experiment_id = {EXPERIMENT_ID}")
    print(f"  run_ids       = {', '.join(RUN_IDS)}")
    print(f"  max_rounds    = {MAX_ROUNDS}")
    print(f"  EXTRACTION_LIMIT_RATIO = {settings.EXTRACTION_LIMIT_RATIO}")
    print(f"  model         = {settings.ANTHROPIC_MODEL}")
    print(f"  temperature   = {settings.TEMPERATURE}")
    print(f"  maliyet sınırı: ${COST_CAP_USD:.2f}")

    app = _build_graph()
    summaries: list[RunSummary] = []
    total_cost = 0.0
    for run_id in RUN_IDS:
        if total_cost >= COST_CAP_USD:
            print(
                f"\n⚠ Maliyet sınırı (${COST_CAP_USD:.2f}) aşıldı — "
                f"kalan koşular atlandı ({run_id} ve sonrası)."
            )
            break
        summary = await _run_single(run_id, app)
        summaries.append(summary)
        total_cost += summary.cost_usd
        if total_cost > COST_CAP_USD:
            print(
                f"\n⚠ Toplam maliyet ${total_cost:.4f} — "
                f"${COST_CAP_USD:.2f} sınırını aştı. Kalan koşular durduruldu."
            )
            break

    _print_db_summary()
    _print_summary_table(summaries)


if __name__ == "__main__":
    asyncio.run(main())
