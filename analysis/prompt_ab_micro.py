"""
Prompt A/B micro: cooperation_assigned {0.2, 0.8} × 5 tekrar, tek agent, tek round.

risk_tolerance_assigned=0.2 sabit; her koşuda decision_agent.run_agent_fanout
doğrudan çağrılır (mock yok, gerçek Haiku LLM). Sonuçlar yalnızca terminale
yazılır; data/results.db'ye dokunulmaz.

Usage (repo root):
    python analysis/prompt_ab_micro.py

Tahmini maliyet: ~10 çağrı × $0.002 ≈ $0.02 (güvenlik sınırı $0.10).
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.decision_agent import (
    _INPUT_COST_PER_M,
    _OUTPUT_COST_PER_M,
    reset_token_usage,
    run_agent_fanout,
    token_usage,
)
from core.config import settings
from core.state import AgentDecision, EnvironmentSnapshot, SimulationState, TraitProfile

COOPERATION_VALUES = (0.2, 0.8)
RUNS_PER_VALUE = 5
RISK_TOLERANCE = 0.2
COST_CAP_USD = 0.10
AGENT_ID = "agent_1"


def _make_state(cooperation: float, run_index: int) -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id="_scratch",
        run_id=f"micro_{cooperation:.1f}_{run_index}",
        max_rounds=1,
        agent_traits={
            AGENT_ID: TraitProfile(
                agent_id=AGENT_ID,
                cooperation_assigned=cooperation,
                risk_tolerance_assigned=RISK_TOLERANCE,
                profile_label=f"coop_{cooperation:.1f}",
            )
        },
        shock_schedule=[],
        environment=EnvironmentSnapshot(
            pool_current=pool,
            pool_capacity=pool,
            regen_rate=1.15,
            max_extractable_this_round=pool * settings.EXTRACTION_LIMIT_RATIO,
            round_number=0,
            is_collapsed=False,
        ),
    )


async def main() -> None:
    reset_token_usage()

    print("Prompt A/B micro (tek agent, tek round, gerçek LLM)")
    print(f"  model                 : {settings.ANTHROPIC_MODEL}")
    print(f"  temperature           : {settings.TEMPERATURE}")
    print(f"  risk_tolerance_assigned: {RISK_TOLERANCE} (sabit)")
    print(f"  cooperation_assigned  : {list(COOPERATION_VALUES)} × {RUNS_PER_VALUE} tekrar")
    print(f"  maliyet güvenlik sınırı: ${COST_CAP_USD:.2f}")
    print(f"  DB                    : yazılmıyor\n")

    extractions: dict[float, list[float]] = {coop: [] for coop in COOPERATION_VALUES}
    decisions_by_coop: dict[float, list[tuple[int, AgentDecision]]] = {
        coop: [] for coop in COOPERATION_VALUES
    }
    stopped_early = False

    for cooperation in COOPERATION_VALUES:
        for run_index in range(1, RUNS_PER_VALUE + 1):
            if token_usage.estimated_cost_usd() >= COST_CAP_USD:
                print(
                    f"\n⚠ Maliyet sınırı (${COST_CAP_USD:.2f}) aşıldı — "
                    "kalan çağrılar atlandı."
                )
                stopped_early = True
                break

            state = _make_state(cooperation, run_index)
            result = await run_agent_fanout(state)
            decision = result["round_decisions"][-1]

            extractions[cooperation].append(decision.extraction_amount)
            decisions_by_coop[cooperation].append((run_index, decision))
            print(
                f"  coop={cooperation:.1f} run={run_index}: "
                f"extraction_amount={decision.extraction_amount:.4f}, "
                f"declared_max={decision.declared_max:.4f}"
            )
            print(f"    justification: {decision.justification}")

        if stopped_early:
            break

    high_coop_runs = decisions_by_coop.get(0.8, [])
    if high_coop_runs:
        print("\n--- cooperation=0.8 tam gerekçeler ---")
        for run_index, decision in high_coop_runs:
            print(f"\n  run {run_index}:")
            print(f"    extraction_amount={decision.extraction_amount:.4f}")
            print(f"    declared_max={decision.declared_max:.4f}")
            print(f"    justification: {decision.justification}")

    print("\n--- Grup ortalamaları (extraction_amount) ---")
    means: dict[float, float] = {}
    for cooperation in COOPERATION_VALUES:
        values = extractions[cooperation]
        if not values:
            print(f"  cooperation={cooperation:.1f}: (veri yok)")
            continue
        mean_val = statistics.mean(values)
        means[cooperation] = mean_val
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        print(
            f"  cooperation={cooperation:.1f}: "
            f"n={len(values)}, mean={mean_val:.4f}, stdev={stdev:.4f}"
        )

    if 0.2 in means and 0.8 in means:
        diff = means[0.8] - means[0.2]
        print(
            f"\n  Ortalama fark (coop=0.8 − coop=0.2): {diff:+.4f}"
        )
        if diff < 0:
            print("  → Yüksek işbirliği grubu daha az çekim yaptı.")
        elif diff > 0:
            print("  → Yüksek işbirliği grubu daha fazla çekim yaptı.")
        else:
            print("  → İki grup ortalaması eşit.")

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
