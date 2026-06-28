"""
Mock trait fidelity baseline — LLM çağrısı yok.

9 koşul (3×3 coop×risk) mock agent ile simüle edilir; beklenen pozitif
coop→cooperation_score yönü doğrulanır. full_experiment_v1 LLM sonuçlarıyla
karşılaştırma için trait_fidelity.py çıktısına bakın.

Usage:
    venv/bin/python analysis/trait_fidelity_mock_baseline.py
"""

from __future__ import annotations

import os

os.environ["LANGSMITH_TRACING"] = "false"

import asyncio
import sys
from itertools import product
from pathlib import Path

from langgraph.graph import END, StateGraph

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import referee.referee_node as referee_node
from agents.mock_agent import run_mock_agent_fanout
from core.config import settings
from core.state import EnvironmentSnapshot, SimulationState, TraitProfile
from environment.shocks import build_mock_dev_shock_schedule

try:
    from scipy.stats import pearsonr
except ImportError as exc:
    raise RuntimeError(f"scipy gerekli: pip install scipy ({exc})") from exc

TRAIT_LEVELS = [0.2, 0.5, 0.8]
AGENT_IDS = [f"agent_{i}" for i in range(1, settings.AGENT_COUNT + 1)]
MAX_ROUNDS = 15


def _noop_save(_state, _db_path) -> None:
    pass


referee_node.save_round_to_db = _noop_save


def _build_graph():
    workflow = StateGraph(SimulationState)
    workflow.add_node("agent_fanout", run_mock_agent_fanout)
    workflow.add_node("referee", referee_node.run_referee)
    workflow.set_entry_point("agent_fanout")
    workflow.add_edge("agent_fanout", "referee")
    workflow.add_conditional_edges(
        "referee",
        lambda state: END if state.is_terminated else "agent_fanout",
    )
    return workflow.compile()


def _make_traits(coop: float, risk: float) -> dict[str, TraitProfile]:
    return {
        agent_id: TraitProfile(
            agent_id=agent_id,
            cooperation_assigned=coop,
            risk_tolerance_assigned=risk,
            profile_label="MockBaseline",
        )
        for agent_id in AGENT_IDS
    }


def _make_state(coop: float, risk: float) -> SimulationState:
    pool = 100.0
    return SimulationState(
        experiment_id="mock_baseline",
        run_id=f"mock_c{coop}_r{risk}",
        max_rounds=MAX_ROUNDS,
        agent_traits=_make_traits(coop, risk),
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


async def _run_all() -> list[tuple[float, float, float, float, int, str | None]]:
    app = _build_graph()
    results = []
    for coop, risk in product(TRAIT_LEVELS, TRAIT_LEVELS):
        final = await app.ainvoke(_make_state(coop, risk))
        mh = final["metrics_history"]
        if not mh:
            continue
        round0 = mh[0]
        avg_coop = sum(m.cooperation_score_avg for m in mh) / len(mh)
        fraction = 0.10 + 0.25 * (1.0 - coop)  # mock formül (sabit)
        results.append(
            (
                coop,
                risk,
                fraction,
                round0.cooperation_score_avg,
                len(mh),
                final["termination_reason"],
            )
        )
    return results


def main() -> None:
    rows = asyncio.run(_run_all())

    print("=" * 72)
    print("MOCK TRAIT FIDELITY BASELINE (LLM yok)")
    print("=" * 72)
    print("  mock_agent: fraction = 0.10 + 0.25×(1 − coop); risk KULLANILMIYOR")
    print()

    print(f"  {'coop':>5} {'risk':>5} | {'fraction':>8} {'coop_scr0':>9} {'rounds':>6} term")
    print("  " + "-" * 48)
    for coop, risk, frac, score0, n_rounds, term in rows:
        print(
            f"  {coop:5.1f} {risk:5.1f} | {frac:8.3f} {score0:9.3f} {n_rounds:6d} {term or '—'}"
        )

    coop_x = [r[0] for r in rows]
    score_y = [r[3] for r in rows]
    frac_y = [r[2] for r in rows]
    risk_x = [r[1] for r in rows]

    r_coop, p_coop = pearsonr(coop_x, score_y)
    r_frac, _ = pearsonr(coop_x, frac_y)
    r_risk, p_risk = pearsonr(risk_x, score_y)

    print()
    print(f"  coop → cooperation_score (round 0): r={r_coop:.3f}, p={p_coop:.2e}")
    print(f"  coop → extraction_fraction (analitik): r={r_frac:.3f} (negatif beklenir: coop↑ frac↓)")
    print(f"  risk → cooperation_score (round 0):   r={r_risk:.3f}, p={p_risk:.2e} (≈0 beklenir)")
    print()
    print("BEKLENEN YÖN (sistem niyeti):")
    print("  coop ↑ → cooperation_score ↑  (pozitif fidelity)")
    print("  risk   → metrik etkisiz       (mock risk kullanmıyor)")
    print()
    print("LLM full_experiment_v1 ile karşılaştır:")
    print("  venv/bin/python analysis/trait_fidelity.py --max-round 0")
    print()
    if r_coop > 0:
        print("  ✓ Mock baseline pozitif coop→score yönünü doğruluyor.")
    else:
        print("  ✗ Mock baseline beklenmeyen yön — mock_agent kodunu kontrol edin.")


if __name__ == "__main__":
    main()
