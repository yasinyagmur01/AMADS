"""Phase 3 quality gate: 10 CPR + 10 bargaining Groq calls (English prompts).

Not an analysis experiment — validation only under experiment_id=groq_smoke_test.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.decision_agent import _decide
from core.state import AgentInputView, EnvironmentSnapshot, TraitProfile
from scenarios.bargaining.agents import _proposer_decide, _responder_decide
from scenarios.bargaining.state import BargainingProposerView, BargainingResponderView

EXPERIMENT_ID = "groq_smoke_test"
MAX_EXTRACT = 12.0
PIE = 100.0


def _cpr_view(coop: float, risk: float, i: int) -> AgentInputView:
    return AgentInputView(
        own_trait=TraitProfile(
            agent_id="agent_1",
            cooperation_assigned=coop,
            risk_tolerance_assigned=risk,
            profile_label=f"smoke_{i}",
        ),
        environment=EnvironmentSnapshot(
            pool_current=100.0,
            pool_capacity=100.0,
            regen_rate=1.15,
            max_extractable_this_round=MAX_EXTRACT,
            round_number=1,
            is_collapsed=False,
        ),
        round_number=1,
    )


def _proposer_view(coop: float, risk: float, i: int) -> BargainingProposerView:
    return BargainingProposerView(
        experiment_id=EXPERIMENT_ID,
        own_trait=TraitProfile(
            agent_id="proposer",
            cooperation_assigned=coop,
            risk_tolerance_assigned=risk,
            profile_label=f"p_{i}",
        ),
        env_snapshot=EnvironmentSnapshot(
            pool_current=PIE,
            pool_capacity=PIE,
            regen_rate=1.0,
            max_extractable_this_round=PIE,
            round_number=1,
            is_collapsed=False,
        ),
        time_pressure=0.3,
        resource_scarcity=0.3,
        round_number=1,
        pie_size=PIE,
        recent_history=[],
    )


def _responder_view(coop: float, risk: float, keep: float, i: int) -> BargainingResponderView:
    return BargainingResponderView(
        experiment_id=EXPERIMENT_ID,
        own_trait=TraitProfile(
            agent_id="responder",
            cooperation_assigned=coop,
            risk_tolerance_assigned=risk,
            profile_label=f"r_{i}",
        ),
        env_snapshot=EnvironmentSnapshot(
            pool_current=PIE,
            pool_capacity=PIE,
            regen_rate=1.0,
            max_extractable_this_round=PIE,
            round_number=1,
            is_collapsed=False,
        ),
        time_pressure=0.3,
        resource_scarcity=0.3,
        round_number=1,
        pie_size=PIE,
        current_offer_keep=keep,
        offer_to_me=PIE - keep,
        recent_history=[],
    )


def _diversity(texts: list[str]) -> dict:
    hashes = [hashlib.md5(t.encode()).hexdigest() for t in texts]
    counts = Counter(hashes)
    return {
        "n": len(texts),
        "unique": len(counts),
        "max_repeat": max(counts.values()) if counts else 0,
    }


async def run_cpr(n: int = 10) -> dict:
    ok = 0
    extractions: list[float] = []
    justifications: list[str] = []
    errors: list[str] = []
    for i in range(n):
        coop = 0.2 if i % 2 == 0 else 0.8
        try:
            d = await _decide(_cpr_view(coop, 0.2, i), experiment_id=EXPERIMENT_ID)
            in_range = 0.0 <= d.extraction_amount <= MAX_EXTRACT
            has_just = bool(d.justification and d.justification.strip())
            if in_range and has_just and d.declared_max is not None:
                ok += 1
            else:
                errors.append(f"cpr[{i}] range/just fail: {d}")
            extractions.append(float(d.extraction_amount))
            justifications.append(d.justification)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cpr[{i}] {type(exc).__name__}: {exc}")
        await asyncio.sleep(0.4)
    return {
        "parse_ok": ok,
        "n": n,
        "parse_fail_rate": 1 - ok / n,
        "extractions": extractions,
        "diversity": _diversity(justifications),
        "errors": errors,
    }


async def run_bargaining(n: int = 10) -> dict:
    ok = 0
    keeps: list[float] = []
    just_p: list[str] = []
    just_r: list[str] = []
    errors: list[str] = []
    for i in range(n):
        coop = 0.2 if i % 2 == 0 else 0.8
        try:
            offer = await _proposer_decide(_proposer_view(coop, 0.2, i))
            keep = float(offer.keep_amount)
            keep_ok = 0.0 <= keep <= PIE
            dec = await _responder_decide(_responder_view(0.5, 0.5, keep, i))
            if keep_ok and offer.justification and dec.justification is not None:
                ok += 1
            else:
                errors.append(f"barg[{i}] fail offer={offer} dec={dec}")
            keeps.append(keep)
            just_p.append(offer.justification)
            just_r.append(dec.justification)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"barg[{i}] {type(exc).__name__}: {exc}")
        await asyncio.sleep(0.4)
    return {
        "parse_ok": ok,
        "n": n,
        "parse_fail_rate": 1 - ok / n,
        "keeps": keeps,
        "diversity_proposer": _diversity(just_p),
        "diversity_responder": _diversity(just_r),
        "errors": errors,
    }


def _pass(report: dict, label: str) -> bool:
    fail_rate = report["parse_fail_rate"]
    print(f"\n=== {label} ===")
    print(f"  parse_ok={report['parse_ok']}/{report['n']} fail_rate={fail_rate:.0%}")
    if "extractions" in report:
        xs = report["extractions"]
        print(f"  extractions: min={min(xs):.2f} max={max(xs):.2f} mean={sum(xs)/len(xs):.2f}")
        print(f"  justification diversity: {report['diversity']}")
    if "keeps" in report:
        ks = report["keeps"]
        print(f"  keep_amount: min={min(ks):.2f} max={max(ks):.2f} mean={sum(ks)/len(ks):.2f}")
        print(f"  proposer diversity: {report['diversity_proposer']}")
        print(f"  responder diversity: {report['diversity_responder']}")
    if report["errors"]:
        print("  errors:")
        for e in report["errors"][:5]:
            print(f"    - {e}")
    # Fail if >20% parse failure or fully identical justifications
    if fail_rate > 0.20:
        print("  VERDICT: FAIL (parse)")
        return False
    div = report.get("diversity") or report.get("diversity_proposer")
    if div and div["unique"] < 2 and div["n"] >= 5:
        print("  VERDICT: FAIL (degenerate identical justifications)")
        return False
    print("  VERDICT: PASS")
    return True


async def main() -> None:
    cpr = await run_cpr(10)
    barg = await run_bargaining(10)
    ok_cpr = _pass(cpr, "CPR decision_agent")
    ok_barg = _pass(barg, "Bargaining proposer+responder")
    if not (ok_cpr and ok_barg):
        raise SystemExit(1)
    print("\nGATE 3: PASS")


if __name__ == "__main__":
    asyncio.run(main())
