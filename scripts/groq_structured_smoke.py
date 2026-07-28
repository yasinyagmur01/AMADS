"""Throwaway Groq structured-output smoke test (5 calls, not an analysis run)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.llm_providers import DEFAULT_GROQ_MODEL, call_agent, resolve_llm_config
from core.state import AgentDecision
from scenarios.bargaining.state import ProposerOffer, ResponderDecision

EXPERIMENT_ID = "groq_smoke_test"


async def _one_call(i: int) -> tuple[str, bool, str]:
    schema_cycle = [AgentDecision, ProposerOffer, ResponderDecision, AgentDecision, ProposerOffer]
    schema = schema_cycle[i]
    if schema is AgentDecision:
        messages = [
            (
                "system",
                "You are a test agent. Return structured fields only.",
            ),
            (
                "human",
                "Return agent_id='agent_1', round_number=1, "
                "extraction_amount between 0 and 10 (prefer 3.5), "
                "justification short, declared_max >= extraction_amount.",
            ),
        ]
    elif schema is ProposerOffer:
        messages = [
            ("system", "You are a bargaining proposer test agent."),
            (
                "human",
                "Return keep_amount between 0 and 100 (prefer 55) and a short justification.",
            ),
        ]
    else:
        messages = [
            ("system", "You are a bargaining responder test agent."),
            (
                "human",
                "Return accept true/false (prefer true) and a short justification.",
            ),
        ]
    try:
        result = await call_agent(messages, schema, experiment_id=EXPERIMENT_ID)
        parsed = result.parsed
        return schema.__name__, True, repr(parsed)[:120]
    except Exception as exc:  # noqa: BLE001 — smoke report
        return schema.__name__, False, f"{type(exc).__name__}: {exc}"


async def main() -> None:
    cfg = resolve_llm_config(EXPERIMENT_ID)
    print(f"Groq smoke: provider={cfg.provider} model={cfg.model}")
    assert cfg.provider == "groq"
    assert cfg.model == DEFAULT_GROQ_MODEL

    results = []
    for i in range(5):
        name, ok, detail = await _one_call(i)
        results.append((name, ok, detail))
        status = "OK" if ok else "FAIL"
        print(f"  [{i+1}/5] {name}: {status} — {detail}")
        await asyncio.sleep(0.5)  # gentle on free-tier RPM

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\nRESULT: {n_ok}/5 parse success")
    if n_ok < 5:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
