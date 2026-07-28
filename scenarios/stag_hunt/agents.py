"""Stag Hunt agents (LLM decision only).

English symbolic-trait prompts (see .cursorrules: new experiment_ids use
English prompts). Both agents decide simultaneously — the graph fans them
out in parallel via asyncio.gather in a single "agents" node (core rule:
agent decisions in a round are independent and parallel).

Trait interpretation for this scenario (documented here since it is an
interpretive choice, not a literal payoff mapping):
  - risk_tolerance: 0 = always hunt hare (safe, risk-averse),
                    1 = always hunt stag (risky, coordination-seeking)
  - cooperation: interpreted as trust/expectation that the partner will
                 choose stag — 0 = expect partner to play hare,
                 1 = expect partner to play stag. This is distinct from
                 risk_tolerance: an agent can *want* to coordinate on stag
                 (high cooperation) while still being too risk-averse
                 (low risk_tolerance) to actually choose it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from anthropic import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.llm_providers import call_agent, resolve_llm_config
from scenarios.stag_hunt.state import (
    AGENT_A_ID,
    AGENT_B_ID,
    HARE_HARE,
    HARE_WHEN_PARTNER_STAGS,
    STAG_STAG,
    STAG_WHEN_PARTNER_HARES,
    StagHuntAgentView,
    StagHuntDecision,
    StagHuntState,
)

# Claude Haiku 4.5 pricing (USD per 1M tokens) — same as CPR/bargaining agents.
# Irrelevant for Groq (free tier) but kept for provider-agnostic accounting.
_INPUT_COST_PER_M = 1.00
_OUTPUT_COST_PER_M = 5.00


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, inp: int, out: int) -> None:
        self.input_tokens += inp
        self.output_tokens += out

    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens * _INPUT_COST_PER_M / 1_000_000
            + self.output_tokens * _OUTPUT_COST_PER_M / 1_000_000
        )


token_usage = TokenUsage()


def reset_token_usage() -> None:
    token_usage.input_tokens = 0
    token_usage.output_tokens = 0


def _record_usage(raw_message) -> None:
    if raw_message is None:
        return
    usage = getattr(raw_message, "usage_metadata", None) or {}
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    if inp is None or out is None:
        meta = getattr(raw_message, "response_metadata", None) or {}
        api_usage = meta.get("usage") or {}
        inp = inp if inp is not None else api_usage.get("input_tokens", 0)
        out = out if out is not None else api_usage.get("output_tokens", 0)
        if not inp:
            inp = api_usage.get("prompt_tokens", 0)
        if not out:
            out = api_usage.get("completion_tokens", 0)
    token_usage.add(int(inp or 0), int(out or 0))


def _history_summary(
    history: list[dict[str, Any]],
    *,
    own_id: str,
    limit: int = 5,
) -> str:
    if not history:
        return "No previous rounds yet."
    own_key = "agent_a_choice" if own_id == AGENT_A_ID else "agent_b_choice"
    opp_key = "agent_b_choice" if own_id == AGENT_A_ID else "agent_a_choice"
    lines = []
    for h in history[-limit:]:
        lines.append(
            f"  round {h['round_number']}: you={h[own_key]}, opponent={h[opp_key]}"
        )
    return "\n".join(lines)


def _build_system_prompt(view: StagHuntAgentView) -> str:
    t = view.own_trait
    return (
        "You are an agent playing an iterated Stag Hunt against one other "
        "agent, over repeated rounds. Each round you both choose "
        "simultaneously: hunt Stag or hunt Hare. Payoffs this round:\n"
        f"  - both hunt Stag: you get {STAG_STAG:.0f}, opponent gets {STAG_STAG:.0f} "
        "(best mutual outcome, but requires trusting the other to also pick stag)\n"
        f"  - both hunt Hare: you get {HARE_HARE:.0f}, opponent gets {HARE_HARE:.0f} "
        "(safe mutual outcome)\n"
        f"  - you hunt Hare, opponent hunts Stag: you get {HARE_WHEN_PARTNER_STAGS:.0f}, "
        f"opponent gets {STAG_WHEN_PARTNER_HARES:.0f}\n"
        f"  - you hunt Stag, opponent hunts Hare: you get {STAG_WHEN_PARTNER_HARES:.0f}, "
        f"opponent gets {HARE_WHEN_PARTNER_STAGS:.0f}\n"
        f"Your risk_tolerance_assigned value is {t.risk_tolerance_assigned:.2f} "
        "(0=always hunt hare / safe, 1=always hunt stag / risky coordination), "
        f"your cooperation_assigned value is {t.cooperation_assigned:.2f} "
        "(0=expect your partner to hunt hare, 1=expect your partner to hunt "
        "stag — i.e. your trust that coordination on stag will succeed). "
        "Make your decision in line with these tendencies, but do not repeat "
        "or explain these numbers in your output."
    )


def _build_human_prompt(view: StagHuntAgentView) -> str:
    hist = _history_summary(view.history, own_id=view.own_id)
    return (
        f"Make your choice for round {view.round_number} of "
        f"{view.max_rounds}.\n"
        f"- Your cumulative score so far: {view.own_score:.1f}\n"
        f"- Opponent's cumulative score so far: {view.opponent_score:.1f}\n"
        f"- Recent rounds (most recent last):\n{hist}\n\n"
        "Structured output: choice ('stag' or 'hare'), justification (brief "
        "rationale, up to 500 characters)."
    )


def _truncate_justification(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _normalize_decision(decision: StagHuntDecision) -> StagHuntDecision:
    return decision.model_copy(
        update={"justification": _truncate_justification(decision.justification)}
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, ValueError)),
    reraise=True,
)
async def _decide(view: StagHuntAgentView) -> StagHuntDecision:
    messages = [
        ("system", _build_system_prompt(view)),
        ("human", _build_human_prompt(view)),
    ]
    cfg = resolve_llm_config(view.experiment_id)
    result = await call_agent(
        messages, StagHuntDecision, experiment_id=view.experiment_id
    )
    if cfg.provider == "anthropic":
        _record_usage(result.raw)
    decision = StagHuntDecision.model_validate(result.parsed)
    return _normalize_decision(decision)


def _agent_view(state: StagHuntState, *, own_id: str) -> StagHuntAgentView:
    if own_id == AGENT_A_ID:
        own_trait, opponent_id = state.agent_a_trait, AGENT_B_ID
        own_score, opponent_score = state.agent_a_score, state.agent_b_score
    else:
        own_trait, opponent_id = state.agent_b_trait, AGENT_A_ID
        own_score, opponent_score = state.agent_b_score, state.agent_a_score
    return StagHuntAgentView(
        experiment_id=state.experiment_id,
        own_trait=own_trait,
        round_number=state.round_number,
        max_rounds=state.max_rounds,
        own_id=own_id,
        opponent_id=opponent_id,
        own_score=own_score,
        opponent_score=opponent_score,
        history=list(state.history),
    )


async def agents_node(state: StagHuntState) -> dict[str, Any]:
    """Fan-out both agents' simultaneous decisions in parallel (asyncio.gather)."""
    view_a = _agent_view(state, own_id=AGENT_A_ID)
    view_b = _agent_view(state, own_id=AGENT_B_ID)
    decision_a, decision_b = await asyncio.gather(_decide(view_a), _decide(view_b))
    return {
        "current_choice_a": decision_a.choice,
        "current_choice_b": decision_b.choice,
        "current_justification_a": decision_a.justification,
        "current_justification_b": decision_b.justification,
    }
