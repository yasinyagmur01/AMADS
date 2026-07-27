"""Bargaining agents — proposer and responder (LLM decision only).

Prompt style matches full_experiment_v1 abstract/symbolic trait labels
(not prompt_revision_v1 behavioral rephrasing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anthropic import RateLimitError
from langchain_anthropic import ChatAnthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import settings
from scenarios.bargaining.state import (
    BargainingProposerView,
    BargainingResponderView,
    BargainingState,
    ProposerOffer,
    ResponderDecision,
)

# Claude Haiku 4.5 pricing (USD per 1M tokens) — same as CPR agents
_INPUT_COST_PER_M = 1.00
_OUTPUT_COST_PER_M = 5.00

_PLACEHOLDER_KEYS = frozenset({"", "your_key_here"})

_proposer_llm = None
_responder_llm = None


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


def _require_anthropic_key() -> str:
    key = settings.ANTHROPIC_API_KEY
    if not key or key in _PLACEHOLDER_KEYS:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your Anthropic API key."
        )
    return key


def _get_proposer_llm():
    global _proposer_llm
    if _proposer_llm is None:
        llm = ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=settings.TEMPERATURE,
            api_key=_require_anthropic_key(),
        )
        _proposer_llm = llm.with_structured_output(ProposerOffer, include_raw=True)
    return _proposer_llm


def _get_responder_llm():
    global _responder_llm
    if _responder_llm is None:
        llm = ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=settings.TEMPERATURE,
            api_key=_require_anthropic_key(),
        )
        _responder_llm = llm.with_structured_output(ResponderDecision, include_raw=True)
    return _responder_llm


def _record_usage(raw_message) -> None:
    usage = getattr(raw_message, "usage_metadata", None) or {}
    inp = usage.get("input_tokens")
    out = usage.get("output_tokens")
    if inp is None or out is None:
        meta = getattr(raw_message, "response_metadata", None) or {}
        api_usage = meta.get("usage") or {}
        inp = inp if inp is not None else api_usage.get("input_tokens", 0)
        out = out if out is not None else api_usage.get("output_tokens", 0)
    token_usage.add(int(inp or 0), int(out or 0))


def _history_summary(recent: list[dict[str, Any]], limit: int = 3) -> str:
    if not recent:
        return "Henüz geçmiş round yok."
    lines = []
    for h in recent[-limit:]:
        lines.append(
            f"  round {h.get('round_number')}: keep={h.get('keep_amount'):.1f}, "
            f"kabul={'evet' if h.get('accepted') else 'hayır'}, "
            f"payoffs=({h.get('proposer_payoff'):.1f}, {h.get('responder_payoff'):.1f})"
        )
    return "\n".join(lines)


def _build_proposer_system(view: BargainingProposerView) -> str:
    t = view.own_trait
    return (
        "Sen bir pazarlık oyununda TEKLİF EDEN (Proposer) rolesindesin. "
        f"Sabit bir kaynak havuzu var: {view.pie_size:.0f} birim. "
        "Teklifin: kendine X birim ayırıyorsun; karşı taraf "
        f"{view.pie_size:.0f}-X alır. Responder kabul ederse paylar gerçekleşir; "
        "reddederse her iki taraf da sıfır alır.\n"
        f"cooperation_assigned değerin {t.cooperation_assigned:.2f} "
        "(0=kendine maksimum pay ayır / agresif teklif, "
        "1=karşı tarafa adil veya cömert pay ayır / düşük kendi payı), "
        f"risk_tolerance_assigned değerin {t.risk_tolerance_assigned:.2f} "
        "(0=red riskinden kaçın / güvenli düşük kendi payı teklifi, "
        "1=red riskini göze al / kendine yüksek pay ayır). "
        "Kararını bu eğilimlere uygun ver, ama bu sayıları çıktında tekrar etme veya açıklama."
    )


def _build_proposer_human(view: BargainingProposerView) -> str:
    return (
        f"Round {view.round_number} için teklifini ver.\n"
        f"- Havuz (pie): {view.pie_size:.0f}\n"
        f"- time_pressure: {view.time_pressure:.2f}\n"
        f"- resource_scarcity: {view.resource_scarcity:.2f}\n"
        f"- Son round'lar:\n{_history_summary(view.recent_history)}\n\n"
        "Yapılandırılmış çıktı: keep_amount (kendine ayırdığın miktar, "
        f"0–{view.pie_size:.0f}), justification (kısa gerekçe, 500 karaktere kadar)."
    )


def _build_responder_system(view: BargainingResponderView) -> str:
    t = view.own_trait
    return (
        "Sen bir pazarlık oyununda YANITLAYAN (Responder) rolesindesin. "
        "Proposer bir teklif yaptı: kendine X, sana pie−X. "
        "Kabul edersen paylar gerçekleşir; reddedersen her iki taraf da sıfır alır.\n"
        f"cooperation_assigned değerin {t.cooperation_assigned:.2f} "
        "(0=sadece kendi payın yüksekse kabul et / talepkâr, "
        "1=düşük teklifleri bile kabul et / esnek işbirlikçi), "
        f"risk_tolerance_assigned değerin {t.risk_tolerance_assigned:.2f} "
        "(0=red riskinden kaçın / teklifi kabul etmeye eğilimli, "
        "1=red riskini göze al / yüksek kendi payı talep et). "
        "Kararını bu eğilimlere uygun ver, ama bu sayıları çıktında tekrar etme veya açıklama."
    )


def _build_responder_human(view: BargainingResponderView) -> str:
    return (
        f"Round {view.round_number} için teklifi değerlendir.\n"
        f"- Havuz (pie): {view.pie_size:.0f}\n"
        f"- Proposer kendine ayırıyor (X): {view.current_offer_keep:.2f}\n"
        f"- Sana teklif edilen: {view.offer_to_me:.2f}\n"
        f"- time_pressure: {view.time_pressure:.2f}\n"
        f"- resource_scarcity: {view.resource_scarcity:.2f}\n"
        f"- Son round'lar:\n{_history_summary(view.recent_history)}\n\n"
        "Yapılandırılmış çıktı: accept (true/false), "
        "justification (kısa gerekçe, 500 karaktere kadar)."
    )


def _truncate_justification(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _offer_from_structured(result: dict) -> ProposerOffer:
    parsed = result.get("parsed")
    if parsed is not None:
        if isinstance(parsed, ProposerOffer):
            return parsed.model_copy(
                update={
                    "justification": _truncate_justification(parsed.justification)
                }
            )
        return ProposerOffer.model_validate(parsed)

    raw = result.get("raw")
    tool_calls = getattr(raw, "tool_calls", None) or []
    if tool_calls:
        args = dict(tool_calls[0].get("args") or {})
        args["justification"] = _truncate_justification(
            str(args.get("justification", ""))
        )
        return ProposerOffer.model_validate(args)

    raise ValueError(
        f"ProposerOffer parse failed: parsing_error={result.get('parsing_error')!r}"
    )


def _decision_from_structured(result: dict) -> ResponderDecision:
    parsed = result.get("parsed")
    if parsed is not None:
        if isinstance(parsed, ResponderDecision):
            return parsed.model_copy(
                update={
                    "justification": _truncate_justification(parsed.justification)
                }
            )
        return ResponderDecision.model_validate(parsed)

    raw = result.get("raw")
    tool_calls = getattr(raw, "tool_calls", None) or []
    if tool_calls:
        args = dict(tool_calls[0].get("args") or {})
        args["justification"] = _truncate_justification(
            str(args.get("justification", ""))
        )
        return ResponderDecision.model_validate(args)

    raise ValueError(
        f"ResponderDecision parse failed: parsing_error={result.get('parsing_error')!r}"
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, ValueError)),
    reraise=True,
)
async def _proposer_decide(view: BargainingProposerView) -> ProposerOffer:
    messages = [
        ("system", _build_proposer_system(view)),
        ("human", _build_proposer_human(view)),
    ]
    result = await _get_proposer_llm().ainvoke(messages)
    _record_usage(result["raw"])
    return _offer_from_structured(result)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, ValueError)),
    reraise=True,
)
async def _responder_decide(view: BargainingResponderView) -> ResponderDecision:
    messages = [
        ("system", _build_responder_system(view)),
        ("human", _build_responder_human(view)),
    ]
    result = await _get_responder_llm().ainvoke(messages)
    _record_usage(result["raw"])
    return _decision_from_structured(result)

def _proposer_view(state: BargainingState) -> BargainingProposerView:
    return BargainingProposerView(
        own_trait=state.proposer_trait,
        env_snapshot=state.env_snapshot,
        time_pressure=state.time_pressure,
        resource_scarcity=state.resource_scarcity,
        round_number=state.round_number,
        pie_size=state.pie_size,
        recent_history=list(state.history),
    )


def _responder_view(state: BargainingState) -> BargainingResponderView:
    if state.current_offer is None:
        raise ValueError("Responder requires current_offer from proposer")
    keep = float(state.current_offer)
    return BargainingResponderView(
        own_trait=state.responder_trait,
        env_snapshot=state.env_snapshot,
        time_pressure=state.time_pressure,
        resource_scarcity=state.resource_scarcity,
        round_number=state.round_number,
        pie_size=state.pie_size,
        current_offer_keep=keep,
        offer_to_me=state.pie_size - keep,
        recent_history=list(state.history),
    )


async def proposer_agent(state: BargainingState) -> dict[str, Any]:
    """Proposer node: produce keep_amount offer."""
    offer = await _proposer_decide(_proposer_view(state))
    keep = max(0.0, min(state.pie_size, float(offer.keep_amount)))
    return {
        "current_offer": keep,
        "current_justification_proposer": offer.justification,
    }


async def responder_agent(state: BargainingState) -> dict[str, Any]:
    """Responder node: accept or reject the current offer."""
    decision = await _responder_decide(_responder_view(state))
    return {
        "current_accept": bool(decision.accept),
        "current_justification_responder": decision.justification,
    }
