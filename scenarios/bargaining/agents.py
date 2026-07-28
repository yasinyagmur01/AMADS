"""Bargaining agents — proposer and responder (LLM decision only).

Prompt style matches full_experiment_v1 abstract/symbolic trait labels
(not prompt_revision_v1 behavioral rephrasing). Locked Turkish experiments
keep Turkish prompts; new experiment_ids use English.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anthropic import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.llm_providers import call_agent, resolve_llm_config
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

_LOCKED_TURKISH_EXPERIMENT_IDS = frozenset(
    {
        "bargaining_v1",
        "bargaining_risk_v1",
    }
)


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


def uses_english_prompts(experiment_id: str) -> bool:
    """English for new experiment_ids; Turkish for locked historical runs."""
    return experiment_id not in _LOCKED_TURKISH_EXPERIMENT_IDS


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
    recent: list[dict[str, Any]],
    *,
    english: bool,
    limit: int = 3,
) -> str:
    if not recent:
        return "No previous rounds yet." if english else "Henüz geçmiş round yok."
    lines = []
    for h in recent[-limit:]:
        if english:
            accepted = "yes" if h.get("accepted") else "no"
            lines.append(
                f"  round {h.get('round_number')}: keep={h.get('keep_amount'):.1f}, "
                f"accepted={accepted}, "
                f"payoffs=({h.get('proposer_payoff'):.1f}, {h.get('responder_payoff'):.1f})"
            )
        else:
            lines.append(
                f"  round {h.get('round_number')}: keep={h.get('keep_amount'):.1f}, "
                f"kabul={'evet' if h.get('accepted') else 'hayır'}, "
                f"payoffs=({h.get('proposer_payoff'):.1f}, {h.get('responder_payoff'):.1f})"
            )
    return "\n".join(lines)


def _build_proposer_system(view: BargainingProposerView) -> str:
    t = view.own_trait
    english = uses_english_prompts(view.experiment_id)
    if english:
        return (
            "You are the PROPOSER in a bargaining (ultimatum) game. "
            f"There is a fixed pie of {view.pie_size:.0f} units. "
            "Your offer: you keep X for yourself; the other party receives "
            f"{view.pie_size:.0f}-X. If the responder accepts, payoffs are realized; "
            "if they reject, both parties get zero.\n"
            f"Your cooperation_assigned value is {t.cooperation_assigned:.2f} "
            "(0=keep maximum for self / aggressive offer, "
            "1=fair or generous share to the other / low own keep), "
            f"your risk_tolerance_assigned value is {t.risk_tolerance_assigned:.2f} "
            "(0=avoid rejection risk / safe low keep, "
            "1=accept rejection risk / high keep for self). "
            "Make your decision in line with these tendencies, but do not repeat "
            "or explain these numbers in your output. "
            "Structured fields ONLY: keep_amount and justification — "
            "do not invent extra fields."
        )
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
    english = uses_english_prompts(view.experiment_id)
    hist = _history_summary(view.recent_history, english=english)
    if english:
        return (
            f"Make your offer for round {view.round_number}.\n"
            f"- Pie size: {view.pie_size:.0f}\n"
            f"- time_pressure: {view.time_pressure:.2f}\n"
            f"- resource_scarcity: {view.resource_scarcity:.2f}\n"
            f"- Recent rounds:\n{hist}\n\n"
            "Structured output: keep_amount (amount you keep for yourself, "
            f"0–{view.pie_size:.0f}), justification (brief rationale, up to 500 characters)."
        )
    return (
        f"Round {view.round_number} için teklifini ver.\n"
        f"- Havuz (pie): {view.pie_size:.0f}\n"
        f"- time_pressure: {view.time_pressure:.2f}\n"
        f"- resource_scarcity: {view.resource_scarcity:.2f}\n"
        f"- Son round'lar:\n{hist}\n\n"
        "Yapılandırılmış çıktı: keep_amount (kendine ayırdığın miktar, "
        f"0–{view.pie_size:.0f}), justification (kısa gerekçe, 500 karaktere kadar)."
    )


def _build_responder_system(view: BargainingResponderView) -> str:
    t = view.own_trait
    english = uses_english_prompts(view.experiment_id)
    if english:
        return (
            "You are the RESPONDER in a bargaining (ultimatum) game. "
            "The proposer offered: keep X for themselves, pie−X for you. "
            "If you accept, payoffs are realized; if you reject, both get zero.\n"
            f"Your cooperation_assigned value is {t.cooperation_assigned:.2f} "
            "(0=accept only if your share is high / demanding, "
            "1=accept even low offers / flexible cooperative), "
            f"your risk_tolerance_assigned value is {t.risk_tolerance_assigned:.2f} "
            "(0=avoid rejection risk / inclined to accept, "
            "1=accept rejection risk / demand high own share). "
            "Make your decision in line with these tendencies, but do not repeat "
            "or explain these numbers in your output. "
            "Structured fields ONLY: accept and justification — "
            "do not invent extra fields."
        )
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
    english = uses_english_prompts(view.experiment_id)
    hist = _history_summary(view.recent_history, english=english)
    if english:
        return (
            f"Evaluate the offer for round {view.round_number}.\n"
            f"- Pie size: {view.pie_size:.0f}\n"
            f"- Proposer keeps (X): {view.current_offer_keep:.2f}\n"
            f"- Offered to you: {view.offer_to_me:.2f}\n"
            f"- time_pressure: {view.time_pressure:.2f}\n"
            f"- resource_scarcity: {view.resource_scarcity:.2f}\n"
            f"- Recent rounds:\n{hist}\n\n"
            "Structured output: accept (true/false), "
            "justification (brief rationale, up to 500 characters)."
        )
    return (
        f"Round {view.round_number} için teklifi değerlendir.\n"
        f"- Havuz (pie): {view.pie_size:.0f}\n"
        f"- Proposer kendine ayırıyor (X): {view.current_offer_keep:.2f}\n"
        f"- Sana teklif edilen: {view.offer_to_me:.2f}\n"
        f"- time_pressure: {view.time_pressure:.2f}\n"
        f"- resource_scarcity: {view.resource_scarcity:.2f}\n"
        f"- Son round'lar:\n{hist}\n\n"
        "Yapılandırılmış çıktı: accept (true/false), "
        "justification (kısa gerekçe, 500 karaktere kadar)."
    )


def _truncate_justification(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _normalize_offer(offer: ProposerOffer) -> ProposerOffer:
    return offer.model_copy(
        update={"justification": _truncate_justification(offer.justification)}
    )


def _normalize_decision(decision: ResponderDecision) -> ResponderDecision:
    return decision.model_copy(
        update={"justification": _truncate_justification(decision.justification)}
    )


def _retry_exc() -> tuple:
    exc: list[type[BaseException]] = [RateLimitError, ValueError]
    try:
        from openai import BadRequestError as OpenAIBadRequestError
        from openai import RateLimitError as OpenAIRateLimitError

        exc.append(OpenAIRateLimitError)
        exc.append(OpenAIBadRequestError)
    except ImportError:
        pass
    return tuple(exc)


@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=4, min=5, max=420),
    retry=retry_if_exception_type(_retry_exc()),
    reraise=True,
)
async def _proposer_decide(view: BargainingProposerView) -> ProposerOffer:
    messages = [
        ("system", _build_proposer_system(view)),
        ("human", _build_proposer_human(view)),
    ]
    cfg = resolve_llm_config(view.experiment_id)
    result = await call_agent(
        messages, ProposerOffer, experiment_id=view.experiment_id
    )
    if cfg.provider == "anthropic":
        _record_usage(result.raw)
    offer = ProposerOffer.model_validate(result.parsed)
    return _normalize_offer(offer)


@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=4, min=5, max=420),
    retry=retry_if_exception_type(_retry_exc()),
    reraise=True,
)
async def _responder_decide(view: BargainingResponderView) -> ResponderDecision:
    messages = [
        ("system", _build_responder_system(view)),
        ("human", _build_responder_human(view)),
    ]
    cfg = resolve_llm_config(view.experiment_id)
    result = await call_agent(
        messages, ResponderDecision, experiment_id=view.experiment_id
    )
    if cfg.provider == "anthropic":
        _record_usage(result.raw)
    decision = ResponderDecision.model_validate(result.parsed)
    return _normalize_decision(decision)


def _proposer_view(state: BargainingState) -> BargainingProposerView:
    return BargainingProposerView(
        experiment_id=state.experiment_id,
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
        experiment_id=state.experiment_id,
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
