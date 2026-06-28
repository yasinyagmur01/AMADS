import asyncio
from dataclasses import dataclass

from anthropic import RateLimitError
from langchain_anthropic import ChatAnthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import settings
from core.state import AgentDecision, AgentInputView, SimulationState

REAL_AGENT_ID = "agent_1"  # run_single_agent_test hibrit koşusu için

# Claude Haiku 4.5 pricing (USD per 1M tokens, Bölüm 10.2)
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

_PLACEHOLDER_KEYS = frozenset({"", "your_key_here"})

_structured_llm = None


def _require_anthropic_key() -> str:
    key = settings.ANTHROPIC_API_KEY
    if not key or key in _PLACEHOLDER_KEYS:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your Anthropic API key."
        )
    return key


def _get_structured_llm():
    global _structured_llm
    if _structured_llm is None:
        llm = ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=settings.TEMPERATURE,
            api_key=_require_anthropic_key(),
        )
        _structured_llm = llm.with_structured_output(AgentDecision, include_raw=True)
    return _structured_llm


def _build_system_prompt(agent_input: AgentInputView) -> str:
    trait = agent_input.own_trait
    return (
        "Sen bir ortak kaynak havuzundan çekim yapan bir agentsın. "
        f"cooperation_assigned değerin {trait.cooperation_assigned:.2f} "
        "(0=tamamen bencil, 1=tamamen işbirlikçi), "
        f"risk_tolerance_assigned değerin {trait.risk_tolerance_assigned:.2f} "
        "(0=çok temkinli, 1=çok risk alan). "
        "Kararını bu eğilimlere uygun ver, ama bu sayıları çıktında tekrar etme veya açıklama."
    )


def _build_human_prompt(agent_input: AgentInputView) -> str:
    env = agent_input.environment
    return (
        f"Round {agent_input.round_number} için çekim kararını ver.\n"
        f"- Havuz: {env.pool_current:.2f} / {env.pool_capacity:.2f}\n"
        f"- Bu round maksimum çekilebilir: {env.max_extractable_this_round:.2f}\n"
        f"- Yenilenme oranı: {env.regen_rate:.2f}\n"
        f"- Havuz çöktü mü: {env.is_collapsed}\n\n"
        "Yapılandırılmış çıktı: extraction_amount (0 ile maksimum arası), "
        "justification (kısa gerekçe, 500 karaktere kadar), "
        "declared_max (>= extraction_amount)."
    )


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(RateLimitError),
    reraise=True,
)
async def _decide_with_anthropic(agent_input: AgentInputView) -> AgentDecision:
    messages = [
        ("system", _build_system_prompt(agent_input)),
        ("human", _build_human_prompt(agent_input)),
    ]
    result = await _get_structured_llm().ainvoke(messages)
    _record_usage(result["raw"])
    decision = AgentDecision.model_validate(result["parsed"])
    return decision.model_copy(
        update={
            "agent_id": agent_input.own_trait.agent_id,
            "round_number": agent_input.round_number,
        }
    )


def _agent_inputs(state: SimulationState) -> list[AgentInputView]:
    return [
        AgentInputView(
            own_trait=trait,
            environment=state.environment,
            round_number=state.round_number,
        )
        for trait in state.agent_traits.values()
    ]


async def run_agent_fanout(state: SimulationState) -> dict:
    """Bölüm 8.1: 5 agent paralel fan-out, hepsi gerçek Anthropic LLM."""
    agent_inputs = _agent_inputs(state)
    decisions = await asyncio.gather(
        *[_decide_with_anthropic(inp) for inp in agent_inputs]
    )
    return {"round_decisions": [*state.round_decisions, *decisions]}


async def run_hybrid_agent_fanout(state: SimulationState) -> dict:
    """Tek-agent testi: agent_1 → LLM; agent_2–5 → mock (Bölüm 9 adım 4)."""
    from agents.mock_agent import _mock_decision

    async def _decide(inp: AgentInputView) -> AgentDecision:
        if inp.own_trait.agent_id == REAL_AGENT_ID:
            return await _decide_with_anthropic(inp)
        return _mock_decision(inp, aggressive=False)

    agent_inputs = _agent_inputs(state)
    decisions = await asyncio.gather(*[_decide(inp) for inp in agent_inputs])
    return {"round_decisions": [*state.round_decisions, *decisions]}


def reset_token_usage() -> None:
    token_usage.input_tokens = 0
    token_usage.output_tokens = 0
