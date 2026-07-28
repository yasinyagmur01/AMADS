import asyncio
from dataclasses import dataclass

from anthropic import RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import settings
from core.llm_providers import call_agent, resolve_llm_config
from core.state import AgentDecision, AgentInputView, SimulationState

REAL_AGENT_ID = "agent_1"  # hybrid single-agent test: agent_1 uses LLM

# Claude Haiku 4.5 pricing (USD per 1M tokens); Groq free-tier treated as $0
_INPUT_COST_PER_M = 1.00
_OUTPUT_COST_PER_M = 5.00

# Locked historical experiments keep Turkish symbolic prompts.
_LOCKED_TURKISH_EXPERIMENT_IDS = frozenset(
    {
        "full_experiment_v1",
        "control_group_v1",
        "heterogeneous_v1",
        "prompt_revision_v1",
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


def uses_english_prompts(experiment_id: str) -> bool:
    """English for new experiment_ids; Turkish for locked historical runs."""
    return experiment_id not in _LOCKED_TURKISH_EXPERIMENT_IDS


# Cooperation wording by experiment_id (Turkish locked baselines).
_COOPERATION_PROMPT_TR_BY_EXPERIMENT: dict[str, str] = {
    "prompt_revision_v1": (
        "0=havuzdan mümkün olan maksimum miktarı çek, "
        "1=havuzu korumak için mümkün olan minimum miktarı çek"
    ),
}
_DEFAULT_COOPERATION_PROMPT_TR = "0=tamamen bencil, 1=tamamen işbirlikçi"
_DEFAULT_COOPERATION_PROMPT_EN = "0=fully selfish, 1=fully cooperative"
_DEFAULT_RISK_PROMPT_TR = "0=çok temkinli, 1=çok risk alan"
_DEFAULT_RISK_PROMPT_EN = "0=very cautious, 1=very risk-taking"


def _cooperation_prompt_clause(experiment_id: str) -> str:
    if uses_english_prompts(experiment_id):
        return _DEFAULT_COOPERATION_PROMPT_EN
    return _COOPERATION_PROMPT_TR_BY_EXPERIMENT.get(
        experiment_id, _DEFAULT_COOPERATION_PROMPT_TR
    )


def _risk_prompt_clause(experiment_id: str) -> str:
    if uses_english_prompts(experiment_id):
        return _DEFAULT_RISK_PROMPT_EN
    return _DEFAULT_RISK_PROMPT_TR


def _build_system_prompt(
    agent_input: AgentInputView,
    experiment_id: str = "full_experiment_v1",
) -> str:
    trait = agent_input.own_trait
    coop_clause = _cooperation_prompt_clause(experiment_id)
    risk_clause = _risk_prompt_clause(experiment_id)
    if uses_english_prompts(experiment_id):
        return (
            "You are an agent extracting from a shared resource pool. "
            f"Your cooperation_assigned value is {trait.cooperation_assigned:.2f} "
            f"({coop_clause}), "
            f"your risk_tolerance_assigned value is {trait.risk_tolerance_assigned:.2f} "
            f"({risk_clause}). "
            "Make your decision in line with these tendencies, but do not repeat "
            "or explain these numbers in your output."
        )
    return (
        "Sen bir ortak kaynak havuzundan çekim yapan bir agentsın. "
        f"cooperation_assigned değerin {trait.cooperation_assigned:.2f} "
        f"({coop_clause}), "
        f"risk_tolerance_assigned değerin {trait.risk_tolerance_assigned:.2f} "
        f"({risk_clause}). "
        "Kararını bu eğilimlere uygun ver, ama bu sayıları çıktında tekrar etme veya açıklama."
    )


def _build_human_prompt(
    agent_input: AgentInputView,
    experiment_id: str = "full_experiment_v1",
) -> str:
    env = agent_input.environment
    if uses_english_prompts(experiment_id):
        return (
            f"Make your extraction decision for round {agent_input.round_number}.\n"
            f"- Pool: {env.pool_current:.2f} / {env.pool_capacity:.2f}\n"
            f"- Maximum extractable this round: {env.max_extractable_this_round:.2f}\n"
            f"- Regeneration rate: {env.regen_rate:.2f}\n"
            f"- Pool collapsed: {env.is_collapsed}\n\n"
            "Structured output: extraction_amount (between 0 and maximum), "
            "justification (brief rationale, up to 500 characters), "
            "declared_max (>= extraction_amount)."
        )
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
        # OpenAI-compatible usage keys
        if not inp:
            inp = api_usage.get("prompt_tokens", 0)
        if not out:
            out = api_usage.get("completion_tokens", 0)
    token_usage.add(int(inp or 0), int(out or 0))


def _retry_exceptions() -> tuple:
    exc: list[type[BaseException]] = [RateLimitError, ValueError]
    try:
        from openai import RateLimitError as OpenAIRateLimitError

        exc.append(OpenAIRateLimitError)
    except ImportError:
        pass
    return tuple(exc)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, ValueError)),
    reraise=True,
)
async def _decide(
    agent_input: AgentInputView,
    experiment_id: str = "full_experiment_v1",
) -> AgentDecision:
    messages = [
        ("system", _build_system_prompt(agent_input, experiment_id=experiment_id)),
        ("human", _build_human_prompt(agent_input, experiment_id=experiment_id)),
    ]
    cfg = resolve_llm_config(experiment_id)
    # Anthropic cost accounting only; Groq free tier reported as $0 by callers
    result = await call_agent(messages, AgentDecision, experiment_id=experiment_id)
    if cfg.provider == "anthropic":
        _record_usage(result.raw)
    decision = AgentDecision.model_validate(result.parsed)
    return decision.model_copy(
        update={
            "agent_id": agent_input.own_trait.agent_id,
            "round_number": agent_input.round_number,
        }
    )


# Back-compat alias for analysis scripts that imported the old name
_decide_with_anthropic = _decide


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
    """Section 8.1: 5-agent parallel fan-out via configured LLM provider."""
    agent_inputs = _agent_inputs(state)
    experiment_id = state.experiment_id
    decisions = await asyncio.gather(
        *[_decide(inp, experiment_id=experiment_id) for inp in agent_inputs]
    )
    return {"round_decisions": [*state.round_decisions, *decisions]}


async def run_hybrid_agent_fanout(state: SimulationState) -> dict:
    """Single-agent test: agent_1 → LLM; agent_2–5 → mock."""
    from agents.mock_agent import _mock_decision

    experiment_id = state.experiment_id

    async def _one(inp: AgentInputView) -> AgentDecision:
        if inp.own_trait.agent_id == REAL_AGENT_ID:
            return await _decide(inp, experiment_id=experiment_id)
        return _mock_decision(inp, aggressive=False)

    agent_inputs = _agent_inputs(state)
    decisions = await asyncio.gather(*[_one(inp) for inp in agent_inputs])
    return {"round_decisions": [*state.round_decisions, *decisions]}


def reset_token_usage() -> None:
    token_usage.input_tokens = 0
    token_usage.output_tokens = 0
