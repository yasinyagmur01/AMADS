import asyncio

from langchain_groq import ChatGroq

from core.config import settings
from core.state import AgentDecision, AgentInputView, SimulationState

if not settings.GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
    )

llm = ChatGroq(model="llama-3.3-70b-versatile")
structured_llm = llm.with_structured_output(AgentDecision)


def _build_system_prompt(agent_input: AgentInputView) -> str:
    trait = agent_input.own_trait
    env = agent_input.environment

    return f"""You are agent "{trait.agent_id}" in a shared-resource extraction simulation.

Trait Profile:
- Profile Label: {trait.profile_label}
- Cooperation Level: {trait.cooperation_assigned:.2f} (0 = selfish, 1 = highly cooperative)
- Risk Tolerance: {trait.risk_tolerance_assigned:.2f} (0 = risk-averse, 1 = risk-seeking)

Input Data:
- Current Round: {agent_input.round_number}
- Pool Current: {env.pool_current:.2f}
- Pool Capacity: {env.pool_capacity:.2f}
- Max Extractable This Round: {env.max_extractable_this_round:.2f}
- Regeneration Rate: {env.regen_rate:.2f}
- Pool Collapsed: {env.is_collapsed}

Decide how much to extract this round.
- extraction_amount must be between 0 and {env.max_extractable_this_round:.2f}
- declared_max is the maximum you are willing to extract (must be >= extraction_amount)

Your justification MUST reflect your trait profile and chosen extraction.
For example, a "Selfish" agent with low cooperation should justify aggressive extraction;
a "Cooperative" agent should explain restraint and long-term pool sustainability.

Return structured output with agent_id="{trait.agent_id}" and round_number={agent_input.round_number}.
Keep justification concise (under 400 characters)."""


async def _decide_for_agent(agent_input: AgentInputView) -> AgentDecision:
    messages = [
        ("system", _build_system_prompt(agent_input)),
        ("human", "Submit your extraction decision for this round."),
    ]
    decision = await structured_llm.ainvoke(messages)
    return decision.model_copy(
        update={
            "agent_id": agent_input.own_trait.agent_id,
            "round_number": agent_input.round_number,
        }
    )


async def run_agent_fanout(state: SimulationState) -> dict:
    agent_inputs = [
        AgentInputView(
            own_trait=trait,
            environment=state.environment,
            round_number=state.round_number,
        )
        for trait in state.agent_traits.values()
    ]

    decisions = await asyncio.gather(*[_decide_for_agent(inp) for inp in agent_inputs])
    return {"round_decisions": [*state.round_decisions, *decisions]}
