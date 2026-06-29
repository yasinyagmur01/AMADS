from core.state import AgentDecision, AgentInputView, SimulationState


def _control_decision(agent_input: AgentInputView) -> AgentDecision:
    max_ext = agent_input.environment.max_extractable_this_round
    coop = agent_input.own_trait.cooperation_assigned
    extraction = max_ext * (1.0 - coop)
    return AgentDecision(
        agent_id=agent_input.own_trait.agent_id,
        round_number=agent_input.round_number,
        extraction_amount=round(extraction, 4),
        justification="control_agent: rule-based, no LLM",
        declared_max=max_ext,
    )


async def run_control_agent_fanout(state: SimulationState) -> dict:
    """Deterministic control group — no LLM, coop-only rule."""
    decisions = [
        _control_decision(
            AgentInputView(
                own_trait=trait,
                environment=state.environment,
                round_number=state.round_number,
            )
        )
        for trait in state.agent_traits.values()
    ]
    return {"round_decisions": [*state.round_decisions, *decisions]}
