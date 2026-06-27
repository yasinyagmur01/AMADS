from core.state import AgentDecision, AgentInputView, SimulationState


def _mock_decision(agent_input: AgentInputView, aggressive: bool) -> AgentDecision:
    max_ext = agent_input.environment.max_extractable_this_round
    coop = agent_input.own_trait.cooperation_assigned

    if aggressive:
        fraction = 0.95
    else:
        fraction = 0.10 + 0.25 * (1.0 - coop)

    extraction = min(max_ext * fraction, max_ext)
    return AgentDecision(
        agent_id=agent_input.own_trait.agent_id,
        round_number=agent_input.round_number,
        extraction_amount=round(extraction, 4),
        justification=f"mock decision (fraction={fraction:.2f})",
        declared_max=max_ext,
    )


async def run_mock_agent_fanout(state: SimulationState) -> dict:
    """Moderate extractions — pool should survive 15 rounds."""
    return await _run_mock_fanout(state, aggressive=False)


async def run_aggressive_mock_agent_fanout(state: SimulationState) -> dict:
    """Near-max extractions — pool should collapse before max_rounds."""
    return await _run_mock_fanout(state, aggressive=True)


async def _run_mock_fanout(state: SimulationState, aggressive: bool) -> dict:
    decisions = [
        _mock_decision(
            AgentInputView(
                own_trait=trait,
                environment=state.environment,
                round_number=state.round_number,
            ),
            aggressive=aggressive,
        )
        for trait in state.agent_traits.values()
    ]
    return {"round_decisions": [*state.round_decisions, *decisions]}
