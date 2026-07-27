"""LangGraph StateGraph for bargaining: proposer → responder → referee → loop."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from scenarios.bargaining.agents import proposer_agent, responder_agent
from scenarios.bargaining.referee import run_referee
from scenarios.bargaining.state import BargainingState

workflow = StateGraph(BargainingState)

workflow.add_node("proposer", proposer_agent)
workflow.add_node("responder", responder_agent)
workflow.add_node("referee", run_referee)

workflow.set_entry_point("proposer")
workflow.add_edge("proposer", "responder")
workflow.add_edge("responder", "referee")
workflow.add_conditional_edges(
    "referee",
    lambda state: END if state.is_terminated else "proposer",
)

app = workflow.compile()
