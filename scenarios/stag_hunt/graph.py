"""LangGraph StateGraph for stag hunt: agents (parallel fan-out) -> referee -> loop."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from scenarios.stag_hunt.agents import agents_node
from scenarios.stag_hunt.referee import run_referee
from scenarios.stag_hunt.state import StagHuntState

workflow = StateGraph(StagHuntState)

workflow.add_node("agents", agents_node)
workflow.add_node("referee", run_referee)

workflow.set_entry_point("agents")
workflow.add_edge("agents", "referee")
workflow.add_conditional_edges(
    "referee",
    lambda state: END if state.is_terminated else "agents",
)

app = workflow.compile()
