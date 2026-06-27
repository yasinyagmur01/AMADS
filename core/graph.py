from langgraph.graph import StateGraph, END
from core.state import SimulationState
# Artık klasör yapısından import ediyoruz
from agents.decision_agent import run_agent_fanout
from referee.referee_node import run_referee

# Graph İskeleti
workflow = StateGraph(SimulationState)

# Node'ları ekle
workflow.add_node("agent_fanout", run_agent_fanout)
workflow.add_node("referee", run_referee)

# Edge mantığı
workflow.set_entry_point("agent_fanout")
workflow.add_edge("agent_fanout", "referee")
workflow.add_conditional_edges(
    "referee",
    lambda state: END if state.is_terminated else "agent_fanout"
)

# Compile
app = workflow.compile()