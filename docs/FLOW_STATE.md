# Flow & State Definition

## 1. State Contract (`SimulationState`)
The global state is managed by a Pydantic v2 model. It is the single source of truth for the simulation.
- **`round_number`**: Incremented strictly by the Referee.
- **`round_decisions`**: Populated by the `agent_fanout` node; aggregated using `operator.add`.
- **`environment`**: Read-only for agents, modified only by the Referee.

## 2. Execution Logic (LangGraph)
- **Node `agent_fanout`**: 
    - Executes in parallel. 
    - Input: `AgentInputView` (Read-only subset of global state).
    - Output: `AgentDecision`.
- **Node `referee`**:
    - Centralized deterministic calculation.
    - Logic: Aggregates `round_decisions` -> Applies environmental rules -> Updates `environment` & `metrics_history` -> Increments `round_number`.
- **Conditional Logic**:
    - The graph checks `is_terminated` after every referee execution to determine if it should cycle back to `agent_fanout` or stop.

## 3. Read-Only Enforcement
To prevent "hallucinated" state changes, the `AgentInputView` is enforced at the function signature level. Agent node functions are strictly prohibited from accessing `SimulationState` directly.