# System Architecture Document

## Overview
AMADS (Academic Multi-Agent Decision Simulation) is a deterministic, LLM-based simulation framework designed for rigorous academic analysis of multi-agent Commons Dilemma scenarios.

## Core Pillars
1. **Deterministic Execution:** The `Referee` node is the single source of truth, performing all mathematical calculations. LLMs are used exclusively for decision generation, never for evaluation.
2. **Read-Only Enforcement:** Agent nodes receive only `AgentInputView`, preventing them from altering the global state.
3. **Parallel Fan-out:** Agent decisions are independent and parallel, enforcing simultaneous game rules.

## Data Flow
- **State:** `SimulationState` (Pydantic v2) acts as the central contract.
- **Loop:** The `Referee` evaluates the `AgentDecision` outputs, applies environmental updates (shocks/pool dynamics), and advances the `round_number`.