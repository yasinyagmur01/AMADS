# AMADS

**Academic Multi-Agent Decision Simulation** — a research framework for studying LLM agent behavior in Commons Dilemma scenarios.

## Overview

AMADS simulates multi-agent resource extraction games where LLM-powered agents make independent extraction decisions each round. A deterministic **Referee** node (no LLM calls) computes all metrics—pool dynamics, Gini coefficient, cooperation scores, collapse detection—and advances simulation state. Agents receive only a read-only `AgentInputView` and are assigned numeric trait profiles (e.g., cooperation, risk tolerance). The framework measures **trait fidelity**: whether assigned traits predict observed extraction behavior across factorial experimental conditions.

## Key Findings

1. **Cooperation inverse fidelity (Haiku):** In `full_experiment_v1` (Claude Haiku 4.5, 45 runs), higher assigned cooperation correlates with *higher* extraction (r ≈ +0.46, p ≈ 0.002)—the opposite of the operational definition. Deterministic control agents show the expected negative direction (r = −1.0).

2. **Cross-model divergence (Sonnet vs Haiku):** A confirmatory micro-replication with Claude Sonnet 4.6 (`sonnet_crossmodel_v1`, n = 40) reverses the cooperation pattern (r ≈ −0.84) while risk fidelity weakens (r ≈ +0.15 vs Haiku r ≈ +0.68), indicating model-dependent trait alignment rather than a universal LLM failure mode.

3. **No predictable category rule across 11 traits:** An 11-language pilot (`prompt_ab_multilang.py`) and extended trait-category probes found no stable linguistic or categorical rule that explains when traits transfer faithfully; inverse cooperation fidelity was strongest in Turkish but appears across multiple languages at varying magnitudes.

## Project Structure

```
core/           # State models (Pydantic), LangGraph wiring, config, SQLite persistence
agents/         # LLM decision agent, mock/control agents
referee/        # Deterministic Referee node (metrics, pool dynamics, termination)
analysis/       # Trait fidelity, clustering, synthesis, prompt A/B scripts
experiments/    # Full factorial and control-group experiment runners
tests/          # Unit tests (Referee metrics, no LLM)
docs/           # Architecture notes, analysis plans, diagrams
data/           # Experiment CSV/MD outputs and SQLite databases (*.db gitignored)
environment/    # Shock schedules and environmental events
scripts/legacy/ # Deprecated one-off dev/test runners (pre-paper cleanup)
```

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY (and optionally LANGSMITH_* keys)
```

## Running the Main Experiment

From the repository root:

```bash
# Preview the 45-run plan (9 conditions × 5 replications) without calling the API
python experiments/run_full_experiment.py --plan

# Run full_experiment_v1 (requires ANTHROPIC_API_KEY; writes to data/results.db)
python experiments/run_full_experiment.py
```

## Running Analysis

```bash
# Synthesis report (fidelity + clustering + control comparison)
python analysis/synthesis_report.py
python analysis/synthesis_report.py --output data/synthesis_report.md

# Trait fidelity tables and plots
python analysis/trait_fidelity.py
python analysis/trait_fidelity.py --max-round 2
```

Requires `data/results.db` populated by `full_experiment_v1`.

## Important Note on Language

The system prompt delivered to LLM agents in full_experiment_v1 was intentionally written in Turkish. This was a deliberate design choice documented in AMADS_MASTER_REFERENCE.md (Section 18.4.1), where Turkish produced the strongest and cleanest cooperation signal. New experiments should use English prompts (see analysis/prompt_ab_multilang.py for cross-language comparison results).

## Citation

[Paper citation when published]
