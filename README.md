# AMADS

**Academic Multi-Agent Decision Simulation** — a research framework for studying LLM agent behavior in Commons Dilemma scenarios.

## Overview

AMADS simulates multi-agent resource extraction games where LLM-powered agents make independent extraction decisions each round. A deterministic **Referee** node (no LLM calls) computes all metrics—pool dynamics, Gini coefficient, cooperation scores, collapse detection—and advances simulation state. Agents receive only a read-only `AgentInputView` and are assigned numeric trait profiles (e.g., cooperation, risk tolerance). The framework measures **trait fidelity**: whether assigned traits predict observed extraction behavior across factorial experimental conditions.

## Key Findings

1. **Cooperation inverse fidelity (Haiku):** In `full_experiment_v1` (Claude Haiku 4.5, 45 runs), higher assigned cooperation correlates with *higher* extraction (r ≈ +0.46, p ≈ 0.002)—the opposite of the operational definition. Deterministic control agents show the expected negative direction (r = −1.0).

2. **Cross-model divergence (Sonnet vs Haiku):** A confirmatory micro-replication with Claude Sonnet 4.6 (`sonnet_crossmodel_v1`, n = 40) reverses the cooperation pattern (r ≈ −0.84) while risk fidelity weakens (r ≈ +0.15 vs Haiku r ≈ +0.68), indicating model-dependent trait alignment rather than a universal LLM failure mode.

3. **No predictable category rule across 11 traits:** An 11-language pilot (`prompt_ab_multilang.py`) and extended trait-category probes found no stable linguistic or categorical rule that explains when traits transfer faithfully; inverse cooperation fidelity was strongest in Turkish but appears across multiple languages at varying magnitudes.

4. **Game structure moderates trait signal (IPD):** An Iterated Prisoner's Dilemma micro-pilot (`iterated_pd_groq_v1`, 10 runs / 120 rounds on Groq) produced near-ceiling cooperation in both trait cells (C-rate ≈ 0.98–1.00; r = +0.333, n.s.). Equilibrium pressure can suppress prompt-level trait variance entirely—trait fidelity is scenario-specific as well as model- and prompt-specific. See `docs/paper_draft.md` §4.4.

## Project Structure

```
core/                 # Shared state, config, SQLite persistence, CPR graph wiring, LLM providers
agents/               # LLM decision agent, mock/control agents (CPR)
referee/              # Deterministic CPR Referee node
scenarios/bargaining/ # Ultimatum bargaining scenario (isolated from CPR)
scenarios/iterated_pd/ # Iterated Prisoner's Dilemma agents + referee
scenarios/stag_hunt/  # Stag Hunt coordination scenario
analysis/             # Trait fidelity, clustering, synthesis, prompt A/B scripts
experiments/cpr/      # CPR factorial / control / heterogeneous / prompt-revision runners
experiments/bargaining/ # Bargaining experiment runner
experiments/iterated_pd/ # IPD micro-pilot runner
experiments/stag_hunt/   # Stag Hunt micro-pilot runner
scripts/cpr/          # CPR condition seed scripts
scripts/bargaining/   # Bargaining condition seed scripts
tests/                # Unit tests (Referee metrics, no LLM)
docs/                 # Master reference, paper draft, figures, analysis plans
data/                 # Experiment CSV/MD/log outputs and SQLite databases (*.db gitignored)
environment/          # Shock schedules and environmental events
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
python experiments/cpr/run_full_experiment.py --plan

# Run full_experiment_v1 (requires ANTHROPIC_API_KEY; writes to data/results.db)
python experiments/cpr/run_full_experiment.py
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

The system prompt delivered to LLM agents in `full_experiment_v1` (and locked bargaining pilots) was intentionally written in Turkish. That is a documented historical variable, not an oversight — see `docs/README.md` and master reference Section 18.4.1. **New experiment_ids use English** symbolic prompts (e.g. `python experiments/cpr/run_full_experiment.py --english` → `full_experiment_en_v1`). Architecture SSOT: `docs/AMADS_MASTER_REFERENCE_EN.md`.

## Citation

[Paper citation when published]
