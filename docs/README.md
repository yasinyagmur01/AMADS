# AMADS Documentation

## Source of truth

- **Active SSOT:** [`AMADS_MASTER_REFERENCE_EN.md`](AMADS_MASTER_REFERENCE_EN.md) (English).
- **Archived:** [`AMADS_MASTER_REFERENCE.md`](AMADS_MASTER_REFERENCE.md) (Turkish). Kept for historical reference; **not maintained**. Prefer the English file for all new architecture decisions.

## Prompt language

Original CPR and bargaining locked findings (`full_experiment_v1`, `bargaining_v1`, `bargaining_risk_v1`, etc.) used **Turkish** system prompts. That was an intentional design choice (strongest cooperation signal in a multilanguage pilot), not an oversight.

From the expansion onward, **all new experiment_ids use English** symbolic trait prompts. Language is therefore a documented experimental variable: locked Turkish baselines remain comparable; new runs are English unless explicitly noted otherwise.

## Other docs

- [`paper_draft.md`](paper_draft.md) — manuscript draft (includes §4.4 IPD micro-pilot: game structure as trait-signal moderator)
- [`expansion_summary.md`](expansion_summary.md) — Groq / cross-scenario expansion deliverable
- [`analysis_plan.md`](analysis_plan.md) — Sonnet cross-model prereg-lite
- [`figures/`](figures/) — paper figures
