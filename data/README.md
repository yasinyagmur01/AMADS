# AMADS Experiment Data Log

Locked datasets must **never** be modified or re-run. All new work uses new `experiment_id` values.

## Locked experiment_ids (do not modify)

| experiment_id | scenario | model | prompt language | purpose |
|---|---|---|---|---|
| `full_experiment_v1` | CPR | Claude Haiku 4.5 | Turkish | Locked main factorial (9×5) |
| `control_group_v1` | CPR | rule-based control | n/a | Deterministic control baseline |
| `heterogeneous_v1` | CPR | Claude Haiku 4.5 | Turkish | Heterogeneous trait compositions |
| `prompt_revision_v1` | CPR | Claude Haiku 4.5 | Turkish (behavioral wording) | Concept-misalignment prompt revision |
| `bargaining_v1` | Bargaining | Claude Haiku 4.5 | Turkish | Coop micro-pilot |
| `bargaining_risk_v1` | Bargaining | Claude Haiku 4.5 | Turkish | Risk micro-pilot |

## Databases

| Path | Contents |
|---|---|
| `results.db` | CPR experiments |
| `bargaining_results.db` | Bargaining experiments |
| `archive_results.db` | Archived CPR-shaped tables |
| `ipd_results.db` | Iterated PD (when present) |
| `stag_hunt_results.db` | Stag Hunt (when present) |

## New / planned experiment_ids

| experiment_id | scenario | model | prompt language | date | purpose |
|---|---|---|---|---|---|
| `full_experiment_en_v1` | CPR | Claude Haiku 4.5 (selectable) | English | 2026-07-28 | English prompt variant of main CPR design (not auto-run) |
| `groq_smoke_test` | CPR / Bargaining smoke | Groq `llama-3.3-70b-versatile` | English | 2026-07-28 | Provider/model validation (structured tool-calling) |
| `full_experiment_groq_v1` | CPR | Groq | English | TBD | Third-model CPR replication |
| `bargaining_groq_v1` | Bargaining | Groq | English | TBD | Third-model bargaining coop pilot |
| `bargaining_risk_groq_v1` | Bargaining | Groq | English | TBD | Third-model bargaining risk pilot |

**Groq note (Phase 2):** Groq was previously removed from architecture docs to avoid mixing providers within a run (confound risk), not because of a technical failure. Re-introduction uses **one provider per experiment_id**. `llama-3.3-70b-versatile` supports tool/function calling (not `json_schema`); AMADS uses `method="function_calling"` for Groq structured outputs.
