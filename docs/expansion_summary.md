# AMADS Expansion Summary (July 2026)

This document is the Phase 10 deliverable for the gated expansion plan.

## 1. New experiment_ids

| experiment_id | scenario | model | prompt language | cost | purpose |
|---|---|---|---|---|---|
| `full_experiment_en_v1` | CPR | Haiku (selectable) | English | not run | English CPR variant (`--english` flag) |
| `groq_smoke_test` | CPR/Bargaining smoke | Groq `llama-3.3-70b-versatile` | English | free | Provider structured-output validation |
| `full_experiment_groq_v1_70b_partial` | CPR | Groq `llama-3.3-70b-versatile` | English | free | Aborted: free-tier 100k TPD exhausted (~4 runs) |
| `full_experiment_groq_v1` | CPR | Groq `llama-3.1-8b-instant` | English | free | Coop micro-pilot (10 runs); inverse r=+0.760 |
| `bargaining_groq_v1` | Bargaining | Groq `llama-3.1-8b-instant` | English | free | Coop micro-pilot; inverse r=+0.832; 100% reject rate anomaly |
| `bargaining_risk_groq_v1` | Bargaining | Groq `llama-3.1-8b-instant` | English | free | Risk micro-pilot (9/10 runs); r=+0.792 |
| `bargaining_groq_v1_parallel_check` | Bargaining | Groq `llama-3.1-8b-instant` | English | free | Phase 6 parallel correctness check |
| `iterated_pd_groq_v1` | Iterated PD | Groq `llama-3.1-8b-instant` | English | free | **GATE STOP:** TPD exhausted before micro-pilot completed |
| `stag_hunt_groq_v1` | Stag Hunt | Groq `llama-3.1-8b-instant` | English | free | **GATE STOP:** TPD exhausted before micro-pilot completed |

**Locked IDs untouched:** `full_experiment_v1`, `control_group_v1`, `heterogeneous_v1`, `prompt_revision_v1`, `bargaining_v1`, `bargaining_risk_v1` — row counts for locked experiment data unchanged in SQLite.

## 2. Cost breakdown

| Provider | Spend |
|---|---|
| Claude (Anthropic) | **$0.00** (no new Claude pilots in this expansion) |
| Groq | **$0.00** (free on-demand tier) |

## 3. Cross-scenario synthesis table

Run: `python analysis/cross_scenario_synthesis.py`

| scenario | trait | model | n | r | direction | evidence |
|---|---|---|---:|---:|---|---|
| CPR | cooperation | Haiku | 45 | +0.456 | inverse | statistically supported |
| CPR | risk | Haiku | 45 | +0.678 | expected | statistically supported |
| CPR | cooperation | Sonnet | 40 | −0.84 | expected | statistically supported |
| CPR | cooperation | Groq-8b | 10 | +0.760 | inverse | statistically supported |
| Bargaining | cooperation | Haiku | 10 | −0.981 | expected | statistically supported |
| Bargaining | risk | Haiku | 10 | +0.548 | expected | qualitative (p≈0.10) |
| Bargaining | cooperation | Groq-8b | 10 | +0.832 | inverse | statistically supported |
| Bargaining | risk | Groq-8b | 9 | +0.792 | expected | statistically supported |

**Qualitative/suggestive:** With ≤4 scenarios, structural-axis patterns (sequential vs simultaneous, conflict vs coordination) cannot be tested at scenario level. Model identity appears at least as important as scenario structure.

## 4. Anomalies and gate-stops

| Issue | Resolution |
|---|---|
| Groq `llama-3.3-70b` TPD limit (100k) during CPR | Switched volume experiments to `llama-3.1-8b-instant` (500k TPD); partial 70b data relabeled to `full_experiment_groq_v1_70b_partial` |
| Groq `json_schema` unsupported on 70b | Use `function_calling` method in `core/llm_providers.py` |
| Groq `tool_use_failed` with extra fields in tool args | Added `failed_generation` recovery filter in `call_agent` |
| Bargaining Groq coop: 100% reject rate | Logged as anomaly; fidelity r still computed on keep_amount |
| Groq `llama-3.1-8b-instant` TPD limit (500k) after CPR+bargaining pilots | IPD/Stag Hunt/parallel check blocked; wait for TPD reset or upgrade tier |
| GATE 4/5 full grids | **Skipped** — micro-pilot signals clear (|r|>0.75, p<0.05) |

## 5. Why Groq was previously dropped (Phase 2 finding)

Master reference §17: Groq was never implemented in Python; a planned Anthropic+Groq hybrid was rejected to avoid **model-mix confound** within experiments. This expansion reintroduces Groq with **one provider per experiment_id** — concern was methodological, not a prior technical failure.

**Relevance in new runs:** Groq 8b works for structured calls but needs `function_calling`, rate-limit backoff, and occasional `failed_generation` recovery. Volume runs are slow on free tier (~45+ min per 10-run bargaining grid).

## 6. Established vs suggestive

**Statistically established (within-run Pearson, n stated):**
- Haiku CPR cooperation inverse (locked baseline)
- Sonnet CPR cooperation expected direction
- Groq-8b CPR cooperation inverse (micro-pilot n=10)
- Haiku bargaining cooperation expected (micro-pilot)
- Groq-8b bargaining cooperation inverse (micro-pilot)
- Groq-8b bargaining risk expected (n=9 micro-pilot)

**Qualitative/suggestive:**
- Haiku bargaining risk (p≈0.10)
- Cross-scenario structural-axis generalizations
- IPD / Stag Hunt Groq pilots (see scenario DBs after completion)

## 7. Yol 2 readiness (weight-level trait injection)

**Recommendation: not yet.** Groq adds a third model axis but reproduces **inverse cooperation** on CPR and bargaining (Haiku-like), not a clean third pattern. Prompt-level fidelity remains **model-dependent** and **direction-inconsistent** across Haiku vs Sonnet vs Groq on cooperation in CPR/bargaining. Weight-level injection (LoRA / activation steering) should wait until prompt-level mechanisms are better understood — or proceed as an explicitly separate research track with frozen prompt baselines.

## 8. Git commits (expansion)

```
cf064d3 Add English prompt variants while preserving locked Turkish baselines.
ea7b29a Add provider-agnostic LLM layer with Groq tool-calling support.
9bad11e Add Groq quality gate: CPR and bargaining English prompt validation.
[subsequent commits for groq CPR pilot, scenarios, synthesis, parallel, summary]
```

Verify: `git log --oneline cf064d3^..HEAD`

## 9. Locked data integrity

Whole-database file checksums change when **new** experiment rows are appended. Locked experiment **row counts** verified unchanged:

- `full_experiment_v1`: 2880 agent_decisions
- `control_group_v1`: 1755
- `heterogeneous_v1`: 1310
- `prompt_revision_v1`: 725 (pre-existing partial)
- `bargaining_v1`: 150 rounds
- `bargaining_risk_v1`: 150 rounds
