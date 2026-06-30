# AMADS — Cross-Model Analysis Plan (prereg-lite)

**Status:** Approved, 2026-06-29  
**Purpose:** Test whether the primary findings are not solely a `claude-haiku-4-5` artifact—specifically, whether directional effects replicate in at least one additional Claude model.

---

## Locked (not re-analyzed)

- `full_experiment_v1` (Haiku, 45 runs) — primary trait fidelity findings
- `control_group_v1`, clustering, `data/synthesis_report.md`

---

## Confirmatory micro-replication

| Parameter | Value |
|---|---|
| `experiment_id` | `sonnet_crossmodel_v1` |
| Model | `claude-sonnet-4-6` (Sonnet 4.6) |
| Script | `analysis/prompt_ab_sonnet_crossmodel.py` |
| Design | cooperation ∈ {0.2, 0.8} × risk_tolerance ∈ {0.2, 0.8} |
| Replicates / cell | n = 10 |
| Total LLM calls | 40 (single agent, single round) |
| Prompt | Turkish — same text as `decision_agent._build_system_prompt` |
| temperature | 0.2 |
| Cost cap | $2.00 |
| DB | `data/sonnet_crossmodel_v1.csv` (not written to results.db) |

---

## Hypotheses (directional test, no p-value claim)

| ID | Hypothesis | Haiku reference (round 0) |
|---|---|---|
| H1 | cooperation_assigned ↑ → extraction_fraction ↑ (inverse fidelity) | r ≈ +0.46 |
| H2 | risk_tolerance_assigned ↑ → extraction_fraction ↑ | r ≈ +0.68 |

**Success criterion (directional):** In Sonnet, H1 and H2 Pearson r signs match Haiku (positive).

---

## Excluded metrics

- Within-run Gini (homogeneous agent design)
- `cooperation_score_avg` literature comparison
- Full 45-run factorial replication (future work)

---

## Output

- `data/sonnet_crossmodel_v1.csv`
- Terminal summary: cell means, r values, estimated cost
