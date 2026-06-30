# Trait Fidelity in LLM-Based Multi-Agent Simulations: Model-Specific Behavioral Divergence in Commons Dilemma Scenarios

# Abstract

We built AMADS (Academic Multi-Agent Decision Simulation), a framework to test whether AI agents given a personality trait in their instructions actually behave according to that trait, using a shared-resource extraction game as the test scenario. We found that one trait worked as expected—higher assigned risk tolerance predicted more extraction—but a "cooperative" trait backfired: agents told to be more cooperative actually took more from the shared pool, not less. AMADS is a LangGraph commons-dilemma framework where LLMs output structured extraction decisions and a deterministic, LLM-free referee computes all outcomes—eliminating circular LLM-as-judge evaluation (LLM rates its own output). In `full_experiment_v1` (45 runs; 9 trait conditions × 5 replicates; Claude Haiku 4.5; Turkish prompts), `risk_tolerance_assigned` transfers reliably (*r* = +0.678, *p* < 0.0001), while `cooperation_assigned` exhibits inverse fidelity (*r* = +0.456, *p* = 0.0016; trait effect reversed): higher assigned cooperation predicts greater extraction. Trait-fidelity analysis (assigned-vs-observed correlation), blind k-means clustering, and a formula-based control agent provide convergent validity (independent confirmation). Eleven-language and eleven-trait pilot screens reveal no stable category rule for transfer; Sonnet 4.6 cross-model replication (*n* = 40) reverses the cooperation pattern (*r* ≈ −0.84) while weakening risk (*r* ≈ +0.15). Trait fidelity (trait-to-behavior match) is empirically idiosyncratic and model-specific—each trait–model–prompt combination requires validation before simulation use.

---

# 1. Introduction

Commons dilemmas—in which self-interested extraction threatens a shared resource—are a foundational testbed for studying collective action under scarcity. When researchers embed psychological traits in LLM agent prompts, a central empirical question arises: does the assigned trait govern measurable behavior, or only surface-level language?

The **trait injection problem** is central: when a researcher assigns `cooperation = 0.8` in a system prompt, does the agent extract less from a shared pool than when assigned `cooperation = 0.2`? Without independent, code-computed behavioral metrics, one cannot distinguish faithful trait transfer from concept misalignment, heuristic defaults, or model-specific reinterpretation of trait labels.

This paper makes three contributions:

1. **AMADS**, an open architecture for commons-dilemma multi-agent simulation with deterministic evaluation, parallel agent fan-out, and read-only agent views—so that every reported metric is computed without LLM involvement.
2. **Empirical trait-fidelity evidence** from a preregistered-style factorial experiment (`full_experiment_v1`, *N* = 45) showing dissociated transfer for cooperation (inverse) versus risk tolerance (expected direction), validated by clustering and a formula-based control group.
3. **Generalization and cross-model replication** demonstrating that neither language nor trait category predicts fidelity, and that the same Turkish prompt yields opposite cooperation behavior on Claude Sonnet 4.6 versus Haiku 4.5—establishing trait fidelity as **model-specific**, not universal.

---

# 2. Literature Review

Hardin (1968) argued that when individuals act on private incentives in a shared-resource setting, rational self-interest can drive collective ruin—the tragedy of the commons—establishing the conceptual foundation for commons-dilemma research. Ostrom (1990) showed that communities can design institutional rules—monitoring, graduated sanctions, and locally adapted governance—to sustain common-pool resources without either privatization or Leviathan control. Walker, Gardner, and Ostrom (1990) provided experimental evidence that rent dissipation in limited-access common-pool resources depends on institutional structure, demonstrating that extraction behavior is measurable and manipulable in laboratory CPR games. Nockur, Pfattheicher, and Keller (2023) extended CPR experimentation to asymmetric versus symmetric extraction opportunities, finding that privileged and underprivileged group members respond differently when consumption rules change—a design lineage AMADS follows in treating numeric extraction as the primary behavioral endpoint.

Park et al. (2023) introduced generative agents—LLM-based characters with memory, reflection, and planning—that produce believable emergent social behavior in a simulated town, inaugurating LLM social simulation as a research paradigm. Piatti et al. (2024) placed LLM agents in GovSim, a sustainability-focused society simulation, and found that cooperation can emerge or collapse depending on agent interaction dynamics and institutional interventions. Nguyen et al. (2025) proposed CFC-Prompt, a trait-conditioned prompting method for common-pool resource dilemmas presented at AAMAS 2025, treating natural-language trait descriptions as levers on cooperative decision-making in multi-agent CPR settings.

Zheng et al. (2023) formalized LLM-as-a-judge evaluation through MT-Bench and Chatbot Arena, showing that strong models such as GPT-4 can approximate human preference rankings with over 80% agreement—popularizing LLM-based scoring but also highlighting position and verbosity biases inherent in judge-model evaluation. Bhandari et al. (2025) assigned OCEAN personality profiles to dyadic LLM conversational agents and measured trait expression with independent LLM judges, finding that persona consistency varies substantially across model pairs, discourse settings, and trait polarity. Bodroža, Dinić, and Bojić (2024) administered personality instruments to seven LLMs at two time points and found limited temporal stability and variable inter-rater agreement, implying that self-reported trait scores on psychological scales may not reflect stable model dispositions. Caron and Srivastava (2023) demonstrated that prefix contexts can manipulate perceived Big Five traits in language models with correlations up to 0.84 between intended and realized trait shifts—establishing early evidence that persona prompting alters generated text, but not necessarily downstream numeric decisions.

Dubedy (2026) assigned socioeconomic personas to GPT-4.1 agents in a simulated gambling task and found that a "Poor" persona reported elevated risk perception while simultaneously making smaller bets—a dissociation between self-reported risk scores and behavioral bet sizes attributed to different components of model response logic. Hartley et al. (2025) investigated how Big Five personality interventions shape LLM risk-taking under cumulative prospect theory and found that trait–risk relationships established in one model generation fail to generalize consistently to other model versions, with legacy models such as GPT-4-Turbo showing unstable personality–risk mappings.

Despite this breadth of work, no prior study combines commons-dilemma extraction as the behavioral endpoint, deterministic code-computed metrics that exclude LLM judges, cross-model replication under identical prompts, and systematic cross-language trait screens within a single experimental program. Classical CPR experiments measure extraction directly; LLM simulation papers often infer cooperation from dialogue or LLM scoring; personality studies typically assess self-report or text style rather than structured numeric decisions in a shared-resource game. AMADS is designed to fill this intersection.

---

# 3. Methodology

## 3.1 AMADS Architecture

AMADS implements a commons-dilemma scenario as a LangGraph `StateGraph` with **parallel fan-out**: five agent nodes decide simultaneously each round without observing co-players' current-round actions, followed by a single **Referee** node that collects decisions, updates pool dynamics, applies scheduled shocks, and computes metrics. Round progression is counter-driven (`round_number < max_rounds`); termination occurs on pool collapse or round completion.

![Figure 1: AMADS agent fan-out and deterministic referee flow](../figures/architecture_diagram.png)

Agents receive only an `AgentInputView`—their own frozen `TraitProfile`, the current `EnvironmentSnapshot`, and `round_number`—not the full `SimulationState`. This is enforced at the function-signature level, not merely in prompts.

## 3.2 Why an LLM-Free Referee

A recurring failure mode in LLM simulation research is **circular evaluation**: the same model (or another LLM) judges behavior it or a sibling model produced. AMADS separates decision generation from measurement entirely. The Referee performs deterministic math only—pool update, shock application, Gini computation, cooperation scoring, violation counting—and **never calls an LLM**. LLMs produce structured `AgentDecision` records; all inferential statistics derive from numeric fields computed in code.

## 3.3 State Structure and AgentDecision Schema

State updates flow through Pydantic v2 models in `core/state.py`:

| Layer | Key types | Write policy |
|---|---|---|
| Frozen inputs | `TraitProfile`, `ShockEvent` schedule | Set at init; never mutated |
| Agent output | `AgentDecision` (appended per round) | Agent nodes append only |
| Environment | `EnvironmentSnapshot` | Referee only |
| Metrics | `MetricsSnapshot` (history list) | Referee only |

Each round, an agent outputs:

```python
class AgentDecision(BaseModel):
    agent_id: str
    round_number: int
    extraction_amount: float = Field(ge=0.0)
    justification: str = Field(max_length=500)  # logging only
    declared_max: float = Field(ge=0.0)
```

`justification` is excluded from all metric formulas. Observed cooperation is computed as `1 − (extraction / max_extractable)`; observed risk-related behavior uses extraction variance in post-shock rounds. **Trait–behavior fidelity** is Pearson *r* between assigned trait values and these code-derived quantities.

Pool dynamics follow a fixed recurrence:  
`pool[t+1] = clamp(pool[t] − Σ extraction_i[t], 0, capacity) × regen_rate`.

## 3.4 Primary Experiment Design (`full_experiment_v1`)

| Parameter | Value |
|---|---|
| Traits | `cooperation_assigned`, `risk_tolerance_assigned` |
| Levels per trait | Low = 0.2, Medium = 0.5, High = 0.8 |
| Conditions | 3 × 3 = **9** (full factorial) |
| Replicates per condition | **5** → **45 runs** total |
| Agents per run | 5 (homogeneous: identical trait pair for all agents in a condition) |
| Rounds | 15; shock at round 7 (`CAPACITY_DROP`, −20%) |
| Model | `claude-haiku-4-5-20251001` |
| Prompt language | Turkish (trait field names and structured output keys in English) |
| Outcome | 45/45 runs finished; all terminated via resource collapse (rounds 10–15) |

Trait values were injected via a locked symmetric system prompt (`agents/decision_agent.py`):

> *"Sen bir ortak kaynak havuzundan çekim yapan bir agentsın. cooperation_assigned değerin {cooperation_assigned:.2f} (0=tamamen bencil, 1=tamamen işbirlikçi), risk_tolerance_assigned değerin {risk_tolerance_assigned:.2f} (0=çok temkinli, 1=çok risk alan). Kararını bu eğilimlere uygun ver, ama bu sayıları çıktında tekrar etme veya açıklama."*

Primary fidelity analyses use **round 0** extraction fractions to avoid confounding with collapse timing (round-averaged cooperation scores correlate *r* ≈ 0.95 with `collapse_round`).

## 3.5 Calibration Decisions

| Parameter | Locked value | Rationale (one sentence) |
|---|---|---|
| `EXTRACTION_LIMIT_RATIO` | **0.12** | Real-LLM calibration showed that 0.12 guarantees shock exposure at round 7 while preventing spurious 15-round completion under aggressive extraction. |
| `COLLAPSE_EPSILON_RATIO` | **0.01** | Float pool values asymptotically approach but never reach exact zero; treating `pool ≤ 1%` of capacity as collapse fixes misclassified "completed" dead runs. |
| `temperature` | **0.2** | Fixed across all runs to reduce stochastic variance without claiming full reproducibility; observed Gini variance (σ² ≈ 4.78×10⁻⁴) already provided sufficient between-run signal. |

Mock-agent calibration at ratio 0.30 appeared viable but failed on real LLMs (0% round-7 shock exposure), underscoring that parameter tuning must be confirmed with actual model calls.

## 3.6 Power Analysis

Using δ = 0.05 as a uniform minimum detectable difference:

- **Gini coefficient** (σ ≈ 0.0197): δ corresponds to a large standardized effect (*d* ≈ 2.5); *N* = 45 is **more than sufficient**.
- **Cooperation score average** (σ ≈ 0.108): δ corresponds to a small effect (*d* ≈ 0.46); *N* = 45 is **insufficient** (~75 replicates per condition would have been needed for conventional power).

We therefore report cooperation-related inferential results with effect sizes and *p*-values while noting limited power for condition-level mean comparisons on cooperation score averages.

---

# 4. Experimental Results

## 4.1 Primary Findings — Haiku

### Cooperation: Inverse Fidelity

On round-0 extraction fractions (*n* = 45):

- **Cooperation → extraction_fraction:** *r* = **+0.456**, *p* = **0.0016** (positive correlation = **inverse fidelity** relative to the experimental operationalization, where higher cooperation should predict *lower* extraction).

Marginal pattern (micro-A/B, risk fixed at 0.2): `cooperation = 0.2` → mean extraction ≈ 4.56; `cooperation = 0.8` → mean extraction ≈ 8.40. The effect persists when risk is held constant, ruling out a risk×cooperation interaction artifact.

An initial round-averaged cooperation score showed *r* = −0.345 (*p* = 0.02) but was confounded with collapse timing; round-0 unconfounded metrics supersede it.

### Risk Tolerance: Expected Direction

- **Risk → extraction_fraction:** *r* = **+0.678**, *p* **< 0.0001** (*n* = 45).

High assigned risk reliably predicts higher extraction fractions; no concept inversion was observed for this trait.

### Convergent Validity (Three Independent Methods)

| Method | Risk | Cooperation |
|---|---|---|
| **Trait fidelity** (Pearson *r*, round 0) | *r* = 0.678, *p* < 0.0001 | *r* = +0.456, *p* = 0.0016 (inverse) |
| **K-means clustering** (*k* = 2, *n* = 45, blind to traits) | Partial overlap (ARI vs. risk level = **0.2862**) | Does not separate (ARI vs. coop level = **0.1235**) |
| **Control group** (deterministic formula agent, *n* = 27) | N/A (risk not implemented in control) | *r* = **−1.000** (expected); LLM *r* = **+0.456** (opposite sign) |

*Clustering run (`analysis/clustering.py`, `full_experiment_v1`): silhouette-selected *k* = 2; combined 9-cell ARI = 0.1469 (low overall overlap).*

The deterministic control agent implements the design-intended mapping; mock agents in code likewise yield cooperation *r* = +1.0. Only real LLM agents invert cooperation—confirming the finding is behavioral, not implementation error. Prompt symmetry between cooperation and risk sentences was verified character-by-character.

### Concept Misalignment Mechanism

Qualitative review of `justification` texts (logging-only, not used in metrics) indicates the model interprets "cooperation" not as "extract less to preserve the commons" but as **"use one's own share responsibly and sustainably."** Representative paraphrases from agent outputs include:

- Framing high extraction (~70% of `declared_max`) as *"ensuring other agents also benefit."*
- Describing aggressive withdrawal as *"supporting sustainability"* of the shared pool.

Under this frame, a "cooperative" agent fulfills duty by extracting decisively yet "fairly," which aligns with **higher** numeric extraction—not lower—relative to our operational definition (`1 − extraction/max_extractable`).

## 4.2 Generalization (Language and Trait Pilots)

### Eleven-Language Pilot (*n* = 5 per cell per language; 110 calls total)

To test whether inverse cooperation fidelity is a Turkish translation artifact, we ran identical micro-A/B screens (risk = 0.2 fixed; cooperation ∈ {0.2, 0.8}) in 11 languages with Haiku 4.5. Classification used |difference| ≥ 0.30 between high- and low-cooperation mean extraction.

![Figure 2: Cooperation extraction difference by language (coop=0.8 − coop=0.2)](../figures/multilang_results.png)

| Language | coop=0.2 avg. | coop=0.8 avg. | Difference | Class |
|---|---|---|---|---|
| **tr** | 4.80 | 8.40 | **+3.60** | inverse fidelity |
| zh | 8.40 | 9.60 | +1.20 | inverse fidelity |
| es | 8.94 | 9.60 | +0.66 | inverse fidelity |
| en | 9.12 | 9.60 | +0.48 | inverse fidelity |
| ja | 9.12 | 9.60 | +0.48 | inverse fidelity |
| hi | 9.16 | 9.60 | +0.44 | inverse fidelity |
| fr | 9.76 | 9.60 | −0.16 | trait-blind |
| ru | 9.60 | 9.60 | 0.00 | trait-blind |
| de | 8.16 → 8.88* | 7.68 → 9.12* | −0.48 → **−0.24*** | trait-blind at *n*=10 |
| pt | 9.60 → 9.36* | 9.12 → 9.60* | −0.48 → **−0.24*** | trait-blind at *n*=10 |
| ar | 10.30 → 10.45* | 9.60 → 9.85* | −0.70 → **−0.60*** | borderline expected |

\*Extended validation (*n* = 10/cell) for de, pt, ar.

**Finding:** Turkish shows the **strongest and most consistent** inverse fidelity (+3.60)—not a translation bug but the clearest signal. Six additional languages show weak inverse direction; two are trait-blind. Initial "expected direction" labels for German and Portuguese collapse to trait-blind at *n* = 10. **No language yields reliable expected-direction cooperation fidelity** at pilot scale.

### Eleven-Trait Pilot (~100 calls; English prompts)

Nine candidate traits beyond cooperation and risk were screened (all others fixed at 0.5; target trait ∈ {0.2, 0.8}; *n* = 5/cell).

| Class | Traits | Evidence |
|---|---|---|
| **Working** (4) | `risk_tolerance`, `aggression`, `fairness`, `greed` | Numeric separation in expected direction |
| **Unreliable / trait-blind** (7) | `cooperation` (inverse), `impatience`, `tolerance`, `creativity`, `hoarding`, `trust`, `caution` | No separation or wrong direction |

**No category rule holds:** early hypothesis that action-oriented traits transfer reliably was **refuted** by `caution_assigned`—defined as the inverse frame of risk (0 = not cautious, 1 = very cautious)—which produced **trait-blind** behavior (fixed extraction ≈ 6.0 at both levels) while `risk_tolerance` worked. Word choice and model heuristics dominate over abstract/concrete or action/value taxonomy.

*Note:* `hoarding` failure may reflect single-round design (stockpiling requires multi-round accumulation) rather than pure trait blindness.

## 4.3 Cross-Model Replication

### Sonnet 4.6 Pilot (`sonnet_crossmodel_v1`, *n* = 40)

| Parameter | Value |
|---|---|
| Model | `claude-sonnet-4-6` |
| Prompt | **Identical Turkish text** to Haiku experiments |
| Grid | cooperation × risk ∈ {0.2, 0.8}², 10 replicates/cell |
| Round | 0 only (single-round, no collapse confound) |

| Trait | Sonnet 4.6 (*n* = 40) | Haiku 4.5 (round 0, *n* = 45) |
|---|---|---|
| Cooperation → extraction | *r* ≈ **−0.84** (expected direction) | *r* ≈ **+0.46** (inverse fidelity) |
| Risk → extraction | *r* ≈ **+0.15** (weak) | *r* ≈ **+0.68** (strong) |

Sonnet marginal means (cooperation): low (0.2) ≈ 10.0 extraction vs. high (0.8) ≈ 5.35—high cooperation predicts **less** extraction, opposite to Haiku.

![Figure 3: Haiku vs. Sonnet trait-fidelity Pearson r (round 0)](../figures/haiku_sonnet_comparison.png)

### Haiku vs. Sonnet Comparison

| Dimension | Haiku 4.5 | Sonnet 4.6 |
|---|---|---|
| Cooperation fidelity | Inverse (*r* ≈ +0.46) | Expected (*r* ≈ −0.84) |
| Risk fidelity | Strong (*r* ≈ +0.68) | Weak (*r* ≈ +0.15) |
| Same prompt | Turkish, locked | Turkish, identical |
| Interpretation | Concept misalignment for cooperation | Cooperation aligns with operational definition |

**Primary cross-model contribution:** Trait fidelity is **model-specific**. The same trait labels, prompt template, and scenario can produce opposite behavioral signatures across models—a pattern invisible when studies report results from a single model family.

Whether the apparent cooperation–risk "trade-off" (strong risk / weak-inverse cooperation on Haiku vs. the reverse on Sonnet) is systematic or coincidental **cannot be determined with *N* = 2 models**.

---

# 5. Discussion

Independent evidence supports the generality of this dissociation between LLM self-report and behavior. Dubedy (2026) found that GPT-4.1 agents assigned a 'Poor' socioeconomic persona reported elevated risk perception while simultaneously making smaller bets in a gambling task—a within-persona negative correlation (ρ = −0.410, p < 2.2×10⁻¹⁶) attributed to self-reported risk score and bet-size decisions being generated by different components of model response logic. This mirrors our cooperation concept misalignment: a trait label is verbally acknowledged but does not consistently govern the corresponding numeric decision. Separately, recent work on personality-conditioned risk-taking (arXiv:2503.04735) reports that trait-risk relationships established in one model generation fail to generalize consistently to other model versions, corroborating our finding that trait fidelity is model-specific rather than a stable property of the LLM paradigm.

## 5.1 Practical Recommendation

Researchers should treat every candidate trait—and every model–trait–prompt combination—as **hypothesis requiring empirical micro-validation** (single-round A/B extraction screens) before committing to full factorial experiments. Category intuition (abstract vs. concrete, action vs. value, synonym pairs like risk/caution) does not predict transfer success. AMADS follows the human CPR literature in treating numeric extraction from a shared pool as the primary behavioral endpoint (Nockur et al., 2023), but our results show that assigning a cooperation label in prompt does not guarantee alignment with that operationalization in LLM agents.

## 5.2 Limitations

- **Models:** Primary data from one model (Haiku 4.5); cross-model evidence from one additional model (Sonnet 4.6) only.
- **Scenario:** Single commons-dilemma formulation; generalization to bargaining or crisis scenarios is untested.
- **Agent design:** Homogeneous trait assignment within each run (all five agents share the same profile); heterogeneous populations may differ.
- **Language:** Main experiment used Turkish prompts; multilingual pilots were short-screen pilots, not full 45-run replications.
- **Power:** Cooperation score averages underpowered at *N* = 5/condition; round-0 fidelity metrics are primary inferential evidence.
- **Shock analysis:** Post-shock risk operationalization deferred; round-0 findings are the locked primary result.

## 5.3 Transparency: `experiment_conditions` Migration

Run-to-trait mappings for `full_experiment_v1` and `control_group_v1` are stored in an `experiment_conditions` database table that underwent a post-hoc schema migration (`core/database.py`) to normalize condition identifiers; all analyses join through this table and remain consistent with locked findings.

## 5.4 Future Work

- **Cross-model panel:** Replicate the full 9×5 factorial on ≥3 models (e.g., Opus, GPT-4 class) under preregistered analysis plans to test whether cooperation–risk trade-offs are systematic.
- **Heterogeneous agent design:** Assign mixed trait profiles within runs to test whether homogeneous conditioning amplifies misalignment.
- **Prompt revision experiments:** Behavioral rewording of cooperation (e.g., "0 = extract maximally, 1 = extract minimally") as a **separate** `experiment_id`—not retroactive relabeling of `full_experiment_v1`.

---

# 6. Conclusion

We built AMADS to measure LLM agent behavior in commons dilemmas through deterministic, non-circular evaluation, and ran a 45-run factorial experiment plus convergent validation, multilingual/trait pilots, and cross-model replication. Haiku 4.5 shows reliable risk transfer but inverse cooperation fidelity driven by concept misalignment; neither language nor trait category predicts success, and Sonnet 4.6 reverses the cooperation pattern under the same prompt. We recommend empirical micro-validation of every trait–model pair before using LLM agents as stand-ins for psychologically profiled populations in simulation research.

---

# References

*(DOIs provided where available; arXiv preprints noted.)*

- Bhandari, P., Fay, N., Wise, M. J., Datta, A., Meek, S., Naseem, U., & Nasim, M. (2025). Can LLM Agents Maintain a Persona in Discourse? In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (pp. 29213–29229). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.emnlp-main.1487
- Bodroža, B., Dinić, B. M., & Bojić, L. (2024). Personality testing of large language models: limited temporal stability, but highlighted prosociality. *Royal Society Open Science*, 11(8), 240180. https://doi.org/10.1098/rsos.240180
- Caron, G., & Srivastava, S. (2023). Manipulating the Perceived Personality Traits of Language Models. In *Findings of the Association for Computational Linguistics: EMNLP 2023* (pp. 2370–2386). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.findings-emnlp.156
- Dubedy, S. (2026). Persona-Conditioned Risk Behavior in Large Language Models: A Simulated Gambling Study with GPT-4.1. *arXiv:2603.15831*.
- Hardin, G. (1968). The tragedy of the commons. *Science*, 162(3859), 1243–1248.
- Hartley, J., Hamill, C., Seddon, D., Batra, D., Okhrati, R., & Khraishi, R. (2025). How Personality Traits Shape LLM Risk-Taking Behaviour. In *Findings of the Association for Computational Linguistics: ACL 2025* (pp. 21068–21092). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.findings-acl.1085
- Nockur, L., Pfattheicher, S., & Keller, J. (2023). From asymmetric to symmetric consumption opportunities: Extractions from common resources by privileged and underprivileged group members. *Group Processes & Intergroup Relations*, 26(8), 1819–1840. https://doi.org/10.1177/13684302221132722
- Nguyen, D., Le, H., Do, K., Gupta, S., Venkatesh, S., & Tran, T. (2025). Navigating Social Dilemmas with LLM-based Agents via Consideration of Future Consequences: Extended Abstract. In *Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)* (pp. 2693–2695). IFAAMAS.
- Ostrom, E. (1990). *Governing the Commons: The Evolution of Institutions for Collective Action*. Cambridge University Press.
- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology* (UIST '23). https://doi.org/10.1145/3586183.3606763
- Piatti, G., Rane, A., Shi, W., Hofstätter, F., Li, B., Gao, Y., Bernstein, A., & Sumita, E. (2024). Cooperate or Collapse: Emergence of Sustainability Behaviors in a Society of LLM Agents. *arXiv:2404.16698*.
- Walker, J. M., Gardner, R., & Ostrom, E. (1990). Rent dissipation in a limited-access common-pool resource: Experimental evidence. *Journal of Environmental Economics and Management*, 19(3), 203–211. https://doi.org/10.1016/0095-0696(90)90008-2
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*. https://arxiv.org/abs/2306.05685
