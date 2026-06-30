# AMADS — Master Architecture Reference
> This file is the single source of truth for all architectural decisions in the project. Reference it when writing documentation or prompting code in Cursor. If a decision is not documented here, it has not yet been made officially.

---

## 1. Project Vision

**Name:** AMADS (Academic Multi-Agent Decision Simulation)

**Purpose:** To analyze decisions made by LLM-based agents with different psychological trait profiles (risk_tolerance, cooperation) in a constrained environmental scenario (Commons Dilemma) using a **measurable, deterministic, non-circular** evaluation layer.

**Core principle:** Not "the LLMs talked, something happened" — every outcome rests on a provable number computed in code. LLM calls are used **only to produce decisions**, never to **evaluate decisions**.

**Out of scope:** Open-ended natural-language bargaining, direct inter-agent messaging, LLM-as-judge evaluation.

---

## 2. Scenario: Commons Dilemma (Resource Sharing)

### 2.1. Why This Scenario
- 50+ years of literature support in game theory (Hardin 1968, Ostrom)
- `cooperation` and `risk_tolerance` traits can be operationalized naturally and numerically
- Convergence is guaranteed because the number of rounds can be fixed
- Natural alignment with the "Black Swan Events" (environment shock) roadmap

### 2.2. Pool Dynamics (Fully Deterministic, No LLM)
```
pool[t+1] = clamp(pool[t] - Σ(extraction_i[t]), 0, capacity) * regen_rate
```
- `capacity`: upper bound of the pool (fixed)
- `regen_rate`: regeneration coefficient (fixed, e.g. 1.15)
- `pool == 0` → simulation enters "collapse" state, early termination is triggered

### 2.3. Action Space (Agent's Sole Output)
Each round, each agent produces **only** the following (via structured output / function calling):
- `extraction_amount: float` — amount withdrawn from the pool
- `justification: str` — FOR LOGGING ONLY, does not enter any metric calculation
- `declared_max: float` — copy of the upper bound communicated to the agent that round (for violation checking)

`max_extractable_this_round` is computed by the environment at the start of each round (e.g. 40% of the current pool) — the agent cannot see or alter this limit; it only chooses within it.

---

## 3. Fixed Parameters (Pilot/Default Values)

| Parameter | Value | Note |
|---|---|---|
| Number of agents | **5** | Aligned with literature, sufficient diversity for clustering |
| Number of rounds | **15** | Leaves room for symmetric pre/post-shock comparison |
| Shock round | **~7-8** (exact number not yet chosen) | Open detail — see Section 11 |
| Determinism (temperature) | 0.2 | NOT a "reproducibility guarantee" — framed as stochastic variance reduction |
| Shock seed source | `experiment_id + run_id` | Same run_id produces the same shocks on every execution |

> These values should be held as easily changeable parameters in config, not hardcoded (`core/config.py`).

---

## 4. State Schema (Pydantic v2 — Full Reference)

### Layer Logic (Who Can Write What)
```
SimulationState
├── environment: EnvironmentSnapshot       # ONLY Referee/Environment node writes
├── agent_traits: Dict[str, TraitProfile]   # Fixed, never changes (frozen)
├── round_decisions: List[AgentDecision]    # Agent nodes ONLY APPEND here
├── metrics_history: List[MetricsSnapshot]  # ONLY Referee writes
├── shock_log: List[ShockEvent]             # Precomputed, frozen
└── round_number: int                       # ONLY Referee increments
```

### 4.1. TraitProfile
```python
from pydantic import BaseModel, Field, ConfigDict

class TraitProfile(BaseModel):
    model_config = ConfigDict(frozen=True)
    agent_id: str
    cooperation_assigned: float = Field(ge=0.0, le=1.0)
    risk_tolerance_assigned: float = Field(ge=0.0, le=1.0)
    profile_label: str  # örn. "high_coop_low_risk"
```

### 4.2. EnvironmentSnapshot
```python
class EnvironmentSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    pool_current: float = Field(ge=0.0)
    pool_capacity: float = Field(gt=0.0)
    regen_rate: float = Field(gt=0.0)
    max_extractable_this_round: float = Field(ge=0.0)
    round_number: int = Field(ge=0)
    is_collapsed: bool = False
```

### 4.3. AgentDecision
```python
class AgentDecision(BaseModel):
    agent_id: str
    round_number: int
    extraction_amount: float = Field(ge=0.0)
    justification: str = Field(max_length=500)  # SADECE log
    declared_max: float = Field(ge=0.0)
```

### 4.4. ShockEvent
```python
from enum import Enum

class ShockType(str, Enum):
    CAPACITY_DROP = "capacity_drop"
    REGEN_BOOST = "regen_boost"
    DEMAND_SURGE = "demand_surge"

class ShockEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    round_number: int
    shock_type: ShockType
    magnitude: float  # örn. -0.30 -> %30 azalma
    seed_source: str  # audit için
```

### 4.5. MetricsSnapshot
```python
class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    round_number: int
    gini_coefficient: float = Field(ge=0.0, le=1.0)
    cooperation_score_avg: float = Field(ge=0.0, le=1.0)
    total_extraction: float = Field(ge=0.0)
    pool_after: float = Field(ge=0.0)
    constraint_violations: int = Field(ge=0, default=0)
```

### 4.6. SimulationState (Main State)
```python
from typing import Dict, List, Optional

class SimulationState(BaseModel):
    experiment_id: str
    run_id: str
    agent_traits: Dict[str, TraitProfile]
    shock_schedule: List[ShockEvent]
    max_rounds: int = Field(gt=0)

    environment: EnvironmentSnapshot
    round_decisions: List[AgentDecision] = []
    metrics_history: List[MetricsSnapshot] = []
    round_number: int = 0

    is_terminated: bool = False
    termination_reason: Optional[str] = None
```

### 4.7. Agent's Read-Only View (Read-Only Enforcement)
```python
class AgentInputView(BaseModel):
    """Agent node'una geçirilen, environment'ı DEĞİŞTİREMEYECEĞİ salt-okunur görünüm"""
    model_config = ConfigDict(frozen=True)
    own_trait: TraitProfile
    environment: EnvironmentSnapshot
    round_number: int
```
> The read-only environment constraint is **not** a prompt-level request — it is a **function-signature-level guarantee**. The agent node function sees only `AgentInputView`, not the full `SimulationState`.

---

## 5. Trait → Behavior Bridge (Operationalization)

| Trait | How it enters the prompt | Measurement (post-hoc, from code, no LLM) |
|---|---|---|
| cooperation (0-1) | Behavioral tendency tone in system prompt | Average of `1 - (agent_extraction / max_extractable)` |
| risk_tolerance (0-1) | Behavioral tendency tone in system prompt | Variance of extraction behavior in post-shock rounds |

> Critical point: the trait is only a **prompt input**; observed behavior is computed **independently** from the agent's numerical output. The correlation between the two (`Trait-Behavior Fidelity`) is one of the project's core findings.

**Open detail:** The actual system-prompt sentence that will convey trait values to the agent has not yet been written (see Section 11).

---

## 6. Metrics (Full Formal Definitions, No LLM)

| Metric | Formula / Definition |
|---|---|
| Sustainability Index | `collapse_round / total_rounds` (1 = never collapsed) |
| Gini Coefficient | standard Gini formula over the `extraction` array |
| Cooperation Score | round-level average (formula in Section 5) |
| Shock Resilience | pre/post-shock `pool` recovery speed (number of rounds) |
| Trait-Behavior Fidelity | `corr(assigned_trait, observed_behavior)` — Pearson correlation |
| Constraint Violations | count of `extraction_amount > declared_max` |

---

## 7. Referee Node — Role

The Referee **does not interpret; it records and enforces rules**:
1. Collects all `AgentDecision` records for the round
2. Applies the pool formula (Section 2.2)
3. Checks the shock schedule and applies shocks if present
4. Computes metrics (Section 6)
5. Increments `round_number` and checks termination conditions

**The Referee makes no LLM calls.** This is the foundation of the project's "no circular evaluation" claim.

---

## 8. LangGraph Architecture (Technical Layer)

### 8.1. Topology: Parallel Fan-out + Reduce
```
                  ┌─→ Agent1 (isolated, parallel) ─┐
Round Start   ───┼─→ Agent2 (isolated, parallel) ─┼─→ Referee (collect, compute, advance)
                  └─→ Agent3...Agent5 ─────────┘
```
- Each agent runs in parallel **without seeing the others that round** → true simultaneous-game rule
- Round progression depends on no LLM decision; it is controlled by a fixed counter (`round_number < max_rounds`)

### 8.2. Technical Decisions
| Decision | Choice | Rationale |
|---|---|---|
| Graph construction style | **StateGraph** (not Functional API) | Each node is visible separately; high LangSmith trace compatibility |
| Parallel fan-out method | **Static parallel edges** (not `Send()` API) | Agent count is fixed (5); no need for dynamic branching |
| State update (reducer) | `round_decisions: Annotated[list, operator.add]` | No collision when 5 agents write in parallel; each appends |
| `environment`, `metrics_history`, `round_number` | NO reducer; single writer (Referee) | Only one node writes anyway |
| Checkpointing backend | **SqliteSaver** | Consistent with existing SQLite decision; `run_id` ↔ `thread_id` mapping |

### 8.3. Conditional Edge Logic (LLM-free Branching)
```
Referee → [pool == 0 ?] → Terminate (termination_reason="collapse")
        → [round_number >= max_rounds ?] → Terminate (termination_reason="completed")
        → [else] → New Round (return to Agent fan-out)
```

### 8.4. Subgraph Proposal (Per Agent)
`decide → self_check_constraint → finalize` — an extra step in which the agent self-checks whether it exceeded its own `declared_max` (not yet detailed; see Section 11).

---

## 9. Implementation Order (Steps to Follow in Cursor)

1. **Skeleton with mock nodes** — test the graph end-to-end with stubs that produce fixed/random `AgentDecision` instead of a real LLM (zero cost)
2. **Conditional edge validation** — test `pool==0` and `round>=max_rounds` with mock data
3. **Checkpointing test** — cut a run midway and resume via `thread_id`
4. **Connect one agent to a real LLM** — not all 5 at once; start with 1 (cost control)
5. **Enable LangSmith tracing** — prove the "no circular evaluation" claim with traces
6. **Connect the remaining 4 agents and run a full experiment**

> Do not proceed to the next step until the current one works — layering debug complexity is the main cause of spaghetti.

---

## 10. Cost and Resource Management

### 10.1. Principles
1. **Dry-run/mock mode** is mandatory (step 1 above)
2. **Cost estimation script** — pre-check that estimates tokens before large N
3. **Tiered model strategy** — pilot/debug: Haiku 4.5; statistical final runs: Sonnet 4.6
4. **Hard caps** — `max_rounds`, `max_agents`, `max_runs_per_batch` as fixed ceilings at config level

### 10.2. Current Prices (June 2026, per MTok)
| Model | Input | Output |
|---|---|---|
| Haiku 4.5 | $1.00 | $5.00 |
| Sonnet 4.6 | $3.00 | $15.00 |

### 10.3. Single-Run Estimate (5 agents × 15 rounds = 75 calls, stateless agents)
| | Sonnet 4.6 | Haiku 4.5 |
|---|---|---|
| Per call (~1200 input / ~200 output tokens) | ~$0.0066 | ~$0.0022 |
| **Full run (75 calls)** | **~$0.50** | **~$0.17** |

> The Referee contributes zero to cost because it makes no LLM calls. Estimates may deviate by ±30-50%; they should be calibrated with real measurements.

---

## 11. Open Technical Details — Status (Updated After Phase 2)

1. ~~Exact shock round number~~ → **LOCKED: round 7**, `CAPACITY_DROP` (-20%), validated across all calibration and `full_experiment_v1` runs (evidenced by LangSmith trace + pool_after break).
2. ~~Actual system-prompt sentence conveying trait values to the agent~~ → **LOCKED**, see Section 18.2 (full text + finding).
3. Exact logic of the subgraph's `self_check_constraint` step → **still open**, not yet addressed (constraint violation is currently counted post-hoc only in the referee; agent self-constraint was not implemented as a separate mechanism).
4. ~~Exact SQLite checkpoint table schema~~ → **LOCKED AND TESTED**: LangGraph's own `SqliteSaver` checkpoint file (separate file, independent of `data/results.db`). Checkpointing test (cut at round 5 and resume via `thread_id`) validated successfully — resumed from round 5, not round 0; `metrics_history` for rounds 0-4 preserved exactly.

## 12. Deliberately Deferred Items — Status

1. ~~Statistical experimental design~~ → **LOCKED AND IMPLEMENTED**, see Section 18.3 (full factorial design, N, power analysis results).
2. **Test strategy** — unit/integration test coverage, which scenarios will be mocked → still open; no formal test strategy documented (progress made via ad-hoc validation scripts; see Section 18.5 — structural audit).

---

## 13. Supplementary Analysis Modules (Separate from Core, Cost-free)

These are applied to existing data **after** the core simulation finishes, with **no additional LLM calls**:

| Module | Question | Data Source | File |
|---|---|---|---|
| Control Group Agent | Does LLM behavior differ from a fixed-rule bot? | Raw extraction data from the same simulation (an `AgentDecision` producer that makes no LLM call) | `agents/control_agent.py` |
| Clustering | How many distinct strategy archetypes emerge spontaneously? | Behavior vectors, scikit-learn k-means | `analysis/clustering.py` |
| Synthesis Report | Are the three methods consistent? | fidelity + clustering + control-group outputs | `analysis/synthesis_report.py` → `data/synthesis_report.md` |

> These three form a staged narrative that builds on one another: first "what happened" (clustering), then "random or meaningful" (control group), then "do they all draw the same picture" (synthesis). **Implemented** — see `data/synthesis_report.md`.

---

## 14. Docker & Service Architecture

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.app
    volumes:
      - ./data:/app/data          # SQLite kalıcı
    env_file:
      - .env                       # ANTHROPIC_API_KEY, LANGSMITH_API_KEY, LANGSMITH_TRACING=true
    command: python run_simulation.py

  analytics:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    volumes:
      - ./data:/app/data:ro        # SADECE okuma — analiz katmanı veriye yazamaz
    ports:
      - "8501:8501"
    depends_on:
      - app
```

| Service | Status |
|---|---|
| app | LangGraph runtime, single container |
| analytics | Streamlit, read-only data access |
| db (SQLite) | NOT a separate container — file-based, shared via volume |
| ChromaDB | Not included — will not be added until isolation risk is resolved |
| LangSmith | Not a container — Cloud SaaS, API key only via `.env` |

---

## 15. Folder Structure (Skeleton)

```
amads/
├── core/
│   ├── state.py          # Bölüm 4'teki tüm Pydantic şemaları
│   ├── graph.py           # LangGraph StateGraph kurulumu
│   └── config.py          # max_rounds, agent sayısı, hard cap'ler
├── agents/
│   ├── decision_agent.py  # Gerçek LLM agent
│   └── control_agent.py   # Kontrol grubu (LLM'siz, sabit kural)
├── referee/
│   └── referee_node.py    # Deterministik hesaplama, LLM çağrısı YOK
├── environment/
│   └── shocks.py           # Seedli şok takvimi üretici
├── analysis/
│   ├── clustering.py
│   ├── trait_fidelity.py
│   ├── synthesis_report.py
│   └── export_experiment_summary.py
├── tests/
│   └── (mock/dry-run senaryoları)
├── docker-compose.yml
├── Dockerfile.app
├── Dockerfile.streamlit
├── CLAUDE.md               # Cursor için kural dosyası
└── README.md
```

---

## 16. Documentation Set (17 Documents, To Be Produced)

| # | Document | Group |
|---|---|---|
| 1 | Project Charter / Vision Doc | Overview |
| 2 | README.md | Overview |
| 3 | Glossary | Overview |
| 4 | System Architecture Document | Architecture |
| 5 | Architecture Decision Records (ADR) | Architecture |
| 6 | Data/State Schema Reference | Architecture |
| 7 | LangGraph Flow & Node Contract Doc | Core |
| 8 | Sequence/Lifecycle Doc | Core |
| 9 | Experimental Design Doc | Academic (last) |
| 10 | Metrics & Operationalization Reference | Academic |
| 11 | Analysis Modules Doc | Academic |
| 12 | Docker & Deployment Guide | Operations |
| 13 | Cost & Resource Management Doc | Operations |
| 14 | Testing Strategy Doc | Quality |
| 15 | CLAUDE.md / Cursor Rules | Quality |
| 16 | Contribution Guide | Contribution |
| 17 | Changelog | Contribution |

> Places requiring diagrams (4, 7, 8) will be produced with a visual tool; text portions will be derived from this reference.

---

## 17. Core Decision History (Summary — "Why This Way" Reminder)

| Decision | Alternatives | Why This Was Chosen |
|---|---|---|
| Commons Dilemma scenario | Bargaining, Crisis | Strongest literature support, most natural trait operationalization |
| Structured output (numeric decision) | Open-ended natural language | Prevents circular evaluation; objectively readable in code |
| Parallel fan-out | Sequential chat, Supervisor | True simultaneous game, convergence guarantee, reproducibility |
| Deterministic Referee | LLM-as-judge | Makes the core promise "subjective not quantitative" real |
| SQLite | Postgres | Sufficient for single user/single machine, simple |
| StateGraph | Functional API | Transparency, LangSmith trace compatibility |
| Static parallel edge | `Send()` API | Fixed agent count, no unnecessary flexibility |
| Temperature = 0.2 (fixed, unchanged) | 0.35-0.5 ("controlled stochasticity") | Real variance data (gini σ²≈4.78e-4) already provided sufficient signal; no need to weaken the determinism principle (Section 18.3) |
| EXTRACTION_LIMIT_RATIO = 0.12 | 0.40 (initial assumption), 0.15, 0.30 | Selected via mock + real LLM calibration: balanced point that always sees round 7 shock but never completes to 15 (Section 18.1) |
| COLLAPSE_EPSILON_RATIO = 0.01 (new parameter) | exact `pool == 0` equality (old, incorrect) | Floats asymptotically approach zero but never reach exact zero; "functional death" was incorrectly counted as "completed" (Section 18.1) |
| Groq fully removed | Groq + Anthropic hybrid | Never existed in architecture; model mixing carried confound risk; cost savings were negligible |

---

## 18. Phase 2 — Calibration, First Experiment, and Findings (Log)

> This section is the full log of the transition from architectural design to real data collection. Sections 1-17 answered "what will we build"; this section answers "we built it, ran it, what did we learn".

### 18.1. Calibration Process and Bugs Resolved

**Epsilon bug (critical, affected data validity):**
The Referee's collapse check formerly looked for `pool_after <= 0` (exact mathematical zero). The pool formula (`clamp` + `regen_rate`) asymptotically shrinks by nature and never reaches exact zero (e.g. it could remain at a residual like 0.0000003 and be counted as "alive" forever). Result: some runs appeared "completed" at 15 rounds with `termination_reason="completed"` even though they were functionally dead (no agent could extract anything after round 3) — a measurement error that would systematically bias Sustainability Index and Cooperation Score.

*Fix:* `COLLAPSE_EPSILON_RATIO=0.01` added → `is_collapsed = pool_after <= pool_capacity * 0.01`. After the fix, the same runs were correctly marked as collapse in rounds 1-2.

**`_apply_shocks` was not running at all (bug):** Shock (CAPACITY_DROP, round 7, -20%) had no effect on the pool — the inter-round pool ratio was completely flat. Fix: shock now proportionally reduces both `pool_capacity` and `pool_current` (`scale = 1 + magnitude`). Validation: ratio dropped from 0.6325→0.5060 between rounds 6→7 (≈20% additional drop, as expected).

**EXTRACTION_LIMIT_RATIO calibration (mock → real LLM):**

| Ratio | Test type | R7 (shock exposure rate) | Done15 (completion) | Decision |
|---|---|---|---|---|
| 0.40 | Mock + real LLM | 0% | 0% | Too aggressive, rejected |
| 0.30 | Mock (n=40) | 37.5% | 22.5% | Promising in mock |
| 0.30 | **Real LLM (n=3)** | 0% (after epsilon fix) | 0% | Much more aggressive than mock — real LLMs behave differently from mock |
| 0.15 | Real LLM (n=3) | 100% | 0% | Always sees shock, always collapse rounds 8-9 |
| **0.12** | **Real LLM (n=3)** | **100%** | **0% (always collapse at round 12)** | **LOCKED** — shock always testable |

> Important note: mock calibration and real LLM behavior **did not align** (0.30 looked "good" in mock but gave 0% R7 on real LLM). This shows mock's random extraction (`uniform(0.05, 0.95)`) is structurally different from real LLM's trait-guided, more consistent behavior. **Conclusion: future parameter calibration must always be finally confirmed with real LLM; mock is only a coarse pre-screen.**

### 18.2. Trait → Prompt Operationalization (LOCKED)

`agents/decision_agent.py`, `_build_system_prompt()`:

```
"Sen bir ortak kaynak havuzundan çekim yapan bir agentsın.
cooperation_assigned değerin {cooperation_assigned:.2f}
(0=tamamen bencil, 1=tamamen işbirlikçi), risk_tolerance_assigned
değerin {risk_tolerance_assigned:.2f} (0=çok temkinli, 1=çok risk alan).
Kararını bu eğilimlere uygun ver, ama bu sayıları çıktında tekrar
etme veya açıklama."
```

This sentence was verified in the latest commit (git diff empty); coop and risk phrasing are symmetric in wording — an important reference point for proving in Section 18.4 that the finding is not due to prompt asymmetry.

### 18.3. Statistical Experimental Design (LOCKED, `full_experiment_v1`)

| Parameter | Value |
|---|---|
| Number of traits | 2 (`cooperation_assigned`, `risk_tolerance_assigned`) |
| Levels/trait | 3 (Low=0.2, Medium=0.5, High=0.8) |
| Number of conditions | 3×3 = 9 (full factorial; same trait pair for all 5 agents in each condition) |
| Replicates/condition (N) | 5 (45 runs total) |
| max_rounds | 15 |
| Model | claude-haiku-4-5-20251001 (pilot/calibration phase; Sonnet 4.6 planned for final statistical runs) |
| Outcome | 45/45 runs completed, $6.27 cost, all `collapse` (rounds 10-15), run_id↔trait mapping stored in DB via `experiment_conditions` table |

**Power analysis finding (important; must be reported honestly):**
The same absolute difference threshold (δ=0.05) yields different statistical power across metrics — because each metric has different natural variance:
- `gini_coefficient`: σ≈0.0197 → δ=0.05 is a large standardized effect (d≈2.5) → **N=45 more than sufficient**
- `cooperation_score_avg`: σ≈0.108 → δ=0.05 is a small standardized effect (d≈0.46) → **N=45 insufficient, ~75/condition would have been needed**

*Reporting implication:* Gini-based comparisons can be made with strong confidence; cooperation_score-based comparisons should be reported with effect size + confidence intervals, without a "p<0.05" claim.

### 18.4. LOCKED FINDING — Inverse Fidelity on the Cooperation Trait

> **Finding:** In `full_experiment_v1` (45 runs) and a confirmatory micro-A/B test (20 additional LLM calls, ~$0.04): a **consistent inverse relationship** was found between assigned `cooperation_assigned` trait and observed extraction behavior — agents assigned high cooperation **extract more**, agents assigned low cooperation **extract less**. `risk_tolerance_assigned` works in the expected direction (high risk → high extraction, consistent).

**Validation chain (in elimination order):**
1. Initial observation (45-run average): coop→cooperation_score_avg r=-0.345 (p=0.02) — but this metric was confounded with `collapse_round` (r≈0.95); round-average was misleading.
2. Reduced to round 0 (unconfounded): coop→extraction_fraction r=+0.456 (p=0.0016, n=45); risk→extraction_fraction r=+0.678 (p<0.0001).
3. **Implementation error ruled out:** trait assignment, agent_id mapping, DB join checked — no error.
4. **Compared with mock baseline:** mock agent (formula reflecting design intent) gives coop→score r=+1.0 — the system "knows" correct behavior; only the real LLM behaves differently. **This proves the LLM's behavior, not the code, is responsible.**
5. **Prompt asymmetry ruled out:** `_build_system_prompt` checked character by character via git diff; coop and risk sentences are symmetric.
6. **Raw data check (declared_max confound ruled out):** in risk=0.8 cells, `declared_max` is always 12.0 (fixed) — fraction differences come entirely from `extraction_amount`, not declared_max.
7. **Risk×coop interaction tested and ruled out:** 10-call micro-test with risk=0.2 (low, fixed) and only coop varied: coop=0.2 → avg. extraction=4.56, coop=0.8 → avg. extraction=8.40. Inverse effect is independent of risk level and present in every condition.
8. **Root cause diagnosed by reading justification texts:** the LLM interprets "cooperation" not as "using less of the resource" but as **"using my own share responsibly/sustainably"**. Justifications legitimize high extraction (~70% of declared_max) with phrases like "ensuring other agents also benefit" and "supporting sustainability".
9. **Third, independent validation (control group + k-means):** confirmed via convergent validity — see `data/synthesis_report.md`.

**Academic framing:** This is not a prompt-engineering error or implementation bug — it is **concept misalignment**: the behavioral frame the LLM assigns to the word "cooperation" does not overlap with the experiment's operational definition (Section 5). No such misalignment exists for `risk_tolerance`. **Generalizable inference: trait fidelity varies by trait type — action-oriented traits (risk) and abstract/value-laden traits (cooperation) are transferred to the LLM with different reliability.**

**This finding is locked and will not be re-questioned on current data/prompt.** If one wishes to clarify the prompt (e.g. behavioral wording like "0=extract as much as possible, 1=extract as little as possible, preserve others' share") and retest, that will be **a separate experiment** (different `experiment_id`); existing `full_experiment_v1` data/findings will not be altered.

#### 18.4.1 — Multilingual Validation (11-Language Pilot Screen)

**Purpose:** To test whether the Turkish "inverse fidelity" finding is a translation/language artifact.

**Method:** Same micro-A/B test (risk=0.2 fixed, cooperation ∈ {0.2, 0.8}, 5 replicates per cell = 10 calls per language), parallel pilot screen in 11 languages. Single agent, single round (round 0), `claude-haiku-4-5-20251001`, `temperature=0.2`. System and human prompts written in each language; trait field names (`cooperation_assigned`, `risk_tolerance_assigned`) and structured output fields (`extraction_amount`, `justification`, `declared_max`) remained English in all languages. Not written to `data/results.db`. Script: `analysis/prompt_ab_multilang.py`. Results: `data/multilang_results.csv`. Total 110 calls, cost ~$0.23 (limit $0.50).

**Classification rule** (difference = coop=0.8 avg. − coop=0.2 avg., threshold |difference| ≥ 0.30):
- `inverse fidelity`: difference > +0.30 (high cooperation → more extraction)
- `expected direction`: difference < −0.30 (high cooperation → less extraction)
- `trait-blind heuristic`: |difference| < 0.30 (cooperation not distinguished numerically)
- `other`: missing data / edge cases

**Results table (n=5/cell, pilot screen):**

| language | coop=0.2 avg. | coop=0.8 avg. | difference | class |
|---|---|---|---|---|
| **tr** | 4.80 | 8.40 | **+3.60** | inverse fidelity |
| zh | 8.40 | 9.60 | +1.20 | inverse fidelity |
| es | 8.94 | 9.60 | +0.66 | inverse fidelity |
| en | 9.12 | 9.60 | +0.48 | inverse fidelity |
| ja | 9.12 | 9.60 | +0.48 | inverse fidelity |
| hi | 9.16 | 9.60 | +0.44 | inverse fidelity |
| fr | 9.76 | 9.60 | −0.16 | trait-blind heuristic |
| ru | 9.60 | 9.60 | 0.00 | trait-blind heuristic |
| de | 8.16 | 7.68 | −0.48 | expected direction |
| pt | 9.60 | 9.12 | −0.48 | expected direction |
| ar | 10.30 | 9.60 | −0.70 | expected direction |

**Follow-up — borderline "expected direction" languages (de, pt, ar):** Three languages that appeared expected-direction in the first batch remained suspicious/borderline (|difference| ≈ 0.48–0.70). +10 additional calls per language (`analysis/prompt_ab_multilang_extend.py`, n=10/cell combined, cost ~$0.06):

| language | n=5 difference | n=10 difference | n=10 class |
|---|---|---|---|
| de | −0.48 | −0.24 | trait-blind heuristic |
| pt | −0.48 | −0.24 | trait-blind heuristic |
| ar | −0.70 | −0.60 | expected direction |

**Findings (in elimination order):**
1. **Not a Turkish artifact — inverse fidelity strongest in Turkish:** difference in `tr` (+3.60) is markedly larger than all other languages; coop=0.2 → 40% of max (4.8), coop=0.8 → 70% (8.4). Consistent with Turkish micro-test in Section 18.4 and `full_experiment_v1` round-0 data.
2. **Weak inverse direction in 6 languages:** `en`, `zh`, `es`, `ja`, `hi` show positive differences (+0.44 to +1.20) but most are near the 9.6 attractor; effect less clear than in Turkish.
3. **2 languages trait-blind:** `fr`, `ru` — cooperation groups not distinguished.
4. **"Expected direction" claim largely collapsed:** `de` and `pt` fell to ±0.24 at n=10 (trait-blind); initial batch −0.48 was borderline noise. `ar` retained negative difference at −0.60 at n=10, but coop=0.2 side shows excessive extraction near max (10.0–10.5) — readable as low-coop cell behaving more aggressively, not clean "cooperative = extract less" fidelity.
5. **English special case (together with prior micro-tests):** In English-only prompt, model sometimes locks to 80% fixed heuristic (9.6) (trait-blind); in multilingual run, weak inverse fidelity (+0.48) was observed. Language alone does not determine behavior; cooperation trait does not work reliably in the expected direction in any language.

**Conclusion:** Inverse fidelity in Turkish is **not** a translation error or language-specific artifact — on the contrary, in the 11-language pilot screen, **the most pronounced and most consistent** inverse fidelity appeared in Turkish. Unreliable transfer of the cooperation trait to the LLM is a language-universal problem; failure mode varies by language (Turkish: strong inverse fidelity; most other languages: weak inverse or trait-blind; borderline "expected direction" claims mostly reduced to trait-blind at n=10).

**This sub-finding supports the locked main finding in Section 18.4; it does not overturn it.** Multilingual pilot screen is kept as a separate experiment record (`experiment_id`: `_scratch_multilang`); `full_experiment_v1` data unchanged.

#### 18.4.2 — Trait Category Generalization Attempt (11-Trait Pilot Screen)

**Purpose:** To test whether cooperation inverse fidelity in Section 18.4 is specific only to the "abstract/value-laden trait" category — or whether concrete/action-oriented words (e.g. `risk_tolerance`) working reliably is a general rule.

**Method:** In addition to the original 2 traits (`cooperation_assigned`, `risk_tolerance_assigned`), **9 new category traits** were screened via micro-A/B: `aggression_assigned`, `fairness_assigned`, `impatience_assigned`, `tolerance_assigned`, `creativity_assigned`, `greed_assigned`, `hoarding_assigned`, `trust_assigned`, `caution_assigned`. For each trait, all other traits held at **0.5**, target trait at **{0.2, 0.8}**; **n=5** per cell (**n=10** combined for fairness with an extra validation batch). Single agent, single round (round 0), English system/human prompt, `claude-haiku-4-5-20251001`, `temperature=0.2`. Not written to `data/results.db`. Script: `analysis/trait_category_pilot.py`. Total category pilot calls ~100, total cost ~$0.18 (limit $0.20/batch).

**Results table** (difference = trait=0.8 avg. extraction − trait=0.2 avg. extraction; descriptive, no p-values):

| Trait | Difference (0.8−0.2) | n/cell | Class |
|---|---|---|---|
| `risk_tolerance_assigned` | expected direction (+) | 45 run (full exp.) + micro | **working** |
| `aggression_assigned` | +3.68 | 5 | **working** |
| `fairness_assigned` | −2.88 | 10 | **working** |
| `greed_assigned` | +4.00 | 5 | **working** |
| `cooperation_assigned` | inverse (+) | 5–45 | unreliable (inverse fidelity) |
| `impatience_assigned` | 0.00 | 5 | trait-blind |
| `tolerance_assigned` | 0.00 | 5 | trait-blind |
| `creativity_assigned` | 0.00 | 5 | trait-blind |
| `hoarding_assigned` | 0.00 | 5 | trait-blind |
| `trust_assigned` | 0.00 | 5 | trait-blind |
| `caution_assigned` | 0.00 | 5 | trait-blind |

**Summary classification:**
- **Working (n=4):** `risk_tolerance`, `aggression`, `fairness`, `greed` — numeric level separates extraction in expected direction.
- **Trait-blind / unreliable (n=7):** `cooperation` (inverse fidelity), `impatience`, `tolerance`, `creativity`, `hoarding`, `trust`, `caution` — groups not distinguished or inverse/inconsistent.

**LOCKED FINDING — `caution_assigned`:** Defined as the inverse frame of `risk_tolerance_assigned` (0=not cautious at all, 1=very cautious); nearly a pure synonym at word level. Nevertheless micro-A/B yielded **trait-blind** (fixed 6.0 at both levels, 50% heuristic). This **refutes** the Section 18.4 hypothesis that "concrete/action-oriented words work reliably": word choice determines fidelity independently of category (abstract/concrete, action/value); `risk` works while `caution` does not.

**FINAL CONCLUSION:** Trait fidelity does **not** follow a predictable category rule (abstract/concrete, action/value, inward/social). Each trait's fidelity is idiosyncratic and must be validated empirically (via micro-A/B) — no general rule can be derived. The early generalization in Section 18.4 ("action-oriented traits are reliable") has been **revised** by this screen.

**Methodological note — `hoarding_assigned`:** Failure likely stems from a separate cause: in a single-round design, "stockpiling" behavior is **not observable** via extraction amount (requires multi-round accumulation). This is a **confound**; it should not be interpreted as trait fidelity failure.

**Practical implication (for framework):** Before defining a trait and proceeding to a full experiment, each candidate trait should be **tested via micro-A/B** in the current prompt and scenario context; do not rely on category intuition or word "concreteness".

**PHASE STATUS: CLOSED.** 11-trait pilot screen completed; **no new trait attempts** within this phase scope. If new traits must be added later, they should be planned as a separate phase/experiment record (different `experiment_id`).

### 18.5. Structural Code Audit (Result: Sound, One Small Fix)

18 `.py` files scanned (read-only). Result: **no orphan files**, production chain clear (`run_full_experiment.py → decision_agent.py → referee_node.py → database.py`), test/debug/analysis files categorizable. One real risk: **duplicate definition** between `core/graph.py` and inline graph in `run_full_experiment.py` (silent inconsistency risk) — merged into single source. Referee carrying 4 responsibilities (physics+shock+metrics+persistence) noted but left unchanged for now because it complies with architectural rule (no LLM).

### 18.6. Cost Ledger (Phase 2 Total)

| Stage | Approximate cost |
|---|---|
| Single-agent test + initial 5-agent tests | ~$0.05 |
| Ratio calibration (validation + revalidation + fine-tune) | ~$0.89 |
| `full_experiment_v1` (45 runs) | $6.27 |
| Micro A/B validation (2 rounds, 20 calls) | ~$0.04 |
| **Total (with Haiku 4.5)** | **~$7.25** |

### 18.7. What's Next

1. ~~Control group agent~~ → **COMPLETED** (`control_group_v1`, `agents/control_agent.py`).
2. ~~k-means clustering~~ → **COMPLETED** (`analysis/clustering.py`).
3. ~~Synthesis report~~ → **COMPLETED** (`data/synthesis_report.md`).
4. **Phase 0 data export** → **COMPLETED** (`analysis/export_experiment_summary.py`, `data/amads_round0_summary.csv`).
5. **Test strategy / `tests/`** — unit tests for Referee metrics (cost-free).
6. ~~**Sonnet 4.6 replication (full 45-run)**~~ → **NOT IMPLEMENTED** — cross-model pilot (`sonnet_crossmodel_v1`, n=40) provided sufficient evidence; full run deferred; see Section 18.8.
7. Docker / Streamlit analytics — operational, low priority.

### 18.8. Cross-Model Replication (Sonnet 4.6, n=40)

**Purpose:** To test whether Haiku findings in Section 18.4 (cooperation inverse fidelity, risk expected direction) are model-specific or general LLM behavior.

**Design:**

| Parameter | Value |
|---|---|
| `experiment_id` | `sonnet_crossmodel_v1` |
| Model | `claude-sonnet-4-6` |
| Prompt | Turkish, **identical** to Haiku micro-A/B (Section 18.2 text) |
| Trait grid | cooperation × risk ∈ {0.2, 0.8}² (4 cells) |
| Replicates/cell | n=10 |
| Total calls | 40 |
| Round | 0 (single round, no confound) |
| Script | `analysis/prompt_ab_sonnet_crossmodel.py` |
| Data | `data/sonnet_crossmodel_v1.csv` (not written to `results.db`) |

**Results (via round-0 extraction_amount):**

| Trait | Sonnet 4.6 (n=40) | Haiku 4.5 reference (Section 18.4, n=45 round-0) |
|---|---|---|
| cooperation → extraction | r ≈ **−0.84** (expected direction) | r ≈ **+0.46** (inverse fidelity) |
| risk → extraction | r ≈ **+0.15** (weak) | r ≈ **+0.68** (strong, expected direction) |

**Marginal averages (cooperation, risk marginalized):**

| cooperation_assigned | Avg. extraction |
|---|---|
| 0.2 (low cooperation) | ~10.0 |
| 0.8 (high cooperation) | ~5.35 |

**Interpretation:** Same prompt, different model → both **direction** and **magnitude** change. Cooperation is in expected direction on Sonnet (high coop → less extraction), inverse on Haiku; risk is nearly ineffective on Sonnet, strong on Haiku. This advances the Section 18.4 finding revised as "trait fidelity does not work by category rule" one step further: **trait fidelity is model-specific** — the same trait+prompt combination can produce different (even opposite) behavior across models.

**Open question:** Is the visible "trade-off" between cooperation and risk (Haiku: strong risk / weak-inverse cooperation; Sonnet: strong-expected cooperation / weak risk) coincidental or systemic? **Cannot be answered with N=2 models** — future work (≥3 models, same grid, prereg/SAP).

**Scope note:** Main experiment (`full_experiment_v1`, Haiku 45-run) **unchanged**. Sonnet run kept as a separate, independent replication record.

---

*This file is the written form of all spoken architectural decisions in the project. When a new decision is made, this file must be updated — otherwise the "single source of truth" principle is violated.*
