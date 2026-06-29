"""
Extend multilang A/B for de, pt, ar: +10 calls each, merge with prior batch (n=10/cell).

Usage (repo root):
    python analysis/prompt_ab_multilang_extend.py
"""

from __future__ import annotations

import asyncio
import csv
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.decision_agent import (
    _INPUT_COST_PER_M,
    _OUTPUT_COST_PER_M,
    reset_token_usage,
    token_usage,
)
from analysis.prompt_ab_multilang import (
    COOPERATION_VALUES,
    DIFF_THRESHOLD,
    RUNS_PER_VALUE,
    RISK_TOLERANCE,
    _classify_comment,
    _decide,
    _make_state,
    _print_summary_table,
)
from core.config import settings
from core.state import AgentInputView

EXTEND_LANGUAGES = ("de", "pt", "ar")
COST_CAP_USD = 0.20
OUTPUT_CSV = _ROOT / "data" / "multilang_results_extend.csv"

# Prior batch (prompt_ab_multilang.py run 1, 5+5 per language)
PRIOR_EXTRACTIONS: dict[str, dict[float, list[float]]] = {
    "de": {
        0.2: [8.4, 8.4, 8.4, 7.2, 8.4],
        0.8: [7.2, 8.4, 8.4, 7.2, 7.2],
    },
    "pt": {
        0.2: [9.6, 9.6, 9.6, 9.6, 9.6],
        0.8: [7.2, 9.6, 9.6, 9.6, 9.6],
    },
    "ar": {
        0.2: [10.0, 10.0, 10.5, 10.5, 10.5],
        0.8: [9.6, 9.6, 9.6, 9.6, 9.6],
    },
}


async def _run_language_extend(lang: str) -> dict[float, list[float]]:
    extractions: dict[float, list[float]] = {coop: [] for coop in COOPERATION_VALUES}

    for cooperation in COOPERATION_VALUES:
        for run_index in range(1, RUNS_PER_VALUE + 1):
            if token_usage.estimated_cost_usd() >= COST_CAP_USD:
                print(f"\n⚠ Cost cap (${COST_CAP_USD:.2f}) reached — stopping.")
                return extractions

            state = _make_state(lang, cooperation, run_index + RUNS_PER_VALUE)
            trait = state.agent_traits["agent_1"]
            agent_input = AgentInputView(
                own_trait=trait,
                environment=state.environment,
                round_number=state.round_number,
            )
            decision = await _decide(lang, agent_input)
            extractions[cooperation].append(decision.extraction_amount)
            print(
                f"  [{lang}] coop={cooperation:.1f} run={run_index} (batch 2): "
                f"ext={decision.extraction_amount:.4f}"
            )

    return extractions


def _summarize_combined(lang: str, combined: dict[float, list[float]]) -> dict:
    mean_low = statistics.mean(combined[0.2])
    mean_high = statistics.mean(combined[0.8])
    diff = mean_high - mean_low
    return {
        "lang": lang,
        "n_per_cell": len(combined[0.2]),
        "coop_0.2_mean": mean_low,
        "coop_0.8_mean": mean_high,
        "diff": diff,
        "comment": _classify_comment(mean_low, mean_high, diff),
        "coop_0.2_values": combined[0.2],
        "coop_0.8_values": combined[0.8],
    }


def _print_detail(rows: list[dict]) -> None:
    for row in rows:
        print(f"\n  {row['lang']} (n={row['n_per_cell']}/cell):")
        print(f"    coop=0.2: {row['coop_0.2_values']}")
        print(f"    coop=0.8: {row['coop_0.8_values']}")


def _write_csv(rows: list[dict]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lang",
                "n_per_cell",
                "coop_0.2_mean",
                "coop_0.8_mean",
                "diff",
                "comment",
                "coop_0.2_values",
                "coop_0.8_values",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "lang": row["lang"],
                    "n_per_cell": row["n_per_cell"],
                    "coop_0.2_mean": f"{row['coop_0.2_mean']:.4f}",
                    "coop_0.8_mean": f"{row['coop_0.8_mean']:.4f}",
                    "diff": f"{row['diff']:+.4f}",
                    "comment": row["comment"],
                    "coop_0.2_values": str(row["coop_0.2_values"]),
                    "coop_0.8_values": str(row["coop_0.8_values"]),
                }
            )


async def main() -> None:
    reset_token_usage()

    print("Multilang extend — de, pt, ar (+10 calls each, merge n=10/cell)")
    print(f"  model       : {settings.ANTHROPIC_MODEL}")
    print(f"  temperature : {settings.TEMPERATURE}")
    print(f"  risk        : {RISK_TOLERANCE} (fixed)")
    print(f"  languages   : {list(EXTEND_LANGUAGES)}")
    print(f"  cost cap    : ${COST_CAP_USD:.2f}\n")

    new_extractions: dict[str, dict[float, list[float]]] = {}

    for lang in EXTEND_LANGUAGES:
        if token_usage.estimated_cost_usd() >= COST_CAP_USD:
            break
        print(f"--- {lang} (batch 2) ---")
        new_extractions[lang] = await _run_language_extend(lang)

    combined_rows: list[dict] = []
    for lang in EXTEND_LANGUAGES:
        if lang not in new_extractions:
            continue
        combined = {
            coop: PRIOR_EXTRACTIONS[lang][coop] + new_extractions[lang][coop]
            for coop in COOPERATION_VALUES
        }
        combined_rows.append(_summarize_combined(lang, combined))

    print("\n--- Prior batch (n=5) recap ---")
    prior_summary = []
    for lang in EXTEND_LANGUAGES:
        p = PRIOR_EXTRACTIONS[lang]
        mean_low = statistics.mean(p[0.2])
        mean_high = statistics.mean(p[0.8])
        diff = mean_high - mean_low
        prior_summary.append(
            {
                "lang": lang,
                "coop_0.2_mean": mean_low,
                "coop_0.8_mean": mean_high,
                "diff": diff,
                "comment": _classify_comment(mean_low, mean_high, diff),
            }
        )
    _print_summary_table(prior_summary)

    print("\n--- Combined n=10/cell ---")
    table_rows = [
        {
            "lang": r["lang"],
            "coop_0.2_mean": r["coop_0.2_mean"],
            "coop_0.8_mean": r["coop_0.8_mean"],
            "diff": r["diff"],
            "comment": r["comment"],
        }
        for r in combined_rows
    ]
    _print_summary_table(table_rows)
    _print_detail(combined_rows)
    _write_csv(combined_rows)

    print("\n--- Token usage and estimated cost ---")
    print(f"  estimated cost: ${token_usage.estimated_cost_usd():.6f} USD")
    print(
        f"  (pricing: ${_INPUT_COST_PER_M:.2f}/M input, "
        f"${_OUTPUT_COST_PER_M:.2f}/M output — Claude Haiku 4.5)"
    )
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
