"""
Generate publication figures for docs/paper_draft.md.

Reads Haiku trait-fidelity Pearson r from data/results.db (read-only),
Sonnet r from data/sonnet_crossmodel_v1.csv, and multilang diffs from
data/multilang_results.csv. Writes PNGs under figures/.

Usage (repo root):
    venv/bin/python analysis/generate_paper_figures.py
"""

from __future__ import annotations

import csv
import sqlite3
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.database import RESULTS_DB_PATH

FIGURES_DIR = _ROOT / "figures"
MULTILANG_CSV = _ROOT / "data" / "multilang_results.csv"
SONNET_CSV = _ROOT / "data" / "sonnet_crossmodel_v1.csv"
HAIKU_EXPERIMENT = "full_experiment_v1"

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
except ImportError as exc:
    raise SystemExit(f"matplotlib required: pip install matplotlib ({exc})") from exc


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return num / (var_x * var_y) ** 0.5


def _fetch_haiku_fidelity() -> tuple[float, float]:
    sql = """
        SELECT
            c.coop_value,
            c.risk_value,
            frac.avg_fraction
        FROM experiment_conditions c
        JOIN (
            SELECT
                d.run_id,
                AVG(
                    CASE WHEN d.declared_max > 0
                         THEN d.extraction_amount / d.declared_max
                         ELSE 0 END
                ) AS avg_fraction
            FROM agent_decisions d
            WHERE d.experiment_id = ?
              AND d.round_number = 0
            GROUP BY d.run_id
        ) frac ON c.run_id = frac.run_id
        WHERE c.experiment_id = ?
    """
    db_path = _ROOT / RESULTS_DB_PATH
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(sql, (HAIKU_EXPERIMENT, HAIKU_EXPERIMENT)).fetchall()

    coops = [float(r[0]) for r in rows]
    risks = [float(r[1]) for r in rows]
    fractions = [float(r[2]) for r in rows]
    return _pearson_r(coops, fractions), _pearson_r(risks, fractions)


def _fetch_sonnet_fidelity() -> tuple[float, float]:
    coops: list[float] = []
    risks: list[float] = []
    fractions: list[float] = []
    with SONNET_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            coops.append(float(row["coop_value"]))
            risks.append(float(row["risk_value"]))
            fractions.append(float(row["extraction_fraction"]))
    return _pearson_r(coops, fractions), _pearson_r(risks, fractions)


def _draw_architecture_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08",
            linewidth=1.5,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    box(0.3, 2.0, 1.6, 1.0, "Round\nStart", "#E8EEF7")
    for i, label in enumerate(["Agent 1", "Agent 2", "Agent 3", "Agent 4", "Agent 5"], start=1):
        y = 4.2 - i * 0.72
        box(2.8, y, 1.8, 0.55, f"{label}\n(parallel)", "#DFF3E3")
        ax.add_patch(
            FancyArrowPatch(
                (1.9, 2.5),
                (2.8, y + 0.28),
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.2,
                color="#555555",
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (4.6, y + 0.28),
                (6.0, 2.5),
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.2,
                color="#555555",
            )
        )

    box(6.0, 1.7, 2.4, 1.6, "Referee\n(collect, compute,\nadvance)\n[LLM-free]", "#FCE8E6")
    ax.text(5.0, 0.35, "Agent fan-out → deterministic referee (no LLM evaluation)", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_haiku_sonnet_comparison(path: Path) -> None:
    haiku_coop, haiku_risk = _fetch_haiku_fidelity()
    sonnet_coop, sonnet_risk = _fetch_sonnet_fidelity()

    traits = ["Cooperation", "Risk tolerance"]
    haiku_vals = [haiku_coop, haiku_risk]
    sonnet_vals = [sonnet_coop, sonnet_risk]

    x = [0, 1]
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], haiku_vals, width, label="Haiku 4.5", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], sonnet_vals, width, label="Sonnet 4.6", color="#F58518")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(traits)
    ax.set_ylabel("Pearson r (trait → extraction fraction)")
    ax.set_title("Cross-model trait fidelity (round 0)")
    ax.set_ylim(-1.0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for offset, vals in [(-width / 2, haiku_vals), (width / 2, sonnet_vals)]:
        for i, val in enumerate(vals):
            ax.text(i + offset, val + (0.04 if val >= 0 else -0.08), f"{val:+.2f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_multilang_results(path: Path) -> None:
    langs: list[str] = []
    diffs: list[float] = []
    with MULTILANG_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            langs.append(row["lang"])
            diff_str = row["diff"].replace("+", "")
            diffs.append(float(diff_str))

    order = sorted(range(len(langs)), key=lambda i: diffs[i], reverse=True)
    langs = [langs[i] for i in order]
    diffs = [diffs[i] for i in order]

    colors = ["#E45756" if d > 0 else "#54A24B" if d < 0 else "#BAB0AC" for d in diffs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(langs, diffs, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Cooperation difference (coop=0.8 − coop=0.2 mean extraction)")
    ax.set_title("Eleven-language pilot: cooperation trait effect")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    arch_path = FIGURES_DIR / "architecture_diagram.png"
    compare_path = FIGURES_DIR / "haiku_sonnet_comparison.png"
    multilang_path = FIGURES_DIR / "multilang_results.png"

    _draw_architecture_diagram(arch_path)
    _draw_haiku_sonnet_comparison(compare_path)
    _draw_multilang_results(multilang_path)

    haiku_coop, haiku_risk = _fetch_haiku_fidelity()
    sonnet_coop, sonnet_risk = _fetch_sonnet_fidelity()
    print("Generated figures:")
    print(f"  {arch_path}")
    print(f"  {compare_path}")
    print(f"  {multilang_path}")
    print(f"Haiku r: coop={haiku_coop:+.4f}, risk={haiku_risk:+.4f}")
    print(f"Sonnet r: coop={sonnet_coop:+.4f}, risk={sonnet_risk:+.4f}")


if __name__ == "__main__":
    main()
