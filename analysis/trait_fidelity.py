"""
Trait fidelity analysis: assigned traits vs observed behavior per run.

Denetim raporu (v2) düzeltmeleri:
  - Birincil metrik: extraction_fraction (extraction / declared_max)
  - Round penceresi (--max-round) ile collapse confound azaltılır
  - risk → gini kaldırıldı (homojen agent tasarımında geçersiz proxy)
  - risk → extraction_fraction + collapse_round
  - 9 koşul hücre ortalaması + marjinal özet
  - Eski run-avg cooperation_score / gini raporu (--legacy) ile opsiyonel

Usage (repo root):
    venv/bin/python analysis/trait_fidelity.py
    venv/bin/python analysis/trait_fidelity.py --max-round 2
    venv/bin/python analysis/trait_fidelity.py --legacy

Mock baseline (LLM yok):
    venv/bin/python analysis/trait_fidelity_mock_baseline.py

Requires: scipy, matplotlib
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.database import RESULTS_DB_PATH

try:
    from scipy.stats import linregress, pearsonr
except ImportError as exc:
    linregress = None  # type: ignore[misc, assignment]
    pearsonr = None  # type: ignore[misc, assignment]
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

DEFAULT_EXPERIMENT_ID = "full_experiment_v1"
COOP_PLOT_PATH = "data/trait_fidelity_coop.png"
RISK_PLOT_PATH = "data/trait_fidelity_risk.png"


@dataclass(frozen=True)
class RunFidelityRow:
    run_id: str
    coop_assigned: float
    risk_assigned: float
    coop_level: str
    risk_level: str
    extraction_fraction: float
    coop_score_avg: float
    gini_avg: float
    rounds_played: int
    collapse_round: int | None


@dataclass(frozen=True)
class CellMean:
    coop_value: float
    risk_value: float
    n_reps: int
    mean_fraction: float
    mean_coop_score: float
    mean_rounds: float


def _interpret_r(r: float) -> str:
    abs_r = abs(r)
    direction = "pozitif" if r > 0 else "negatif" if r < 0 else "sıfır"
    if abs_r >= 0.7:
        strength = "güçlü"
    elif abs_r >= 0.4:
        strength = "orta"
    elif abs_r >= 0.2:
        strength = "zayıf"
    else:
        strength = "ihmal edilebilir"
    return f"r={r:.3f} → {strength} {direction} ilişki"


def _interpret_p(p: float, alpha: float = 0.05) -> str:
    if p < alpha:
        return f"p={p:.4f} < {alpha} → istatistiksel olarak anlamlı"
    return f"p={p:.4f} ≥ {alpha} → istatistiksel olarak anlamlı değil"


def _require_scipy() -> None:
    if pearsonr is None or linregress is None:
        raise RuntimeError(
            f"scipy gerekli. Kurulum: pip install scipy ({_SCIPY_IMPORT_ERROR})"
        )


def fetch_run_fidelity(
    conn: sqlite3.Connection,
    experiment_id: str,
    *,
    max_round: int | None,
) -> list[RunFidelityRow]:
    round_filter_dec = ""
    round_filter_met = ""
    params: list = [experiment_id]
    if max_round is not None:
        round_filter_dec = "AND d.round_number <= ?"
        round_filter_met = "AND m.round_number <= ?"
        params.append(max_round)
    params.append(experiment_id)
    if max_round is not None:
        params.append(max_round)
    params.append(experiment_id)

    sql = f"""
        SELECT
            c.run_id,
            c.coop_value,
            c.risk_value,
            c.coop_level,
            c.risk_level,
            frac.avg_fraction,
            met.avg_coop_score,
            met.avg_gini,
            met.rounds_played,
            met.last_round
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
            {round_filter_dec}
            GROUP BY d.run_id
        ) frac ON c.run_id = frac.run_id
        JOIN (
            SELECT
                m.run_id,
                AVG(m.cooperation_score_avg) AS avg_coop_score,
                AVG(m.gini_coefficient) AS avg_gini,
                COUNT(m.round_number) AS rounds_played,
                MAX(m.round_number) AS last_round
            FROM metrics_snapshots m
            WHERE m.experiment_id = ?
            {round_filter_met}
            GROUP BY m.run_id
        ) met ON c.run_id = met.run_id
        WHERE c.experiment_id = ?
        ORDER BY c.coop_level, c.risk_level, c.replication
    """
    rows = conn.execute(sql, params).fetchall()
    result: list[RunFidelityRow] = []
    for (
        run_id,
        coop_val,
        risk_val,
        coop_level,
        risk_level,
        avg_frac,
        avg_coop,
        avg_gini,
        rounds_played,
        last_round,
    ) in rows:
        collapse = int(last_round) if rounds_played < 15 else None
        result.append(
            RunFidelityRow(
                run_id=run_id,
                coop_assigned=coop_val,
                risk_assigned=risk_val,
                coop_level=coop_level,
                risk_level=risk_level,
                extraction_fraction=avg_frac,
                coop_score_avg=avg_coop,
                gini_avg=avg_gini,
                rounds_played=rounds_played,
                collapse_round=collapse,
            )
        )
    return result


def _cell_means(rows: list[RunFidelityRow]) -> list[CellMean]:
    buckets: dict[tuple[float, float], list[RunFidelityRow]] = {}
    for row in rows:
        buckets.setdefault((row.coop_assigned, row.risk_assigned), []).append(row)

    cells: list[CellMean] = []
    for (coop, risk), members in sorted(buckets.items()):
        n = len(members)
        cells.append(
            CellMean(
                coop_value=coop,
                risk_value=risk,
                n_reps=n,
                mean_fraction=sum(m.extraction_fraction for m in members) / n,
                mean_coop_score=sum(m.coop_score_avg for m in members) / n,
                mean_rounds=sum(m.rounds_played for m in members) / n,
            )
        )
    return cells


def _marginal_coop_effect(rows: list[RunFidelityRow]) -> dict[float, float]:
    """Sabit risk seviyelerinde coop → ort. extraction_fraction."""
    by_risk: dict[float, list[RunFidelityRow]] = {}
    for row in rows:
        by_risk.setdefault(row.risk_assigned, []).append(row)

    out: dict[float, float] = {}
    for risk in sorted(by_risk):
        members = by_risk[risk]
        xs = [m.coop_assigned for m in members]
        ys = [m.extraction_fraction for m in members]
        if len(set(xs)) >= 2:
            out[risk] = pearsonr(xs, ys).statistic  # type: ignore[union-attr]
    return out


def _save_scatter(
    *,
    x: list[float],
    y: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    if plt is None:
        print(f"  ⚠ matplotlib yok — grafik atlandı: {out_path}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.75, edgecolors="k", linewidths=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Grafik kaydedildi: {out_path}")


def _print_correlation_block(
    label: str,
    x: list[float],
    y: list[float],
    *,
    n_note: str,
) -> tuple[float, float]:
    r, p = pearsonr(x, y)  # type: ignore[misc]
    print(f"--- {label} ({n_note}) ---")
    print(f"  Pearson r     : {r:.4f}")
    print(f"  p-değeri      : {p:.4f}")
    print(f"  Yorum         : {_interpret_r(r)}")
    print(f"                : {_interpret_p(p)}")
    print()
    return r, p


def _print_legacy_block(rows: list[RunFidelityRow]) -> None:
    print("=" * 72)
    print("LEGACY RAPOR (run-avg, confound uyarılı — yalnızca karşılaştırma için)")
    print("=" * 72)
    coop_x = [r.coop_assigned for r in rows]
    coop_y = [r.coop_score_avg for r in rows]
    risk_x = [r.risk_assigned for r in rows]
    gini_y = [r.gini_avg for r in rows]
    rounds = [r.rounds_played for r in rows]

    _print_correlation_block(
        "coop → ort. cooperation_score_avg (tüm round'lar)",
        coop_x,
        coop_y,
        n_note=f"n={len(rows)} run",
    )
    _print_correlation_block(
        "risk → ort. gini (tüm round'lar) [GEÇERSİZ PROXY — homojen agent]",
        risk_x,
        gini_y,
        n_note=f"n={len(rows)} run, uyarı: artefakt riski yüksek",
    )
    r_surv, p_surv = pearsonr(rounds, coop_y)  # type: ignore[misc]
    print(f"  confound kontrolü: rounds ↔ coop_score  r={r_surv:.3f}, p={p_surv:.2e}")
    print()


def run_analysis(
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    db_path: str = RESULTS_DB_PATH,
    *,
    max_round: int | None = 0,
    include_legacy: bool = False,
    no_plots: bool = False,
) -> None:
    _require_scipy()

    path = Path(db_path)
    if not path.exists():
        print(f"Hata: veritabanı bulunamadı: {path}", file=sys.stderr)
        sys.exit(1)

    with sqlite3.connect(path) as conn:
        rows = fetch_run_fidelity(conn, experiment_id, max_round=max_round)

    if not rows:
        print(
            f"Hata: '{experiment_id}' için experiment_conditions + karar/metrik verisi yok.",
            file=sys.stderr,
        )
        sys.exit(1)

    window_label = (
        f"round 0–{max_round}" if max_round is not None else "tüm round'lar"
    )

    print("=" * 72)
    print("TRAIT FIDELITY ANALYSIS (v2)")
    print("=" * 72)
    print(f"  experiment_id : {experiment_id}")
    print(f"  database      : {path}")
    print(f"  run sayısı    : {len(rows)}")
    print(f"  round penceresi: {window_label}")
    print(f"  birincil metrik: extraction_fraction (extraction / declared_max)")
    print()

    cells = _cell_means(rows)
    print("9 koşul hücre ortalaması (extraction_fraction):")
    print(f"  {'coop':>5} {'risk':>5} | {'fraction':>8} {'coop_scr':>8} {'rounds':>6} n")
    print("  " + "-" * 44)
    for cell in cells:
        print(
            f"  {cell.coop_value:5.1f} {cell.risk_value:5.1f} | "
            f"{cell.mean_fraction:8.3f} {cell.mean_coop_score:8.3f} "
            f"{cell.mean_rounds:6.1f} {cell.n_reps}"
        )
    print()

    coop_x = [r.coop_assigned for r in rows]
    frac_y = [r.extraction_fraction for r in rows]
    risk_x = [r.risk_assigned for r in rows]
    rounds = [r.rounds_played for r in rows]

    _print_correlation_block(
        "coop → extraction_fraction (birincil fidelity metriği)",
        coop_x,
        frac_y,
        n_note=f"n={len(rows)} run, {window_label}",
    )
    _print_correlation_block(
        "risk → extraction_fraction (risk davranış proxy'si)",
        risk_x,
        frac_y,
        n_note=f"n={len(rows)} run, {window_label}",
    )

    collapse_rounds = [
        r.collapse_round for r in rows if r.collapse_round is not None
    ]
    if collapse_rounds and max_round is None:
        risk_with_collapse = [
            r.risk_assigned for r in rows if r.collapse_round is not None
        ]
        r_col, p_col = pearsonr(risk_with_collapse, collapse_rounds)  # type: ignore[misc]
        print("--- risk → collapse_round (erken çöküş) ---")
        print(f"  Pearson r     : {r_col:.4f}  (düşük round = erken collapse)")
        print(f"  p-değeri      : {p_col:.4f}")
        print(f"  Yorum         : {_interpret_r(r_col)}")
        print()

    cell_coop = [c.coop_value for c in cells]
    cell_frac = [c.mean_fraction for c in cells]
    _print_correlation_block(
        "coop → extraction_fraction (9 hücre ortalaması)",
        cell_coop,
        cell_frac,
        n_note="n=9 koşul, tekrarlar birleştirilmiş",
    )

    marg = _marginal_coop_effect(rows)
    if marg:
        print("Sabit risk seviyesinde coop → fraction (marjinal Pearson):")
        for risk, r_val in sorted(marg.items()):
            print(f"  risk={risk:.1f}: r={r_val:.3f}")
        print()

    slope, intercept, r_val, p_val, _ = linregress(coop_x, frac_y)  # type: ignore[misc]
    print("Basit doğrusal regresyon: fraction ~ coop (risk karışık, 45 run)")
    print(f"  slope={slope:.4f}, intercept={intercept:.4f}, r={r_val:.4f}, p={p_val:.4f}")
    print("  (Not: tekrarlar bağımsız sayılmamalı; hücre ortalaması tercih edilir.)")
    print()

    r_surv, p_surv = pearsonr(rounds, frac_y)  # type: ignore[misc]
    if len(set(rounds)) > 1:
        print(
            f"Confound kontrolü: rounds ↔ extraction_fraction  "
            f"r={r_surv:.3f}, p={p_surv:.2e}"
        )
    else:
        print("Confound kontrolü: tek round — collapse/süre confound'u yok.")
    print()

    print("--- Fidelity yorumu (tasarım niyeti vs gözlem) ---")
    print("  Tasarım niyeti (mock): coop ↑ → extraction_fraction ↓ (negatif r beklenir)")
    print(f"  Gözlem (round 0):      coop → fraction r={pearsonr(coop_x, frac_y).statistic:.3f}")  # type: ignore[misc]
    if pearsonr(coop_x, frac_y).statistic > 0:  # type: ignore[misc]
        print("  → TERS fidelity: yüksek coop atanan run'lar DAHA FAZLA çekiyor.")
    else:
        print("  → Uyumlu fidelity: yüksek coop → daha az çekim.")
    print("  Mock baseline karşılaştırması: venv/bin/python analysis/trait_fidelity_mock_baseline.py")
    print()

    print("Run bazlı özet (ilk 5):")
    print(
        f"  {'run_id':<28} {'coop':>5} {'risk':>5} "
        f"{'frac':>6} {'rounds':>6}"
    )
    for row in rows[:5]:
        print(
            f"  {row.run_id:<28} {row.coop_assigned:5.1f} {row.risk_assigned:5.1f} "
            f"{row.extraction_fraction:6.3f} {row.rounds_played:6d}"
        )
    if len(rows) > 5:
        print(f"  ... ve {len(rows) - 5} run daha")
    print()

    if include_legacy:
        with sqlite3.connect(path) as conn:
            full_rows = fetch_run_fidelity(conn, experiment_id, max_round=None)
        _print_legacy_block(full_rows)

    if not no_plots:
        print("--- Scatter plotlar ---")
        suffix = f"r0-{max_round}" if max_round is not None else "full"
        _save_scatter(
            x=coop_x,
            y=frac_y,
            xlabel="cooperation_assigned",
            ylabel="extraction_fraction (düşük = daha işbirlikçi)",
            title=f"Coop Fidelity — fraction ({experiment_id}, {window_label})",
            out_path=Path(f"data/trait_fidelity_coop_{suffix}.png"),
        )
        _save_scatter(
            x=risk_x,
            y=frac_y,
            xlabel="risk_tolerance_assigned",
            ylabel="extraction_fraction",
            title=f"Risk Effect — fraction ({experiment_id}, {window_label})",
            out_path=Path(f"data/trait_fidelity_risk_{suffix}.png"),
        )
        # Geriye dönük uyumluluk
        _save_scatter(
            x=coop_x,
            y=frac_y,
            xlabel="cooperation_assigned",
            ylabel="extraction_fraction",
            title=f"Trait Fidelity — Cooperation ({experiment_id})",
            out_path=Path(COOP_PLOT_PATH),
        )
        _save_scatter(
            x=risk_x,
            y=frac_y,
            xlabel="risk_tolerance_assigned",
            ylabel="extraction_fraction",
            title=f"Trait Fidelity — Risk ({experiment_id})",
            out_path=Path(RISK_PLOT_PATH),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assigned trait vs observed behavior (v2 — denetim düzeltmeleri).",
    )
    parser.add_argument(
        "--experiment-id",
        default=DEFAULT_EXPERIMENT_ID,
        help=f"Experiment ID (default: {DEFAULT_EXPERIMENT_ID})",
    )
    parser.add_argument(
        "--db",
        default=RESULTS_DB_PATH,
        help=f"SQLite path (default: {RESULTS_DB_PATH})",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=0,
        metavar="N",
        help="Round penceresi üst sınırı (varsayılan: 0 = yalnızca round 0)",
    )
    parser.add_argument(
        "--all-rounds",
        action="store_true",
        help="Tüm round'ları dahil et (collapse confound riski)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Eski run-avg cooperation_score / gini raporunu da yazdır",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Scatter plot oluşturma",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    max_round = None if args.all_rounds else args.max_round
    run_analysis(
        args.experiment_id,
        args.db,
        max_round=max_round,
        include_legacy=args.legacy,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
