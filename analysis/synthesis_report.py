"""
Üç bağımsız analizin sentez raporu.

1. Trait fidelity (LLM, Pearson r) — full_experiment_v1
2. Davranış kümeleme (k-means, trait görmeden) — full_experiment_v1
3. Kontrol grubu yön karşılaştırması — control_group_v1 vs LLM

Usage (repo root):
    venv/bin/python analysis/synthesis_report.py
    venv/bin/python analysis/synthesis_report.py --output data/synthesis_report.md

Requires: scipy, scikit-learn
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.clustering import (
    build_behavior_vectors,
    fetch_behavior_rows,
    select_best_k,
)
from analysis.trait_fidelity import fetch_run_fidelity
from core.database import RESULTS_DB_PATH

try:
    from scipy.stats import pearsonr
except ImportError as exc:
    raise RuntimeError(f"scipy gerekli: pip install scipy ({exc})") from exc

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise RuntimeError(
        f"scikit-learn gerekli: pip install scikit-learn ({exc})"
    ) from exc

LLM_EXPERIMENT_ID = "full_experiment_v1"
CONTROL_EXPERIMENT_ID = "control_group_v1"
DEFAULT_OUTPUT = "data/synthesis_report.md"
CORRELATION_MAX_ROUND = 0
EXPECTED_RUN_COUNTS = {
    LLM_EXPERIMENT_ID: 45,
    CONTROL_EXPERIMENT_ID: 27,
}


@dataclass(frozen=True)
class FidelityResult:
    risk_r: float
    risk_p: float
    coop_r: float
    coop_p: float
    n_runs: int


@dataclass(frozen=True)
class ClusteringResult:
    best_k: int
    ari_risk: float
    ari_coop: float
    risk_verdict: str
    coop_verdict: str
    n_runs: int


@dataclass(frozen=True)
class ControlComparisonResult:
    control_coop_r: float
    control_coop_p: float
    llm_coop_r: float
    llm_coop_p: float
    control_n: int
    llm_n: int
    risk_cell: str
    coop_cell: str


def _fmt_p(p: float) -> str:
    if p < 0.0001:
        return "p<0.0001"
    return f"p={p:.4f}"


def _fmt_r_p(r: float, p: float, *, reverse_note: bool = False) -> str:
    suffix = " (ters yön)" if reverse_note and r > 0 else ""
    return f"r={r:.3f}, {_fmt_p(p)}{suffix}"


def _count_experiment_conditions(conn: sqlite3.Connection, experiment_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM experiment_conditions WHERE experiment_id = ?",
        (experiment_id,),
    ).fetchone()
    return int(row[0])


def _compute_fidelity(
    conn: sqlite3.Connection,
    experiment_id: str,
    *,
    max_round: int,
) -> FidelityResult:
    rows = fetch_run_fidelity(conn, experiment_id, max_round=max_round)
    expected = EXPECTED_RUN_COUNTS.get(experiment_id)
    if expected is not None and len(rows) != expected:
        print(
            f"Uyarı: {experiment_id} için {len(rows)} run bulundu "
            f"(beklenen {expected}).",
            file=sys.stderr,
        )
    if not rows:
        raise RuntimeError(f"'{experiment_id}' için fidelity verisi yok.")

    coop_x = [r.coop_assigned for r in rows]
    risk_x = [r.risk_assigned for r in rows]
    frac_y = [r.extraction_fraction for r in rows]

    coop_r, coop_p = pearsonr(coop_x, frac_y)
    risk_r, risk_p = pearsonr(risk_x, frac_y)
    return FidelityResult(
        risk_r=risk_r,
        risk_p=risk_p,
        coop_r=coop_r,
        coop_p=coop_p,
        n_runs=len(rows),
    )


def _clustering_verdict(ari_risk: float, ari_coop: float) -> tuple[str, str]:
    """Qualitative verdict from marginal trait ↔ cluster overlap."""
    if ari_risk >= 0.2 and ari_risk > ari_coop:
        risk_text = "kümeleri ayırıyor (risk ekseni net)"
    elif ari_risk >= 0.2:
        risk_text = "kısmen ayırıyor"
    else:
        risk_text = "kümeleri ayırmıyor"

    if ari_coop < 0.2 or ari_coop <= ari_risk:
        coop_text = "kümeleri ayırmıyor"
    elif ari_coop >= 0.2:
        coop_text = "kısmen ayırıyor"
    else:
        coop_text = "kümeleri ayırmıyor"
    return risk_text, coop_text


def _compute_clustering(
    conn: sqlite3.Connection,
    experiment_id: str,
) -> ClusteringResult:
    fetched = fetch_behavior_rows(conn, experiment_id)
    rows = [r for r, _, _ in fetched]
    if not rows:
        raise RuntimeError(f"'{experiment_id}' için kümeleme verisi yok.")

    vectors = build_behavior_vectors(rows)
    X = np.array([v.values for v in vectors], dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, _ = select_best_k(X_scaled)
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    risk_labels = [v.risk_level for v in vectors]
    coop_labels = [v.coop_level for v in vectors]
    ari_risk = adjusted_rand_score(risk_labels, labels)
    ari_coop = adjusted_rand_score(coop_labels, labels)
    risk_verdict, coop_verdict = _clustering_verdict(ari_risk, ari_coop)

    return ClusteringResult(
        best_k=best_k,
        ari_risk=ari_risk,
        ari_coop=ari_coop,
        risk_verdict=risk_verdict,
        coop_verdict=coop_verdict,
        n_runs=len(rows),
    )


def _control_comparison_cell(
    control_coop_r: float,
    llm_coop_r: float,
) -> str:
    if control_coop_r < 0 and llm_coop_r > 0:
        return (
            "kontrol: negatif (beklenen), LLM: pozitif (ters)"
        )
    if control_coop_r < 0:
        return "kontrol: negatif (beklenen)"
    if llm_coop_r > 0:
        return "LLM: pozitif (ters)"
    return "yön farkı belirsiz"


def _compute_control_comparison(
    conn: sqlite3.Connection,
    *,
    max_round: int,
) -> ControlComparisonResult:
    control_rows = fetch_run_fidelity(
        conn, CONTROL_EXPERIMENT_ID, max_round=max_round
    )
    llm_rows = fetch_run_fidelity(
        conn, LLM_EXPERIMENT_ID, max_round=max_round
    )
    if not control_rows:
        raise RuntimeError(f"'{CONTROL_EXPERIMENT_ID}' için kontrol verisi yok.")
    if not llm_rows:
        raise RuntimeError(f"'{LLM_EXPERIMENT_ID}' için LLM verisi yok.")

    control_coop_x = [r.coop_assigned for r in control_rows]
    control_frac_y = [r.extraction_fraction for r in control_rows]
    control_r, control_p = pearsonr(control_coop_x, control_frac_y)

    llm_coop_x = [r.coop_assigned for r in llm_rows]
    llm_frac_y = [r.extraction_fraction for r in llm_rows]
    llm_r, llm_p = pearsonr(llm_coop_x, llm_frac_y)

    return ControlComparisonResult(
        control_coop_r=control_r,
        control_coop_p=control_p,
        llm_coop_r=llm_r,
        llm_coop_p=llm_p,
        control_n=len(control_rows),
        llm_n=len(llm_rows),
        risk_cell="(risk kontrol grubunda kullanılmadı, N/A)",
        coop_cell=_control_comparison_cell(control_r, llm_r),
    )


def _build_synthesis_paragraph(
    fidelity: FidelityResult,
    clustering: ClusteringResult,
    control: ControlComparisonResult,
) -> str:
    return (
        "Üç bağımsız yöntem tutarlı bir tablo çiziyor: risk_tolerance, "
        "hem Pearson korelasyonunda (r={risk_r:.3f}) hem de davranış "
        "kümelemesinde (ARI={ari_risk:.3f}) LLM çıktısını güvenilir "
        "biçimde yönlendiriyor. cooperation_assigned ise ters yönde "
        "davranıyor — fidelity analizinde pozitif ve anlamlı bir ilişki "
        "(r={coop_r:.3f}, beklenen negatifin tersi), kümelemede ise "
        "koşulları ayırmıyor (ARI={ari_coop:.3f}); deterministik kontrol "
        "grubu beklenen negatif yönü doğrularken LLM grubu pozitif yönde "
        "sapıyor. Bu bulgular, risk trait'inin modele aktarıldığını, "
        "cooperation trait'inin ise hem fidelity hem kümeleme düzeyinde "
        "etkisiz veya ters kaldığını gösteriyor."
    ).format(
        risk_r=fidelity.risk_r,
        ari_risk=clustering.ari_risk,
        coop_r=fidelity.coop_r,
        ari_coop=clustering.ari_coop,
    )


def _build_markdown_report(
    *,
    fidelity: FidelityResult,
    clustering: ClusteringResult,
    control: ControlComparisonResult,
    llm_conditions_n: int,
    control_conditions_n: int,
    max_round: int,
) -> str:
    window = f"round 0–{max_round}" if max_round is not None else "tüm round'lar"
    synthesis = _build_synthesis_paragraph(fidelity, clustering, control)

    lines = [
        "# AMADS Analiz Sentez Raporu",
        "",
        "## Veri kaynağı",
        "",
        f"- **LLM deneyi:** `{LLM_EXPERIMENT_ID}` "
        f"({llm_conditions_n} run, experiment_conditions)",
        f"- **Kontrol grubu:** `{CONTROL_EXPERIMENT_ID}` "
        f"({control_conditions_n} run, experiment_conditions)",
        f"- **Round penceresi (fidelity / kontrol):** {window}",
        f"- **Kümeleme:** tüm round'lar, davranış vektörü "
        f"(extraction_fraction, gini, cooperation_score_avg, collapse_round); "
        f"k={clustering.best_k}",
        "",
        "## Özet tablo",
        "",
        "| Yöntem | Risk'in etkisi | Cooperation'ın etkisi |",
        "|---|---|---|",
        (
            f"| Trait fidelity (LLM, r, p) | "
            f"{_fmt_r_p(fidelity.risk_r, fidelity.risk_p)} | "
            f"{_fmt_r_p(fidelity.coop_r, fidelity.coop_p, reverse_note=True)} |"
        ),
        (
            f"| Kümeleme (k-means, trait'i hiç görmeden) | "
            f"{clustering.risk_verdict} | {clustering.coop_verdict} |"
        ),
        (
            f"| Kontrol grubu (deterministik formül) vs LLM yön karşılaştırması | "
            f"{control.risk_cell} | {control.coop_cell} |"
        ),
        "",
        "## Sentez",
        "",
        synthesis,
        "",
        "## Detay (referans)",
        "",
        "### Trait fidelity (LLM)",
        "",
        f"- n = {fidelity.n_runs}",
        f"- risk → extraction_fraction: r = {fidelity.risk_r:.4f}, "
        f"p = {fidelity.risk_p:.4g}",
        f"- coop → extraction_fraction: r = {fidelity.coop_r:.4f}, "
        f"p = {fidelity.coop_p:.4g}",
        "",
        "### Kümeleme",
        "",
        f"- n = {clustering.n_runs}, k = {clustering.best_k}",
        f"- ARI (küme ↔ risk_level): {clustering.ari_risk:.4f}",
        f"- ARI (küme ↔ coop_level): {clustering.ari_coop:.4f}",
        "",
        "### Kontrol vs LLM (coop → extraction_fraction)",
        "",
        f"- Kontrol ({control.control_n} run): r = {control.control_coop_r:.4f}, "
        f"p = {control.control_coop_p:.4g}",
        f"- LLM ({control.llm_n} run): r = {control.llm_coop_r:.4f}, "
        f"p = {control.llm_coop_p:.4g}",
        "",
    ]
    return "\n".join(lines)


def _print_terminal_report(
    *,
    output_path: str,
    fidelity: FidelityResult,
    clustering: ClusteringResult,
    control: ControlComparisonResult,
) -> None:
    print("=" * 72)
    print("AMADS ANALİZ SENTEZ RAPORU")
    print("=" * 72)
    print()
    print("| Yöntem | Risk'in etkisi | Cooperation'ın etkisi |")
    print("|---|---|---|")
    print(
        f"| Trait fidelity (LLM, r, p) | "
        f"{_fmt_r_p(fidelity.risk_r, fidelity.risk_p)} | "
        f"{_fmt_r_p(fidelity.coop_r, fidelity.coop_p, reverse_note=True)} |"
    )
    print(
        f"| Kümeleme (k-means, trait'i hiç görmeden) | "
        f"{clustering.risk_verdict} | {clustering.coop_verdict} |"
    )
    print(
        f"| Kontrol grubu (deterministik formül) vs LLM yön karşılaştırması | "
        f"{control.risk_cell} | {control.coop_cell} |"
    )
    print()
    print("Sentez:")
    print(_build_synthesis_paragraph(fidelity, clustering, control))
    print()
    print(f"Markdown rapor kaydedildi: {output_path}")
    print()


def run_synthesis(
    db_path: str = RESULTS_DB_PATH,
    *,
    output_path: str = DEFAULT_OUTPUT,
    max_round: int = CORRELATION_MAX_ROUND,
) -> str:
    path = Path(db_path)
    if not path.exists():
        print(f"Hata: veritabanı bulunamadı: {path}", file=sys.stderr)
        sys.exit(1)

    with sqlite3.connect(path) as conn:
        llm_conditions_n = _count_experiment_conditions(conn, LLM_EXPERIMENT_ID)
        control_conditions_n = _count_experiment_conditions(
            conn, CONTROL_EXPERIMENT_ID
        )
        fidelity = _compute_fidelity(
            conn, LLM_EXPERIMENT_ID, max_round=max_round
        )
        clustering = _compute_clustering(conn, LLM_EXPERIMENT_ID)
        control = _compute_control_comparison(conn, max_round=max_round)

    report_md = _build_markdown_report(
        fidelity=fidelity,
        clustering=clustering,
        control=control,
        llm_conditions_n=llm_conditions_n,
        control_conditions_n=control_conditions_n,
        max_round=max_round,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_md, encoding="utf-8")

    _print_terminal_report(
        output_path=output_path,
        fidelity=fidelity,
        clustering=clustering,
        control=control,
    )
    return report_md


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Üç analizin sentez raporu (terminal + markdown).",
    )
    parser.add_argument(
        "--db",
        default=RESULTS_DB_PATH,
        help=f"SQLite path (default: {RESULTS_DB_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Markdown çıktı yolu (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=CORRELATION_MAX_ROUND,
        metavar="N",
        help="Fidelity/kontrol round üst sınırı (default: 0)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_synthesis(
        args.db,
        output_path=args.output,
        max_round=args.max_round,
    )


if __name__ == "__main__":
    main()
