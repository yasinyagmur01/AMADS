"""
K-means clustering of run-level behavior vectors from results.db.

For each run, builds a 4D behavior vector (all rounds):
  extraction_fraction, gini_coefficient, cooperation_score_avg, collapse_round

No LLM calls — reads SQLite only.

Usage (repo root):
    venv/bin/python analysis/clustering.py
    venv/bin/python analysis/clustering.py --experiment-id full_experiment_v1

Requires: scikit-learn, matplotlib
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

from core.config import settings
from core.database import RESULTS_DB_PATH

try:
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise RuntimeError(
        "scikit-learn and matplotlib required. "
        "Install: pip install scikit-learn matplotlib"
    ) from exc

from analysis.trait_fidelity import RunFidelityRow

DEFAULT_EXPERIMENT_ID = "full_experiment_v1"
PLOT_PATH = "data/clustering_result.png"
FEATURE_NAMES = [
    "extraction_fraction",
    "gini_coefficient",
    "cooperation_score_avg",
    "collapse_round",
]
K_RANGE = range(2, 7)
TRAIT_LEVELS: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
}


@dataclass(frozen=True)
class BehaviorVector:
    run_id: str
    coop_level: str
    risk_level: str
    coop_value: float
    risk_value: float
    values: tuple[float, float, float, float]


def _parse_run_id(run_id: str) -> tuple[str, str, float, float, int] | None:
    """Parse cond_{coop}_{risk}_rep{N} when experiment_conditions row is missing."""
    if not run_id.startswith("cond_") or "_rep" not in run_id:
        return None
    body, rep_str = run_id.removeprefix("cond_").rsplit("_rep", 1)
    if not rep_str.isdigit():
        return None
    parts = body.split("_", 1)
    if len(parts) != 2:
        return None
    coop_level, risk_level = parts
    if coop_level not in TRAIT_LEVELS or risk_level not in TRAIT_LEVELS:
        return None
    return (
        coop_level,
        risk_level,
        TRAIT_LEVELS[coop_level],
        TRAIT_LEVELS[risk_level],
        int(rep_str),
    )


def fetch_behavior_rows(
    conn: sqlite3.Connection,
    experiment_id: str,
) -> list[tuple[RunFidelityRow, str, str]]:
    """All runs with metrics/decisions; trait from conditions or run_id parse."""
    sql = """
        SELECT
            frac.run_id,
            c.coop_value,
            c.risk_value,
            c.coop_level,
            c.risk_level,
            frac.avg_fraction,
            met.avg_coop_score,
            met.avg_gini,
            met.rounds_played,
            met.last_round
        FROM (
            SELECT
                d.run_id,
                AVG(
                    CASE WHEN d.declared_max > 0
                         THEN d.extraction_amount / d.declared_max
                         ELSE 0 END
                ) AS avg_fraction
            FROM agent_decisions d
            WHERE d.experiment_id = ?
            GROUP BY d.run_id
        ) frac
        JOIN (
            SELECT
                m.run_id,
                AVG(m.cooperation_score_avg) AS avg_coop_score,
                AVG(m.gini_coefficient) AS avg_gini,
                COUNT(m.round_number) AS rounds_played,
                MAX(m.round_number) AS last_round
            FROM metrics_snapshots m
            WHERE m.experiment_id = ?
            GROUP BY m.run_id
        ) met ON frac.run_id = met.run_id
        LEFT JOIN experiment_conditions c
            ON c.run_id = frac.run_id AND c.experiment_id = ?
        ORDER BY frac.run_id
    """
    rows = conn.execute(sql, (experiment_id, experiment_id, experiment_id)).fetchall()
    result: list[tuple[RunFidelityRow, str, str]] = []
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
        if coop_level is None or risk_level is None:
            parsed = _parse_run_id(run_id)
            if parsed is None:
                continue
            coop_level, risk_level, coop_val, risk_val, _rep = parsed
        collapse = int(last_round) if rounds_played < settings.MAX_ROUNDS else None
        row = RunFidelityRow(
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
        result.append((row, coop_level, risk_level))
    return result


def _collapse_round_numeric(row: RunFidelityRow) -> float:
    """Survival endpoint: last collapse round, or MAX_ROUNDS if completed."""
    if row.collapse_round is not None:
        return float(row.collapse_round)
    return float(settings.MAX_ROUNDS)


def build_behavior_vectors(rows: list[RunFidelityRow]) -> list[BehaviorVector]:
    vectors: list[BehaviorVector] = []
    for row in rows:
        vectors.append(
            BehaviorVector(
                run_id=row.run_id,
                coop_level=row.coop_level,
                risk_level=row.risk_level,
                coop_value=row.coop_assigned,
                risk_value=row.risk_assigned,
                values=(
                    row.extraction_fraction,
                    row.gini_avg,
                    row.coop_score_avg,
                    _collapse_round_numeric(row),
                ),
            )
        )
    return vectors


def _trait_label(coop_level: str, risk_level: str) -> str:
    return f"{coop_level}/{risk_level}"


def select_best_k(X_scaled: np.ndarray) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    for k in K_RANGE:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def _cluster_centroids_table(
    scaler: StandardScaler,
    model: KMeans,
) -> None:
    centroids_scaled = model.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    print("Küme merkezleri (orijinal ölçek):")
    header = f"  {'cluster':>7} | " + " | ".join(f"{name:>22}" for name in FEATURE_NAMES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for idx, center in enumerate(centroids):
        vals = " | ".join(f"{v:22.4f}" for v in center)
        print(f"  {idx:7d} | {vals}")
    print()


def _print_silhouette_scores(scores: dict[int, float], best_k: int) -> None:
    print("Silhouette skorları (k=2..6):")
    for k in sorted(scores):
        marker = " ← seçildi" if k == best_k else ""
        print(f"  k={k}: {scores[k]:.4f}{marker}")
    print()


def _print_trait_overlap(
    vectors: list[BehaviorVector],
    labels: np.ndarray,
    best_k: int,
) -> None:
    print("=" * 72)
    print("KÜME ↔ TRAIT ÖRTÜŞMESİ")
    print("=" * 72)

    trait_keys = sorted({_trait_label(v.coop_level, v.risk_level) for v in vectors})
    trait_to_idx = {t: i for i, t in enumerate(trait_keys)}
    trait_labels = np.array(
        [_trait_label(v.coop_level, v.risk_level) for v in vectors]
    )

    print(f"\nKoşul hücreleri ({len(trait_keys)}): {', '.join(trait_keys)}")
    print()

    for cluster_id in range(best_k):
        members = [v for v, lbl in zip(vectors, labels) if lbl == cluster_id]
        n = len(members)
        print(f"--- Küme {cluster_id} (n={n}) ---")
        combo_counts: dict[str, int] = {}
        for member in members:
            key = _trait_label(member.coop_level, member.risk_level)
            combo_counts[key] = combo_counts.get(key, 0) + 1
        for combo, count in sorted(combo_counts.items(), key=lambda x: (-x[1], x[0])):
            pct = 100 * count / n
            print(f"  {combo:>12}: {count}/{n} ({pct:5.1f}%)")
        dominant = max(combo_counts, key=combo_counts.get)
        purity = combo_counts[dominant] / n
        print(f"  Baskın hücre: {dominant} (purity={purity:.2f})")
        print()

    # Cross-tab
    print("Çapraz tablo (küme × trait hücresi):")
    col_w = max(12, max(len(t) for t in trait_keys) + 2)
    header = f"  {'cluster':>7} | " + " ".join(f"{t:>{col_w}}" for t in trait_keys) + " | total"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cluster_id in range(best_k):
        counts = [0] * len(trait_keys)
        for v, lbl in zip(vectors, labels):
            if lbl == cluster_id:
                counts[trait_to_idx[_trait_label(v.coop_level, v.risk_level)]] += 1
        row_total = sum(counts)
        cells = " ".join(f"{c:>{col_w}d}" for c in counts)
        print(f"  {cluster_id:7d} | {cells} | {row_total:5d}")
    print()

    ari = adjusted_rand_score(trait_labels, labels)
    print(f"Adjusted Rand Index (küme vs trait hücresi): {ari:.4f}")
    if ari >= 0.5:
        verdict = "Kümeler trait hücreleriyle güçlü örtüşüyor — davranış tasarım koşullarına yakın."
    elif ari >= 0.2:
        verdict = "Kısmi örtüşme — hem trait hem doğal strateji bileşenleri var."
    else:
        verdict = (
            "Düşük örtüşme — kümeler trait'ten bağımsız doğal strateji grupları "
            "gibi görünüyor."
        )
    print(f"Yorum: {verdict}")
    print()


def _save_pca_plot(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    best_k: int,
    out_path: Path,
) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.colormaps["tab10"].resampled(best_k)
    for cluster_id in range(best_k):
        mask = labels == cluster_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[cmap(cluster_id)],
            label=f"Küme {cluster_id}",
            alpha=0.85,
            edgecolors="k",
            linewidths=0.4,
            s=70,
        )

    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}% var.)")
    ax.set_title(
        f"Run Davranış Kümeleri — k={best_k} "
        f"({DEFAULT_EXPERIMENT_ID}, n={len(labels)})"
    )
    ax.legend(title="Küme")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"PCA scatter kaydedildi: {out_path}")


def run_analysis(
    experiment_id: str = DEFAULT_EXPERIMENT_ID,
    db_path: str = RESULTS_DB_PATH,
    *,
    plot_path: str = PLOT_PATH,
    no_plot: bool = False,
) -> int:
    path = Path(db_path)
    if not path.exists():
        print(f"Hata: veritabanı bulunamadı: {path}", file=sys.stderr)
        return 1

    with sqlite3.connect(path) as conn:
        fetched = fetch_behavior_rows(conn, experiment_id)
        rows = [r for r, _, _ in fetched]

    if not rows:
        print(
            f"Hata: '{experiment_id}' için veri yok.",
            file=sys.stderr,
        )
        return 1

    vectors = build_behavior_vectors(rows)
    X = np.array([v.values for v in vectors], dtype=float)

    print("=" * 72)
    print("DAVRANIŞ KÜMELEME ANALİZİ")
    print("=" * 72)
    print(f"  experiment_id : {experiment_id}")
    print(f"  database      : {path}")
    print(f"  run sayısı    : {len(vectors)}")
    print(f"  round penceresi: tüm round'lar (0–{settings.MAX_ROUNDS - 1})")
    print(f"  özellikler    : {', '.join(FEATURE_NAMES)}")
    print()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, silhouette_scores = select_best_k(X_scaled)
    _print_silhouette_scores(silhouette_scores, best_k)

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    print(f"Seçilen k: {best_k}")
    print()
    _cluster_centroids_table(scaler, model)

    print("Run atamaları:")
    print(f"  {'run_id':<28} {'cluster':>7} {'coop/risk':>12}")
    for vec, lbl in sorted(zip(vectors, labels), key=lambda x: (x[1], x[0].run_id)):
        trait = _trait_label(vec.coop_level, vec.risk_level)
        print(f"  {vec.run_id:<28} {lbl:7d} {trait:>12}")
    print()

    _print_trait_overlap(vectors, labels, best_k)

    if not no_plot:
        _save_pca_plot(X_scaled, labels, best_k, Path(plot_path))

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="K-means clustering of run behavior vectors (SQLite only, $0).",
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
        "--plot",
        default=PLOT_PATH,
        help=f"PCA scatter output path (default: {PLOT_PATH})",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib output",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    sys.exit(
        run_analysis(
            args.experiment_id,
            args.db,
            plot_path=args.plot,
            no_plot=args.no_plot,
        )
    )


if __name__ == "__main__":
    main()
