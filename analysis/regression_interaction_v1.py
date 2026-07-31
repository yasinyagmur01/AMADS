# Regression analysis for full_experiment_v1
# Addresses simultaneous trait manipulation confound
# Output saved to analysis/regression_interaction_v1.txt

"""
Multiple linear regression with cooperation × risk interaction on
full_experiment_v1 round-0 extraction fractions (N=45).

Usage (repo root):
    venv/bin/python analysis/regression_interaction_v1.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = _ROOT / "data" / "amads_round0_summary.csv"
OUT_PATH = _ROOT / "analysis" / "regression_interaction_v1.txt"

# INTERPRETATION (full_experiment_v1 OLS with interaction; N=45):
#
# 1. Is the interaction term (cooperation × risk) significant?
#    YES. β₃ = −0.850, SE = 0.165, p = 6.99e-06 (p < 0.05).
#
# 2. Does it change interpretation of the main effects?
#    The cooperation main effect does NOT flip. Simple slopes
#    ∂ŷ/∂coop = β₁ + β₃·risk remain POSITIVE at all tested risk levels
#    (risk=0.2: 0.518; risk=0.5: 0.263; risk=0.8: 0.008), so inverse
#    fidelity holds across the design grid. The interaction attenuates
#    the cooperation slope toward zero at high risk rather than reversing it.
#    (Note: in this operationalization inverse fidelity is a POSITIVE
#    coop→extraction slope, not a negative one.)
#
# 3. N/A — interaction is significant; the additive-only claim is not made.
#    Micro-A/B (risk fixed at 0.2) remains valid: that is exactly where the
#    cooperation simple slope is largest.
#


def load_full_experiment() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    full = df[df["experiment_id"] == "full_experiment_v1"].copy()
    # Map export column names to regression variable names.
    full = full.rename(
        columns={
            "coop_value": "cooperation_assigned",
            "risk_value": "risk_tolerance_assigned",
            "extraction_fraction_r0": "extraction_fraction_round0",
        }
    )
    return full


def fit_interaction_ols(df: pd.DataFrame):
    # Formula uses C-style interaction via `:`; main effects + product.
    model = smf.ols(
        "extraction_fraction_round0 ~ cooperation_assigned"
        " + risk_tolerance_assigned"
        " + cooperation_assigned:risk_tolerance_assigned",
        data=df,
    )
    return model.fit()


def build_interpretation(result) -> str:
    params = result.params
    pvalues = result.pvalues
    b_interact = float(params["cooperation_assigned:risk_tolerance_assigned"])
    p_interact = float(pvalues["cooperation_assigned:risk_tolerance_assigned"])
    b_coop = float(params["cooperation_assigned"])
    p_coop = float(pvalues["cooperation_assigned"])
    b_risk = float(params["risk_tolerance_assigned"])
    p_risk = float(pvalues["risk_tolerance_assigned"])

    lines = [
        "",
        "=" * 72,
        "INTERPRETATION",
        "=" * 72,
        "",
        "1. Is the interaction term (cooperation × risk) significant?",
    ]

    if p_interact < 0.05:
        lines.append(
            f"   YES — β₃ = {b_interact:.6f}, p = {p_interact:.6g} (p < 0.05)."
        )
        lines.extend(
            [
                "",
                "2. Does the interaction change interpretation of main effects?",
                "   Simple slope of cooperation at risk = r:",
                "     ∂ŷ/∂coop = β₁ + β₃·r",
            ]
        )
        for r in (0.2, 0.5, 0.8):
            slope = b_coop + b_interact * r
            lines.append(f"     risk={r}: slope = {slope:.6f}")
        signs = [b_coop + b_interact * r for r in (0.2, 0.5, 0.8)]
        if all(s > 0 for s in signs):
            lines.append(
                "   Cooperation main effect remains POSITIVE (inverse fidelity)"
                " across all tested risk levels; interaction modulates magnitude,"
                " not direction."
            )
        elif all(s < 0 for s in signs):
            lines.append(
                "   Cooperation slope remains NEGATIVE across all tested risk"
                " levels (expected fidelity direction)."
            )
        else:
            lines.append(
                "   Cooperation slope CHANGES SIGN across risk levels — inverse"
                " fidelity does not hold uniformly; interpretation is risk-conditional."
            )
        lines.append("")
        lines.append("3. N/A (interaction is significant; additive-only claim not made).")
    else:
        lines.append(
            f"   NO — β₃ = {b_interact:.6f}, p = {p_interact:.6g} (p ≥ 0.05)."
        )
        lines.extend(
            [
                "",
                "2. N/A (interaction not significant).",
                "",
                "3. The additive model is sufficient. The cooperation × risk",
                "   interaction is not statistically significant, so simultaneous",
                "   trait manipulation does not confound the main effects.",
                "   The micro-A/B defense holds (risk fixed at 0.2 still isolates",
                "   the cooperation effect).",
                f"   Main effects for reference: β_coop = {b_coop:.6f}"
                f" (p = {p_coop:.6g}), β_risk = {b_risk:.6f} (p = {p_risk:.6g}).",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    df = load_full_experiment()
    n = len(df)
    print("=== DATA PREVIEW ===")
    print("Columns:", list(df.columns))
    print(f"N (full_experiment_v1) = {n}")
    print(df.head(10).to_string())
    print()

    if n != 45:
        print(f"WARNING: expected N=45, found N={n}; proceeding with available rows.")

    result = fit_interaction_ols(df)
    summary = result.summary().as_text()

    # Explicit coefficient report
    terms = [
        "cooperation_assigned",
        "risk_tolerance_assigned",
        "cooperation_assigned:risk_tolerance_assigned",
    ]
    coef_lines = [
        "",
        "=" * 72,
        "COEFFICIENT REPORT (β, SE, p)",
        "=" * 72,
    ]
    for term in terms:
        coef_lines.append(
            f"  {term}:"
            f"  β = {result.params[term]:.6f},"
            f"  SE = {result.bse[term]:.6f},"
            f"  p = {result.pvalues[term]:.6g}"
        )
    coef_lines.append(f"  Intercept:  β = {result.params['Intercept']:.6f}")
    coef_lines.append(f"  R-squared          = {result.rsquared:.6f}")
    coef_lines.append(f"  Adj. R-squared     = {result.rsquared_adj:.6f}")
    coef_lines.append(f"  N                  = {int(result.nobs)}")
    p_int = float(result.pvalues["cooperation_assigned:risk_tolerance_assigned"])
    coef_lines.append(
        f"  Interaction significant (p < 0.05)?  {'YES' if p_int < 0.05 else 'NO'}"
        f"  (p = {p_int:.6g})"
    )

    interpretation = build_interpretation(result)

    # Update module docstring-style comment block with concrete answers.
    comment_block = (
        "\n"
        "# INTERPRETATION COMMENT BLOCK\n"
        + "\n".join("# " + line if line else "#" for line in interpretation.splitlines())
        + "\n"
    )

    out_text = "\n".join(
        [
            "Regression analysis for full_experiment_v1",
            "Model: extraction_fraction_round0 ~ cooperation_assigned"
            " + risk_tolerance_assigned"
            " + cooperation_assigned:risk_tolerance_assigned",
            f"Source: {CSV_PATH.relative_to(_ROOT)}",
            f"N = {n}",
            "",
            summary,
            "\n".join(coef_lines),
            interpretation,
            comment_block,
        ]
    )

    OUT_PATH.write_text(out_text, encoding="utf-8")
    print(out_text)
    print(f"\nSaved to {OUT_PATH.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
