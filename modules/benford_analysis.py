"""
Module 1: Benford's Law Analysis
---------------------------------
Applies Benford's Law (first-digit law) to financial datasets to detect
anomalies that may indicate fabricated or manipulated numbers.

Benford's Law predicts that in many naturally occurring datasets, the
leading digit d (1–9) appears with probability P(d) = log10(1 + 1/d).
Fraudulent data often deviates significantly from this distribution.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List


# Expected Benford distribution for first digits 1-9
BENFORD_EXPECTED = {
    d: np.log10(1 + 1 / d) for d in range(1, 10)
}


def extract_first_digit(value) -> int | None:
    """Extract the first significant (non-zero) digit from a number."""
    try:
        num = abs(float(value))
        if num == 0:
            return None
        s = f"{num:.10f}".lstrip("0").lstrip(".")
        for ch in s:
            if ch.isdigit() and ch != "0":
                return int(ch)
        return None
    except (ValueError, TypeError):
        return None


def benford_analysis(data: pd.Series, column_name: str = "values") -> Dict:
    """
    Perform Benford's Law analysis on a numeric series.

    Parameters
    ----------
    data : pd.Series
        Numeric data to analyse.
    column_name : str
        Label used in the report.

    Returns
    -------
    dict with keys:
        - observed    : dict {digit: proportion}
        - expected    : dict {digit: proportion}
        - chi2_stat   : float
        - chi2_p      : float
        - mad         : float  (Mean Absolute Deviation)
        - verdict     : str    ('Conforming', 'Suspicious', 'Non-conforming')
        - digit_flags : list of digits whose deviation exceeds threshold
        - summary     : str    human-readable summary
    """

    first_digits = data.apply(extract_first_digit).dropna().astype(int)
    n = len(first_digits)

    if n < 50:
        return {
            "observed": {},
            "expected": dict(BENFORD_EXPECTED),
            "chi2_stat": None,
            "chi2_p": None,
            "mad": None,
            "verdict": "Insufficient data (need ≥ 50 values)",
            "digit_flags": [],
            "summary": f"Only {n} usable values found in '{column_name}'. "
                       f"Benford analysis requires at least 50 observations.",
        }

    # Observed proportions
    counts = first_digits.value_counts().sort_index()
    observed = {d: counts.get(d, 0) / n for d in range(1, 10)}

    # Chi-square goodness-of-fit
    obs_counts = np.array([counts.get(d, 0) for d in range(1, 10)])
    exp_counts = np.array([BENFORD_EXPECTED[d] * n for d in range(1, 10)])
    chi2_stat, chi2_p = stats.chisquare(obs_counts, f_exp=exp_counts)

    # Mean Absolute Deviation
    mad = np.mean([abs(observed[d] - BENFORD_EXPECTED[d]) for d in range(1, 10)])

    # Flag individual digits deviating > 2 percentage points
    digit_flags = [
        d for d in range(1, 10)
        if abs(observed[d] - BENFORD_EXPECTED[d]) > 0.02
    ]

    # Verdict
    if chi2_p > 0.05 and mad < 0.008:
        verdict = "✅ Conforming — data follows Benford's Law"
    elif chi2_p > 0.01 and mad < 0.015:
        verdict = "⚠️ Marginally Suspicious — minor deviations detected"
    else:
        verdict = "🚨 Non-conforming — significant deviation from Benford's Law (potential data manipulation)"

    summary = (
        f"Benford's Law Analysis for '{column_name}' (n = {n}):\n"
        f"  Chi-square statistic : {chi2_stat:.4f}\n"
        f"  p-value              : {chi2_p:.6f}\n"
        f"  Mean Abs. Deviation  : {mad:.6f}\n"
        f"  Flagged digits       : {digit_flags if digit_flags else 'None'}\n"
        f"  Verdict              : {verdict}"
    )

    return {
        "observed": observed,
        "expected": dict(BENFORD_EXPECTED),
        "chi2_stat": float(chi2_stat),
        "chi2_p": float(chi2_p),
        "mad": float(mad),
        "verdict": verdict,
        "digit_flags": digit_flags,
        "summary": summary,
        "n": n,
    }
