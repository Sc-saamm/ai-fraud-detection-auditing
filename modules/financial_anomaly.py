"""
Module 3: Financial Anomaly Detection
--------------------------------------
Uses statistical and ML methods to detect anomalies in financial
datasets — duplicate transactions, outliers, round-number bias,
weekend/holiday entries, and isolation-forest–based anomaly scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def detect_duplicates(df: pd.DataFrame, subset: List[str] = None) -> Dict:
    """
    Detect duplicate rows in a DataFrame.
    If subset is None, checks all columns.
    """
    if subset is None:
        subset = df.columns.tolist()

    valid_subset = [c for c in subset if c in df.columns]
    if not valid_subset:
        return {"duplicates_found": 0, "details": "No valid columns to check."}

    dupes = df[df.duplicated(subset=valid_subset, keep=False)]
    return {
        "duplicates_found": len(dupes),
        "duplicate_indices": dupes.index.tolist()[:100],  # cap at 100 for display
        "pct_of_total": round(len(dupes) / max(len(df), 1) * 100, 2),
        "flag": "🚨 Duplicates detected" if len(dupes) > 0 else "✅ No duplicates",
    }


def detect_round_numbers(series: pd.Series, column_name: str = "amount") -> Dict:
    """
    Detect over-representation of round numbers (ending in 0 or 00).
    In natural financial data, ~10% of values end in 0 and ~1% in 00.
    Significantly higher proportions may indicate fabricated data.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)
    if n < 20:
        return {"flag": "Insufficient data", "n": n}

    # Ends in 0
    ends_0 = sum(1 for v in clean if v % 10 == 0)
    pct_0 = round(ends_0 / n * 100, 2)

    # Ends in 00
    ends_00 = sum(1 for v in clean if v % 100 == 0)
    pct_00 = round(ends_00 / n * 100, 2)

    # Ends in 000
    ends_000 = sum(1 for v in clean if v % 1000 == 0)
    pct_000 = round(ends_000 / n * 100, 2)

    flags = []
    risk = 0
    if pct_0 > 20:
        flags.append(f"⚠️ {pct_0}% of values end in 0 (expected ~10%)")
        risk += 15
    if pct_00 > 10:
        flags.append(f"⚠️ {pct_00}% of values end in 00 (expected ~1%)")
        risk += 20
    if pct_000 > 5:
        flags.append(f"🚨 {pct_000}% of values end in 000 — strong round-number bias")
        risk += 25

    return {
        "column": column_name,
        "n": n,
        "pct_ending_0": pct_0,
        "pct_ending_00": pct_00,
        "pct_ending_000": pct_000,
        "flags": flags if flags else ["✅ Round-number distribution appears normal"],
        "risk_score": min(risk, 100),
    }


def detect_statistical_outliers(
    series: pd.Series,
    column_name: str = "amount",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> Dict:
    """
    Detect outliers using both Z-score and IQR methods.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)
    if n < 10:
        return {"flag": "Insufficient data", "n": n}

    # Z-score method
    mean = clean.mean()
    std = clean.std()
    if std == 0:
        z_outliers_idx = []
    else:
        z_scores = (clean - mean) / std
        z_outliers_idx = clean[z_scores.abs() > z_threshold].index.tolist()

    # IQR method
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    iqr_outliers_idx = clean[(clean < lower) | (clean > upper)].index.tolist()

    # Combined
    all_outliers = list(set(z_outliers_idx + iqr_outliers_idx))

    return {
        "column": column_name,
        "n": n,
        "mean": round(float(mean), 2),
        "std": round(float(std), 2),
        "z_outliers_count": len(z_outliers_idx),
        "iqr_outliers_count": len(iqr_outliers_idx),
        "total_unique_outliers": len(all_outliers),
        "outlier_indices": all_outliers[:50],
        "pct_outliers": round(len(all_outliers) / n * 100, 2),
        "flag": (
            "🚨 Significant outliers detected"
            if len(all_outliers) > n * 0.05
            else "✅ Outlier proportion within normal range"
        ),
    }


def detect_weekend_entries(
    df: pd.DataFrame,
    date_column: str,
) -> Dict:
    """Flag transactions posted on weekends (Saturday/Sunday) or holidays."""
    try:
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    except (KeyError, TypeError):
        return {"error": f"Column '{date_column}' not found or not parseable."}

    n = len(dates)
    if n == 0:
        return {"error": "No valid dates found."}

    weekend = dates[dates.dt.dayofweek.isin([4, 5])]  # Friday=4, Saturday=5 (Bangladesh weekend)
    pct = round(len(weekend) / n * 100, 2)

    return {
        "total_entries": n,
        "weekend_entries": len(weekend),
        "pct_weekend": pct,
        "weekend_indices": weekend.index.tolist()[:50],
        "flag": (
            f"⚠️ {pct}% of entries on weekends (Fri-Sat in BD context)"
            if pct > 5
            else "✅ Weekend entry proportion appears normal"
        ),
    }


def isolation_forest_analysis(
    df: pd.DataFrame,
    numeric_columns: List[str],
    contamination: float = 0.05,
) -> Dict:
    """
    Apply Isolation Forest to detect anomalous rows in multi-dimensional
    financial data.
    """
    valid_cols = [c for c in numeric_columns if c in df.columns]
    if not valid_cols:
        return {"error": "No valid numeric columns provided."}

    subset = df[valid_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < 20:
        return {"error": "Insufficient data for Isolation Forest (need ≥ 20 rows)."}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    preds = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)

    anomaly_mask = preds == -1
    anomaly_indices = subset.index[anomaly_mask].tolist()

    return {
        "columns_used": valid_cols,
        "total_rows": len(subset),
        "anomalies_detected": int(anomaly_mask.sum()),
        "anomaly_indices": anomaly_indices[:100],
        "pct_anomalies": round(anomaly_mask.sum() / len(subset) * 100, 2),
        "anomaly_scores": {
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "mean": round(float(scores.mean()), 4),
        },
        "flag": (
            f"🚨 {int(anomaly_mask.sum())} anomalous transactions detected by Isolation Forest"
            if anomaly_mask.sum() > 0
            else "✅ No anomalies detected"
        ),
    }


def run_full_financial_analysis(
    df: pd.DataFrame,
    amount_column: str = None,
    date_column: str = None,
) -> Dict:
    """
    Run all financial anomaly detection tests on a DataFrame.
    Auto-detects amount and date columns if not specified.
    """
    results = {}

    # Auto-detect columns
    if amount_column is None:
        for col in df.columns:
            if any(kw in col.lower() for kw in ["amount", "total", "value", "price", "debit", "credit"]):
                amount_column = col
                break
    if date_column is None:
        for col in df.columns:
            if any(kw in col.lower() for kw in ["date", "time", "posted", "entry"]):
                date_column = col
                break

    # 1. Duplicates
    results["duplicates"] = detect_duplicates(df)

    # 2. Benford (import inline to avoid circular)
    if amount_column and amount_column in df.columns:
        from modules.benford_analysis import benford_analysis
        results["benford"] = benford_analysis(df[amount_column], amount_column)

        # 3. Round numbers
        results["round_numbers"] = detect_round_numbers(df[amount_column], amount_column)

        # 4. Outliers
        results["outliers"] = detect_statistical_outliers(df[amount_column], amount_column)

    # 5. Weekend entries
    if date_column and date_column in df.columns:
        results["weekend_entries"] = detect_weekend_entries(df, date_column)

    # 6. Isolation Forest on all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 1:
        results["isolation_forest"] = isolation_forest_analysis(df, numeric_cols)

    # Overall risk
    scores = []
    for key, val in results.items():
        if isinstance(val, dict) and "risk_score" in val:
            scores.append(val["risk_score"])
    overall = round(np.mean(scores), 1) if scores else 0

    if overall <= 15:
        overall_verdict = "✅ LOW RISK — Financial data appears legitimate"
    elif overall <= 40:
        overall_verdict = "⚠️ MEDIUM RISK — Some anomalies warrant investigation"
    else:
        overall_verdict = "🚨 HIGH RISK — Multiple fraud indicators detected"

    results["overall"] = {
        "risk_score": overall,
        "verdict": overall_verdict,
        "amount_column_used": amount_column,
        "date_column_used": date_column,
    }

    return results
