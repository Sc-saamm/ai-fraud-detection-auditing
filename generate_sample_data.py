"""
Generate sample datasets for testing the fraud detection prototype.
Creates both clean and fraudulent financial data.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)


def generate_clean_transactions(n: int = 500) -> pd.DataFrame:
    """Generate a dataset of realistic, non-fraudulent financial transactions."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")  # business days
    vendors = [f"Vendor_{i:03d}" for i in range(1, 51)]
    departments = ["Finance", "Operations", "Sales", "HR", "IT", "Procurement"]

    data = {
        "Transaction_ID": [f"TXN-{i:05d}" for i in range(1, n + 1)],
        "Date": np.random.choice(dates, n),
        "Vendor": np.random.choice(vendors, n),
        "Department": np.random.choice(departments, n),
        "Amount_BDT": np.round(np.random.lognormal(mean=10, sigma=1.5, size=n), 2),
        "Invoice_Number": [f"INV-{np.random.randint(10000,99999)}" for _ in range(n)],
        "Approved_By": np.random.choice(["Manager_A", "Manager_B", "Manager_C", "Director_X"], n),
    }

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def generate_fraudulent_transactions(n: int = 500) -> pd.DataFrame:
    """
    Generate a dataset with embedded fraud patterns:
    - Duplicate entries (same invoice, different amounts)
    - Round-number bias
    - Weekend entries
    - Benford-violating amounts
    - Unusually large outliers
    """
    df = generate_clean_transactions(n)

    # --- Inject fraud pattern 1: Duplicate invoices ---
    dup_indices = np.random.choice(range(50, n), size=15, replace=False)
    for idx in dup_indices:
        df.loc[idx, "Invoice_Number"] = df.loc[idx - 1, "Invoice_Number"]
        df.loc[idx, "Amount_BDT"] = df.loc[idx - 1, "Amount_BDT"] * np.random.uniform(0.98, 1.02)

    # --- Inject fraud pattern 2: Round numbers ---
    round_indices = np.random.choice(range(n), size=80, replace=False)
    for idx in round_indices:
        df.loc[idx, "Amount_BDT"] = float(np.random.choice([
            10000, 20000, 50000, 100000, 200000, 500000, 1000000
        ]))

    # --- Inject fraud pattern 3: Weekend entries ---
    weekend_indices = np.random.choice(range(n), size=25, replace=False)
    saturdays = pd.date_range("2024-01-06", periods=52, freq="W-SAT")
    for idx in weekend_indices:
        df.loc[idx, "Date"] = np.random.choice(saturdays)

    # --- Inject fraud pattern 4: Benford violations (too many leading 1s and 5s) ---
    benford_indices = np.random.choice(range(n), size=60, replace=False)
    for idx in benford_indices:
        leading = np.random.choice([1, 5])
        df.loc[idx, "Amount_BDT"] = float(leading) * 10 ** np.random.randint(3, 6) + np.random.randint(0, 999)

    # --- Inject fraud pattern 5: Extreme outliers ---
    outlier_indices = np.random.choice(range(n), size=8, replace=False)
    for idx in outlier_indices:
        df.loc[idx, "Amount_BDT"] = float(np.random.uniform(50_000_000, 200_000_000))

    df["Date"] = pd.to_datetime(df["Date"])
    return df


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "sample_data")
    os.makedirs(output_dir, exist_ok=True)

    clean = generate_clean_transactions(500)
    clean.to_csv(os.path.join(output_dir, "clean_transactions.csv"), index=False)
    clean.to_excel(os.path.join(output_dir, "clean_transactions.xlsx"), index=False)
    print(f"✅ Clean dataset saved ({len(clean)} rows)")

    fraud = generate_fraudulent_transactions(500)
    fraud.to_csv(os.path.join(output_dir, "fraudulent_transactions.csv"), index=False)
    fraud.to_excel(os.path.join(output_dir, "fraudulent_transactions.xlsx"), index=False)
    print(f"✅ Fraudulent dataset saved ({len(fraud)} rows)")

    print("\nSample data generated in:", output_dir)
