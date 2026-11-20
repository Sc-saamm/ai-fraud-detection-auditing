# 🔍 AI-Powered Fraud Detection Prototype

> **Research:** "Artificial Intelligence Adoption and Fraud Detection Effectiveness in Auditing: Evidence from Organizations in Bangladesh"

A comprehensive prototype demonstrating AI/ML-driven fraud detection techniques for auditing, built with Python, Streamlit, and scikit-learn.

---

## 🧩 Detection Modules

| Module | Techniques | File Types |
|--------|-----------|------------|
| **Financial Data Analysis** | Benford's Law, Duplicate Detection, Round-Number Bias, Statistical Outliers (Z-score/IQR), Isolation Forest, Weekend Entry Check | CSV, XLSX |
| **Document Forensics** | PDF/DOCX metadata analysis, timestamp consistency, tool-chain verification, author mismatch detection | PDF, DOCX |
| **Text/NLP Analysis** | Vague language detection, hedging phrases, pressure/urgency keywords, concealment language, number consistency | TXT, DOCX, paste |
| **Image Forensics** | Error Level Analysis (ELA), noise analysis, edge consistency | PNG, JPG, TIFF |
| **Full Audit Report** | Consolidated multi-file analysis with risk scoring | All above |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data (Optional)

```bash
python generate_sample_data.py
```

This creates clean and fraudulent sample datasets in `sample_data/`.

### 3. Run the Application

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
fraud-detection-prototype/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── generate_sample_data.py     # Sample data generator
├── README.md                   # This file
├── modules/
│   ├── __init__.py
│   ├── benford_analysis.py     # Benford's Law detection
│   ├── document_forensics.py   # PDF/DOCX metadata analysis
│   ├── financial_anomaly.py    # Statistical & ML anomaly detection
│   ├── text_analysis.py        # NLP-based fraud indicators
│   └── image_forensics.py      # Image tampering detection
├── sample_data/                # Generated sample datasets
│   ├── clean_transactions.csv
│   ├── clean_transactions.xlsx
│   ├── fraudulent_transactions.csv
│   └── fraudulent_transactions.xlsx
└── uploads/                    # Temporary upload directory
```

---

## 🔬 How Each Module Works

### Benford's Law Analysis
Benford's Law predicts that in naturally occurring datasets, the first digit `d` appears with probability `P(d) = log₁₀(1 + 1/d)`. Fabricated financial data often violates this distribution. We use **Chi-square goodness-of-fit** and **Mean Absolute Deviation (MAD)** to quantify conformance.

### Isolation Forest
An unsupervised ML algorithm that isolates anomalies by randomly partitioning features. Transactions requiring fewer partitions to isolate are flagged as anomalous. Configured with 200 estimators and 5% contamination rate.

### Error Level Analysis (ELA)
Re-saves an image at a known JPEG quality level and computes pixel-level differences. Manipulated regions typically show higher error levels because they were saved at different compression quality than the original.

### NLP Fraud Indicators
Pattern-matching engine scanning for 40+ linguistic red flags across 5 categories:
- **Vague language** ("approximately", "roughly")
- **Hedging** ("it appears that", "we believe")
- **Pressure** ("urgent", "bypass", "ASAP")
- **Concealment** ("off-book", "do not disclose", "unrecorded")
- **Number inconsistencies** (same label, different values)

---

## 📊 Risk Scoring

Each module produces a **risk score (0–100)**:

| Score Range | Label | Meaning |
|-------------|-------|---------|
| 0–15 | 🟢 Low Risk | Data appears legitimate |
| 16–40 | 🟡 Medium Risk | Anomalies warrant investigation |
| 41–100 | 🔴 High Risk | Multiple fraud indicators detected |

---

## 🛡️ Limitations & Disclaimer

- This is a **research prototype** — not production audit software.
- Results are **indicative**, not conclusive. All flagged items require human auditor judgment.
- Image forensics requires `scipy` for full noise analysis.
- The NLP module uses rule-based patterns; a production system would use transformer models.

---

## 📚 References

- Nigrini, M. J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. Wiley.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *ICDM*.
- Krawetz, N. (2007). A Picture's Worth: Digital Image Analysis and Forensics. *Black Hat Briefings*.

---

**License:** MIT | **Author:** Research Prototype
