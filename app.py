"""
🔍 AI-Powered Fraud Detection Prototype
=========================================
A Streamlit web application for detecting fraud in documents and
financial files using multiple AI/ML and statistical techniques.

Aligned with the research:
"Artificial Intelligence Adoption and Fraud Detection Effectiveness
 in Auditing: Evidence from Organizations in Bangladesh"

Modules:
  1. Benford's Law Analysis
  2. Document Metadata Forensics (PDF / DOCX)
  3. Financial Anomaly Detection (duplicates, outliers, Isolation Forest)
  4. NLP-Based Text Fraud Indicators
  5. Image Forensics (ELA, noise, edge analysis)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
import json
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.benford_analysis import benford_analysis, BENFORD_EXPECTED
from modules.document_forensics import analyze_file
from modules.financial_anomaly import (
    run_full_financial_analysis,
    detect_duplicates,
    detect_round_numbers,
    detect_statistical_outliers,
    detect_weekend_entries,
    isolation_forest_analysis,
)
from modules.text_analysis import analyze_text_for_fraud
from modules.image_forensics import analyze_image

# ── Page config ──
st.set_page_config(
    page_title="🔍 AI Fraud Detection Prototype",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high { background-color: #ff4b4b; color: white; padding: 10px 20px; border-radius: 8px; font-weight: bold; }
    .risk-medium { background-color: #ffa500; color: white; padding: 10px 20px; border-radius: 8px; font-weight: bold; }
    .risk-low { background-color: #21ba45; color: white; padding: 10px 20px; border-radius: 8px; font-weight: bold; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown('<p class="main-header">🔍 AI-Powered Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Prototype for Auditing &mdash; Evidence from Organizations in Bangladesh</p>',
    unsafe_allow_html=True,
)

# ── Sidebar ──
st.sidebar.image("https://img.icons8.com/fluency/96/detective.png", width=80)
st.sidebar.title("Navigation")

module = st.sidebar.radio(
    "Select Detection Module",
    [
        "🏠 Dashboard",
        "📊 Financial Data Analysis",
        "📄 Document Forensics",
        "📝 Text / NLP Analysis",
        "🖼️ Image Forensics",
        "📋 Full Audit Report",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This prototype demonstrates AI-driven fraud detection "
    "techniques applicable to auditing in Bangladesh. "
    "It integrates **Benford's Law**, **Isolation Forest**, "
    "**NLP pattern matching**, and **image forensics**."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Version:** 1.0.0  \n**Author:** Research Prototype")


# ========================================================================
# MODULE: Dashboard
# ========================================================================
if module == "🏠 Dashboard":
    st.header("Welcome to the AI Fraud Detection Prototype")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Detection Modules", "5", help="Number of fraud detection algorithms available")
    with col2:
        st.metric("File Types Supported", "7", help="CSV, XLSX, PDF, DOCX, PNG, JPG, TXT")
    with col3:
        st.metric("ML Algorithms", "3", help="Isolation Forest, Benford's Law, Statistical Outlier")
    with col4:
        st.metric("NLP Patterns", "40+", help="Linguistic fraud indicators checked")

    st.markdown("---")

    st.subheader("🧩 Available Detection Modules")

    mcol1, mcol2 = st.columns(2)

    with mcol1:
        st.markdown("""
        #### 📊 Financial Data Analysis
        - **Benford's Law** — Detects fabricated numbers by checking first-digit distribution
        - **Duplicate Detection** — Finds repeated transactions or invoices
        - **Round-Number Bias** — Flags over-representation of round amounts
        - **Statistical Outliers** — Z-score and IQR-based outlier detection
        - **Isolation Forest** — ML-based multi-dimensional anomaly detection
        - **Weekend Entry Check** — Flags transactions on non-business days

        #### 📄 Document Forensics
        - **PDF Metadata Analysis** — Checks creation dates, tools, tampering signs
        - **DOCX Metadata Analysis** — Author mismatches, revision anomalies
        - **Date Consistency** — Flags future dates and timeline anomalies
        """)

    with mcol2:
        st.markdown("""
        #### 📝 Text / NLP Analysis
        - **Vague Language Detection** — Flags imprecise wording
        - **Hedging Phrase Detection** — Identifies evasive language
        - **Pressure Language** — Detects urgency that may signal fraud
        - **Concealment Phrases** — Finds language suggesting hidden information
        - **Number Consistency** — Checks for contradictory figures

        #### 🖼️ Image Forensics
        - **Error Level Analysis (ELA)** — Detects manipulated regions
        - **Noise Analysis** — Finds inconsistent noise patterns
        - **Edge Consistency** — Identifies splicing artefacts
        """)

    st.markdown("---")

    # Architecture diagram
    st.subheader("🏗️ System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    USER INTERFACE (Streamlit)                    │
    ├────────┬────────┬──────────┬──────────────┬─────────────────────┤
    │ Upload │ Config │ Visualize│   Reports    │   Export Results    │
    ├────────┴────────┴──────────┴──────────────┴─────────────────────┤
    │                     DETECTION ENGINE                            │
    ├──────────┬───────────┬───────────┬───────────┬─────────────────┤
    │ Benford  │ Document  │ Financial │   Text    │     Image       │
    │ Analysis │ Forensics │  Anomaly  │   NLP     │   Forensics     │
    ├──────────┼───────────┼───────────┼───────────┼─────────────────┤
    │ Chi-sq   │ PDF Meta  │ Isolation │ Regex     │ ELA             │
    │ MAD      │ DOCX Meta │ Forest    │ Pattern   │ Noise Analysis  │
    │ Digit    │ Date      │ Z-score   │ Matching  │ Edge Detection  │
    │ Analysis │ Checks    │ IQR       │ NER       │ Block Matching  │
    └──────────┴───────────┴───────────┴───────────┴─────────────────┘
    ```
    """)


# ========================================================================
# MODULE: Financial Data Analysis
# ========================================================================
elif module == "📊 Financial Data Analysis":
    st.header("📊 Financial Data Analysis")
    st.markdown("Upload a CSV or Excel file containing financial transactions to scan for fraud indicators.")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload Financial Data",
        type=["csv", "xlsx", "xls"],
        help="Accepted: CSV, XLSX, XLS",
    )

    # Or use sample data
    use_sample = st.checkbox("Use built-in sample data for demonstration")

    df = None

    if use_sample:
        sample_path = PROJECT_ROOT / "sample_data" / "fraudulent_transactions.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            st.success(f"Loaded sample fraudulent dataset: {len(df)} rows")
        else:
            st.warning("Sample data not found. Run `python generate_sample_data.py` first.")

    elif uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Loaded: {uploaded_file.name} ({len(df)} rows, {len(df.columns)} columns)")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if df is not None:
        # Show data preview
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

        # Column selection
        st.markdown("### Configure Analysis")
        col1, col2 = st.columns(2)
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            amount_col = st.selectbox(
                "Select Amount/Value Column",
                options=["Auto-detect"] + numeric_cols,
            )
        with col2:
            all_cols = df.columns.tolist()
            date_col = st.selectbox(
                "Select Date Column",
                options=["Auto-detect"] + all_cols,
            )

        if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Running all fraud detection algorithms..."):
                amt = None if amount_col == "Auto-detect" else amount_col
                dt = None if date_col == "Auto-detect" else date_col

                results = run_full_financial_analysis(df, amount_column=amt, date_column=dt)

            # ── Display Overall Verdict ──
            overall = results.get("overall", {})
            risk = overall.get("risk_score", 0)

            if risk <= 15:
                st.markdown(f'<div class="risk-low">OVERALL RISK: {risk}/100 — {overall.get("verdict", "")}</div>', unsafe_allow_html=True)
            elif risk <= 40:
                st.markdown(f'<div class="risk-medium">OVERALL RISK: {risk}/100 — {overall.get("verdict", "")}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-high">OVERALL RISK: {risk}/100 — {overall.get("verdict", "")}</div>', unsafe_allow_html=True)

            st.markdown("---")

            # ── Benford's Law ──
            if "benford" in results:
                st.subheader("📐 Benford's Law Analysis")
                b = results["benford"]
                st.markdown(f"**Verdict:** {b.get('verdict', 'N/A')}")

                if b.get("observed") and b.get("expected"):
                    digits = list(range(1, 10))
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=digits,
                        y=[b["expected"][d] * 100 for d in digits],
                        name="Expected (Benford)",
                        marker_color="#636EFA",
                    ))
                    fig.add_trace(go.Bar(
                        x=digits,
                        y=[b["observed"][d] * 100 for d in digits],
                        name="Observed",
                        marker_color="#EF553B",
                    ))
                    fig.update_layout(
                        title="First-Digit Distribution: Expected vs. Observed",
                        xaxis_title="First Digit",
                        yaxis_title="Frequency (%)",
                        barmode="group",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                bcol1, bcol2, bcol3 = st.columns(3)
                with bcol1:
                    st.metric("Chi-Square", f"{b.get('chi2_stat', 'N/A'):.4f}" if b.get('chi2_stat') else "N/A")
                with bcol2:
                    st.metric("p-value", f"{b.get('chi2_p', 'N/A'):.6f}" if b.get('chi2_p') else "N/A")
                with bcol3:
                    st.metric("MAD", f"{b.get('mad', 'N/A'):.6f}" if b.get('mad') else "N/A")

            st.markdown("---")

            # ── Duplicates ──
            if "duplicates" in results:
                st.subheader("🔁 Duplicate Detection")
                d = results["duplicates"]
                st.markdown(f"**Result:** {d.get('flag', '')}")
                st.metric("Duplicates Found", d.get("duplicates_found", 0))

            # ── Round Numbers ──
            if "round_numbers" in results:
                st.subheader("🔢 Round-Number Bias")
                r = results["round_numbers"]
                for f in r.get("flags", []):
                    st.markdown(f"- {f}")
                rcol1, rcol2, rcol3 = st.columns(3)
                with rcol1:
                    st.metric("Ending in 0", f"{r.get('pct_ending_0', 0)}%")
                with rcol2:
                    st.metric("Ending in 00", f"{r.get('pct_ending_00', 0)}%")
                with rcol3:
                    st.metric("Ending in 000", f"{r.get('pct_ending_000', 0)}%")

            st.markdown("---")

            # ── Outliers ──
            if "outliers" in results:
                st.subheader("📈 Statistical Outliers")
                o = results["outliers"]
                st.markdown(f"**Result:** {o.get('flag', '')}")
                ocol1, ocol2, ocol3 = st.columns(3)
                with ocol1:
                    st.metric("Z-Score Outliers", o.get("z_outliers_count", 0))
                with ocol2:
                    st.metric("IQR Outliers", o.get("iqr_outliers_count", 0))
                with ocol3:
                    st.metric("Total Unique", o.get("total_unique_outliers", 0))

            # ── Isolation Forest ──
            if "isolation_forest" in results:
                st.subheader("🌲 Isolation Forest Anomaly Detection")
                iso = results["isolation_forest"]
                st.markdown(f"**Result:** {iso.get('flag', '')}")
                st.metric("ML-Detected Anomalies", iso.get("anomalies_detected", 0))

            # ── Weekend Entries ──
            if "weekend_entries" in results:
                st.subheader("📅 Weekend Entry Check")
                w = results["weekend_entries"]
                st.markdown(f"**Result:** {w.get('flag', '')}")
                st.metric("Weekend Entries", w.get("weekend_entries", 0))

            # ── Export ──
            st.markdown("---")
            st.download_button(
                "📥 Download Analysis Report (JSON)",
                data=json.dumps(results, indent=2, default=str),
                file_name="fraud_analysis_report.json",
                mime="application/json",
            )


# ========================================================================
# MODULE: Document Forensics
# ========================================================================
elif module == "📄 Document Forensics":
    st.header("📄 Document Metadata Forensics")
    st.markdown("Upload a **PDF** or **DOCX** file to inspect its metadata for signs of tampering.")

    uploaded_doc = st.file_uploader(
        "Upload Document",
        type=["pdf", "docx"],
        help="Accepted: PDF, DOCX",
    )

    if uploaded_doc:
        # Save to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_doc.name).suffix) as tmp:
            tmp.write(uploaded_doc.read())
            tmp_path = tmp.name

        with st.spinner("Analyzing document metadata..."):
            result = analyze_file(tmp_path)

        # Clean up
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        if "error" in result:
            st.error(result["error"])
        else:
            # Verdict
            risk = result.get("risk_score", 0)
            verdict = result.get("verdict", "")

            if risk <= 15:
                st.markdown(f'<div class="risk-low">RISK SCORE: {risk}/100 — {verdict}</div>', unsafe_allow_html=True)
            elif risk <= 40:
                st.markdown(f'<div class="risk-medium">RISK SCORE: {risk}/100 — {verdict}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-high">RISK SCORE: {risk}/100 — {verdict}</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Metadata table
            st.subheader("📋 Extracted Metadata")
            meta = result.get("metadata", {})
            if meta:
                meta_df = pd.DataFrame(list(meta.items()), columns=["Property", "Value"])
                st.table(meta_df)

            # Flags
            st.subheader("🚩 Fraud Indicators")
            flags = result.get("flags", [])
            if flags:
                for flag in flags:
                    st.markdown(f"- {flag}")
            else:
                st.success("No fraud indicators detected.")

            # Export
            st.download_button(
                "📥 Download Report (JSON)",
                data=json.dumps(result, indent=2, default=str),
                file_name=f"metadata_report_{result.get('file', 'doc')}.json",
                mime="application/json",
            )


# ========================================================================
# MODULE: Text / NLP Analysis
# ========================================================================
elif module == "📝 Text / NLP Analysis":
    st.header("📝 Text & NLP Fraud Indicator Analysis")
    st.markdown(
        "Paste text from a financial report, audit memo, or correspondence to scan for "
        "linguistic fraud indicators."
    )

    input_method = st.radio("Input method", ["Paste Text", "Upload TXT / DOCX File"], horizontal=True)

    text_input = ""

    if input_method == "Paste Text":
        text_input = st.text_area(
            "Paste document text below",
            height=300,
            placeholder=(
                "Paste the text of a financial report, invoice description, "
                "audit memo, or any document you want to scan..."
            ),
        )
    else:
        uploaded_txt = st.file_uploader("Upload", type=["txt", "docx"])
        if uploaded_txt:
            if uploaded_txt.name.endswith(".txt"):
                text_input = uploaded_txt.read().decode("utf-8", errors="ignore")
            elif uploaded_txt.name.endswith(".docx"):
                try:
                    from docx import Document as DocxDocument
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                        tmp.write(uploaded_txt.read())
                        tmp_path = tmp.name
                    doc = DocxDocument(tmp_path)
                    text_input = "\n".join([p.text for p in doc.paragraphs])
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Error reading DOCX: {e}")

    # Demo text
    if st.checkbox("Load demo text with fraud indicators"):
        text_input = """
CONFIDENTIAL MEMO — FOR INTERNAL USE ONLY

Subject: Urgent — Override Approval for Vendor Payment

To: Accounts Payable
From: M. Rahman, Procurement Manager

We need to process this payment ASAP. Please bypass the standard three-quote
requirement as this is a sole-source vendor. The total amount is approximately
BDT 45,00,000 (roughly forty-five lakh). It appears that the vendor has
provided satisfactory services, and we believe the pricing is competitive
based on our informal arrangement with them.

Please do not disclose this side agreement to the external auditors. The
vendor's invoice (INV-88712) reflects a total of BDT 42,00,000, but the
actual agreed amount is around BDT 45,00,000. The difference is for
unrecorded consulting fees that are subject to a separate oral agreement.

The total revenue for this project is BDT 2.1 crore. As noted in the Q3
report, total revenue was BDT 1.8 crore. We need to reconcile these figures
before the audit team arrives next week. No delay — this is priority.

To the best of our knowledge, all transactions are legitimate. It seems that
the documentation could be improved, but management asserts that the current
process is generally adequate.

Do not keep copies of this memo. Deadline: tomorrow EOD.
        """

    if text_input and st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        with st.spinner("Scanning for linguistic fraud indicators..."):
            result = analyze_text_for_fraud(text_input, source="User Input")

        if "error" in result:
            st.error(result["error"])
        else:
            risk = result.get("overall_risk_score", 0)
            verdict = result.get("verdict", "")

            if risk <= 15:
                st.markdown(f'<div class="risk-low">RISK SCORE: {risk}/100 — {verdict}</div>', unsafe_allow_html=True)
            elif risk <= 40:
                st.markdown(f'<div class="risk-medium">RISK SCORE: {risk}/100 — {verdict}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-high">RISK SCORE: {risk}/100 — {verdict}</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Detail panels
            categories = [
                ("🔤 Vague Language", "vague_language"),
                ("🤔 Hedging Phrases", "hedging_language"),
                ("⚡ Pressure / Urgency", "pressure_language"),
                ("🙈 Concealment Language", "concealment_language"),
                ("🔢 Number Consistency", "number_consistency"),
            ]

            for label, key in categories:
                data = result.get(key, {})
                if isinstance(data, dict):
                    with st.expander(f"{label} — {data.get('flag', data.get('flag', 'N/A'))}"):
                        if "count" in data:
                            st.metric("Occurrences", data["count"])
                        if "density_per_1k_words" in data:
                            st.metric("Density (per 1,000 words)", data["density_per_1k_words"])
                        if "examples" in data and data["examples"]:
                            st.markdown("**Examples found:**")
                            for ex in data["examples"]:
                                st.markdown(f"- *\"{ex}\"*")
                        if "details" in data and data["details"]:
                            st.json(data["details"])

            # Monetary values found
            mv = result.get("monetary_values_found", {})
            if mv.get("count", 0) > 0:
                with st.expander(f"💰 Monetary Values Extracted ({mv['count']} found)"):
                    for v in mv.get("sample", []):
                        st.markdown(f"- `{v}`")

            st.download_button(
                "📥 Download NLP Report (JSON)",
                data=json.dumps(result, indent=2, default=str),
                file_name="nlp_fraud_report.json",
                mime="application/json",
            )


# ========================================================================
# MODULE: Image Forensics
# ========================================================================
elif module == "🖼️ Image Forensics":
    st.header("🖼️ Image Forensics for Scanned Documents")
    st.markdown(
        "Upload a scanned document image (PNG, JPG) to check for signs of "
        "digital manipulation such as splicing, cloning, or content removal."
    )

    uploaded_img = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Accepted: PNG, JPG, JPEG, BMP, TIFF",
    )

    if uploaded_img:
        # Save to temp
        suffix = Path(uploaded_img.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_img.read())
            tmp_path = tmp.name

        # Show original
        st.image(tmp_path, caption="Original Image", use_container_width=True)

        if st.button("🔬 Run Image Forensics", type="primary", use_container_width=True):
            with st.spinner("Performing forensic analysis..."):
                try:
                    result = analyze_image(tmp_path)
                except Exception as e:
                    st.error(f"Analysis error: {e}")
                    result = None

            if result:
                overall = result.get("overall", {})
                risk = overall.get("risk_score", 0)
                verdict = overall.get("verdict", "")

                if risk <= 15:
                    st.markdown(f'<div class="risk-low">RISK SCORE: {risk}/100 — {verdict}</div>', unsafe_allow_html=True)
                elif risk <= 35:
                    st.markdown(f'<div class="risk-medium">RISK SCORE: {risk}/100 — {verdict}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-high">RISK SCORE: {risk}/100 — {verdict}</div>', unsafe_allow_html=True)

                st.markdown("---")

                # ELA
                ela = result.get("error_level_analysis", {})
                if "error" not in ela:
                    st.subheader("🔍 Error Level Analysis (ELA)")
                    for f in ela.get("flags", []):
                        st.markdown(f"- {f}")
                    ela_path = ela.get("ela_image_path")
                    if ela_path and os.path.exists(ela_path):
                        st.image(ela_path, caption="ELA Image — bright regions may indicate manipulation", use_container_width=True)

                # Edge consistency
                edge = result.get("edge_consistency", {})
                if "error" not in edge:
                    st.subheader("📐 Edge Consistency Analysis")
                    for f in edge.get("flags", []):
                        st.markdown(f"- {f}")
                    ecol1, ecol2 = st.columns(2)
                    with ecol1:
                        st.metric("Mean Edge Density", edge.get("mean_edge_density", "N/A"))
                    with ecol2:
                        st.metric("Coefficient of Variation", edge.get("coefficient_of_variation", "N/A"))

                # Noise
                noise = result.get("noise_analysis", {})
                if "error" not in noise and "note" not in noise:
                    st.subheader("🔊 Noise Analysis")
                    for f in noise.get("flags", []):
                        st.markdown(f"- {f}")

                st.download_button(
                    "📥 Download Image Forensics Report (JSON)",
                    data=json.dumps(result, indent=2, default=str),
                    file_name="image_forensics_report.json",
                    mime="application/json",
                )

        # Clean up
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ========================================================================
# MODULE: Full Audit Report
# ========================================================================
elif module == "📋 Full Audit Report":
    st.header("📋 Comprehensive Fraud Audit Report")
    st.markdown(
        "Upload **multiple files** (financial data + documents + images) "
        "and generate a consolidated fraud risk assessment report."
    )

    st.info(
        "💡 **Tip:** Upload a mix of file types to get a comprehensive analysis. "
        "The system will automatically route each file to the appropriate detection module."
    )

    uploaded_files = st.file_uploader(
        "Upload Files for Comprehensive Audit",
        type=["csv", "xlsx", "pdf", "docx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("🚀 Generate Comprehensive Report", type="primary", use_container_width=True):
        all_results = {}
        overall_risks = []

        progress = st.progress(0)
        status = st.status("Analyzing files...", expanded=True)

        for i, f in enumerate(uploaded_files):
            ext = Path(f.name).suffix.lower()
            status.write(f"📂 Analyzing: **{f.name}**")

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            try:
                if ext in (".csv", ".xlsx", ".xls"):
                    if ext == ".csv":
                        df = pd.read_csv(tmp_path)
                    else:
                        df = pd.read_excel(tmp_path)
                    result = run_full_financial_analysis(df)
                    risk = result.get("overall", {}).get("risk_score", 0)
                    all_results[f.name] = {"type": "financial", "result": result, "risk": risk}

                elif ext in (".pdf", ".docx"):
                    result = analyze_file(tmp_path)
                    risk = result.get("risk_score", 0)
                    all_results[f.name] = {"type": "document", "result": result, "risk": risk}

                elif ext == ".txt":
                    text = open(tmp_path, "r", encoding="utf-8", errors="ignore").read()
                    result = analyze_text_for_fraud(text, source=f.name)
                    risk = result.get("overall_risk_score", 0)
                    all_results[f.name] = {"type": "text", "result": result, "risk": risk}

                elif ext in (".png", ".jpg", ".jpeg"):
                    result = analyze_image(tmp_path)
                    risk = result.get("overall", {}).get("risk_score", 0)
                    all_results[f.name] = {"type": "image", "result": result, "risk": risk}

                overall_risks.append(risk)

            except Exception as e:
                all_results[f.name] = {"type": "error", "error": str(e), "risk": 0}

            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            progress.progress((i + 1) / len(uploaded_files))

        status.update(label="✅ Analysis Complete!", state="complete")

        # Summary
        st.markdown("---")
        st.subheader("📊 Consolidated Risk Summary")

        avg_risk = round(np.mean(overall_risks), 1) if overall_risks else 0
        max_risk = max(overall_risks) if overall_risks else 0

        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.metric("Files Analyzed", len(uploaded_files))
        with scol2:
            st.metric("Average Risk Score", f"{avg_risk}/100")
        with scol3:
            st.metric("Highest Risk Score", f"{max_risk}/100")

        # Per-file results
        st.subheader("📄 Per-File Results")
        for fname, data in all_results.items():
            risk = data.get("risk", 0)
            icon = "🟢" if risk <= 15 else "🟡" if risk <= 40 else "🔴"
            with st.expander(f"{icon} {fname} — Risk: {risk}/100"):
                if data.get("type") == "error":
                    st.error(data.get("error"))
                else:
                    st.json(data.get("result", {}))

        # Download consolidated report
        st.download_button(
            "📥 Download Consolidated Report (JSON)",
            data=json.dumps(all_results, indent=2, default=str),
            file_name="consolidated_fraud_audit_report.json",
            mime="application/json",
        )


# ── Footer ──
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#999; font-size:0.85rem;">'
    '🔍 AI-Powered Fraud Detection Prototype v1.0 | '
    'Research: "AI Adoption & Fraud Detection in Auditing — Evidence from Bangladesh" | '
    'Built with Streamlit + scikit-learn + Plotly'
    '</p>',
    unsafe_allow_html=True,
)
