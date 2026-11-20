"""
Module 4: Text & NLP-Based Fraud Indicators
--------------------------------------------
Scans text content extracted from documents for linguistic red flags
commonly associated with fraudulent reports: vague language, excessive
hedging, missing disclosures, inconsistent numbers, etc.

Uses rule-based NLP (no large model dependency).
"""

import re
from typing import Dict, List, Tuple
from collections import Counter


# ── Fraud-indicative linguistic patterns ──

VAGUE_PHRASES = [
    r"\bapproximately\b", r"\broughly\b", r"\babout\b", r"\baround\b",
    r"\bestimated\b", r"\bpossibly\b", r"\bmay\s+be\b", r"\bcould\s+be\b",
    r"\bmight\s+be\b", r"\bin\s+the\s+range\s+of\b", r"\bmore\s+or\s+less\b",
    r"\bto\s+some\s+extent\b",
]

HEDGING_PHRASES = [
    r"\bit\s+appears\s+that\b", r"\bit\s+seems\b", r"\bwe\s+believe\b",
    r"\bto\s+the\s+best\s+of\s+our\s+knowledge\b", r"\bsubject\s+to\b",
    r"\bgenerally\b", r"\bin\s+our\s+opinion\b", r"\bmanagement\s+asserts\b",
    r"\bas\s+far\s+as\s+we\s+know\b",
]

PRESSURE_KEYWORDS = [
    r"\burgent\b", r"\bimmediately\b", r"\basap\b", r"\bno\s+delay\b",
    r"\bdeadline\b", r"\brush\b", r"\bpriority\b", r"\boverride\b",
    r"\bbypass\b", r"\bskip\s+approval\b", r"\bno\s+questions\b",
]

CONCEALMENT_PHRASES = [
    r"\boff[\s-]?book\b", r"\bside\s+agreement\b", r"\binformal\s+arrangement\b",
    r"\boral\s+agreement\b", r"\bkeep\s+confidential\b",
    r"\bdo\s+not\s+disclose\b", r"\bnot\s+for\s+audit\b",
    r"\bunrecorded\b", r"\bundisclosed\b",
]

NUMBER_PATTERN = re.compile(
    r"(?:BDT|USD|EUR|GBP|৳|\$|£|€)\s*[\d,]+(?:\.\d+)?|"
    r"[\d,]+(?:\.\d+)?\s*(?:BDT|USD|EUR|taka|crore|lakh|million|billion)",
    re.IGNORECASE,
)


def _count_pattern_matches(text: str, patterns: List[str]) -> Tuple[int, List[str]]:
    """Count how many times patterns match in the text. Return count + matched phrases."""
    count = 0
    matched = []
    for pat in patterns:
        finds = re.findall(pat, text, re.IGNORECASE)
        count += len(finds)
        matched.extend(finds)
    return count, matched


def extract_numbers(text: str) -> List[str]:
    """Extract monetary values and large numbers from text."""
    return NUMBER_PATTERN.findall(text)


def check_number_consistency(text: str) -> Dict:
    """
    Look for the same label associated with different numbers, e.g.,
    'total revenue' appearing with 500,000 in one place and 520,000 elsewhere.
    """
    lines = text.split("\n")
    label_values: Dict[str, List[str]] = {}
    pattern = re.compile(
        r"(total\s+\w+|net\s+\w+|gross\s+\w+|balance|receivable[s]?|payable[s]?)"
        r"[:\s]+([^\n]{0,50})",
        re.IGNORECASE,
    )
    for line in lines:
        match = pattern.search(line)
        if match:
            label = match.group(1).strip().lower()
            nums = NUMBER_PATTERN.findall(match.group(2))
            if nums:
                label_values.setdefault(label, []).extend(nums)

    inconsistencies = []
    for label, values in label_values.items():
        unique = list(set(values))
        if len(unique) > 1:
            inconsistencies.append({
                "label": label,
                "different_values": unique,
            })

    return {
        "inconsistencies_found": len(inconsistencies),
        "details": inconsistencies,
        "flag": (
            f"🚨 {len(inconsistencies)} numerical inconsistencies found"
            if inconsistencies
            else "✅ No obvious numerical inconsistencies"
        ),
    }


def analyze_text_for_fraud(text: str, source: str = "document") -> Dict:
    """
    Perform comprehensive NLP-based fraud indicator analysis on text.

    Parameters
    ----------
    text : str
        Full text content of the document.
    source : str
        Label for the document.

    Returns
    -------
    dict with detailed fraud indicator results.
    """
    if not text or len(text.strip()) < 50:
        return {
            "source": source,
            "error": "Text too short for meaningful analysis (< 50 characters).",
        }

    word_count = len(text.split())
    results: Dict = {"source": source, "word_count": word_count}
    total_risk = 0

    # 1. Vague language
    vague_count, vague_matches = _count_pattern_matches(text, VAGUE_PHRASES)
    vague_density = round(vague_count / max(word_count, 1) * 1000, 2)  # per 1000 words
    results["vague_language"] = {
        "count": vague_count,
        "density_per_1k_words": vague_density,
        "examples": list(set(vague_matches))[:10],
        "flag": (
            f"⚠️ High vague-language density ({vague_density}/1000 words)"
            if vague_density > 5
            else "✅ Normal"
        ),
    }
    if vague_density > 5:
        total_risk += 15

    # 2. Hedging phrases
    hedge_count, hedge_matches = _count_pattern_matches(text, HEDGING_PHRASES)
    hedge_density = round(hedge_count / max(word_count, 1) * 1000, 2)
    results["hedging_language"] = {
        "count": hedge_count,
        "density_per_1k_words": hedge_density,
        "examples": list(set(hedge_matches))[:10],
        "flag": (
            f"⚠️ Excessive hedging detected ({hedge_density}/1000 words)"
            if hedge_density > 3
            else "✅ Normal"
        ),
    }
    if hedge_density > 3:
        total_risk += 15

    # 3. Pressure / urgency keywords
    pressure_count, pressure_matches = _count_pattern_matches(text, PRESSURE_KEYWORDS)
    results["pressure_language"] = {
        "count": pressure_count,
        "examples": list(set(pressure_matches))[:10],
        "flag": (
            f"🚨 Pressure/urgency language detected ({pressure_count} instances)"
            if pressure_count > 2
            else "✅ Normal"
        ),
    }
    if pressure_count > 2:
        total_risk += 20

    # 4. Concealment phrases
    conceal_count, conceal_matches = _count_pattern_matches(text, CONCEALMENT_PHRASES)
    results["concealment_language"] = {
        "count": conceal_count,
        "examples": list(set(conceal_matches))[:10],
        "flag": (
            f"🚨 Concealment language detected ({conceal_count} instances)"
            if conceal_count > 0
            else "✅ Normal"
        ),
    }
    if conceal_count > 0:
        total_risk += 30

    # 5. Number consistency
    results["number_consistency"] = check_number_consistency(text)
    if results["number_consistency"]["inconsistencies_found"] > 0:
        total_risk += 20

    # 6. Extracted monetary values
    numbers_found = extract_numbers(text)
    results["monetary_values_found"] = {
        "count": len(numbers_found),
        "sample": numbers_found[:20],
    }

    # Overall
    total_risk = min(total_risk, 100)
    if total_risk <= 15:
        verdict = "✅ LOW RISK — No significant linguistic fraud indicators"
    elif total_risk <= 40:
        verdict = "⚠️ MEDIUM RISK — Some linguistic anomalies warrant review"
    else:
        verdict = "🚨 HIGH RISK — Multiple linguistic fraud indicators detected"

    results["overall_risk_score"] = total_risk
    results["verdict"] = verdict

    return results
