"""
Module 2: Document Metadata Forensics
--------------------------------------
Analyses PDF and DOCX file metadata to detect signs of tampering,
such as mismatched creation/modification dates, suspicious authoring
tools, or inconsistent producer fields.
"""

import os
import datetime
from typing import Dict, List, Optional
from pathlib import Path

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


# Known legitimate PDF producers / creators
KNOWN_LEGITIMATE_PRODUCERS = {
    "Microsoft", "Adobe", "LibreOffice", "Google", "Apple",
    "Foxit", "Nitro", "PDFium", "Chromium", "Mozilla",
}

# Suspicious tool signatures that may indicate forgery
SUSPICIOUS_TOOL_KEYWORDS = [
    "pdf editor", "pdf eraser", "infix", "pdf-xchange",
    "pdftk", "ghostscript", "hexeditor", "hex editor",
    "nitro pro", "phantompdf",  # legitimate but watch for edits
]


def _parse_pdf_date(date_str: str) -> Optional[datetime.datetime]:
    """Parse a PDF date string like D:20231015123045+05'30' into datetime."""
    if not date_str:
        return None
    s = date_str.strip()
    if s.startswith("D:"):
        s = s[2:]
    # Remove timezone offset for simplicity
    s = s.replace("'", "").replace("Z", "")
    for fmt in ("%Y%m%d%H%M%S%z", "%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"):
        try:
            return datetime.datetime.strptime(s[:len(fmt.replace('%', ''))+6], fmt)
        except (ValueError, IndexError):
            continue
    return None


def analyze_pdf_metadata(filepath: str) -> Dict:
    """
    Analyse a PDF file's metadata for fraud indicators.

    Returns
    -------
    dict with keys: metadata, flags, risk_score, summary
    """
    if PdfReader is None:
        return {"error": "PyPDF2 not installed"}

    flags: List[str] = []
    risk_score = 0
    metadata = {}

    try:
        reader = PdfReader(filepath)
        info = reader.metadata
        num_pages = len(reader.pages)

        if info:
            metadata = {
                "title": str(info.get("/Title", "N/A")),
                "author": str(info.get("/Author", "N/A")),
                "creator": str(info.get("/Creator", "N/A")),
                "producer": str(info.get("/Producer", "N/A")),
                "creation_date": str(info.get("/CreationDate", "N/A")),
                "modification_date": str(info.get("/ModDate", "N/A")),
                "num_pages": num_pages,
            }

            # --- Flag 1: Missing author ---
            if info.get("/Author") in (None, "", "N/A"):
                flags.append("⚠️ Author field is empty — may indicate auto-generated or stripped metadata")
                risk_score += 10

            # --- Flag 2: Creator ≠ Producer mismatch ---
            creator = str(info.get("/Creator", "")).lower()
            producer = str(info.get("/Producer", "")).lower()
            if creator and producer and creator != producer:
                flags.append(
                    f"⚠️ Creator/Producer mismatch: Creator='{info.get('/Creator')}', "
                    f"Producer='{info.get('/Producer')}' — file may have been re-processed"
                )
                risk_score += 15

            # --- Flag 3: Suspicious editing tools ---
            combined = f"{creator} {producer}"
            for kw in SUSPICIOUS_TOOL_KEYWORDS:
                if kw in combined:
                    flags.append(f"🚨 Suspicious tool detected in metadata: '{kw}'")
                    risk_score += 25
                    break

            # --- Flag 4: Creation date after modification date ---
            cdate = _parse_pdf_date(str(info.get("/CreationDate", "")))
            mdate = _parse_pdf_date(str(info.get("/ModDate", "")))
            if cdate and mdate:
                if cdate > mdate:
                    flags.append("🚨 Creation date is AFTER modification date — metadata likely tampered")
                    risk_score += 30
                elif (mdate - cdate).total_seconds() < 5:
                    flags.append("⚠️ Creation and modification dates are nearly identical — single-pass generation")
                    risk_score += 5

            # --- Flag 5: Future dates ---
            now = datetime.datetime.now()
            for label, dt in [("Creation", cdate), ("Modification", mdate)]:
                if dt and dt.replace(tzinfo=None) > now:
                    flags.append(f"🚨 {label} date is in the future ({dt}) — clock manipulation or forgery")
                    risk_score += 30

            # --- Flag 6: No known legitimate producer ---
            if producer and not any(lp.lower() in producer for lp in KNOWN_LEGITIMATE_PRODUCERS):
                flags.append(f"⚠️ Unknown PDF producer: '{info.get('/Producer')}' — verify authenticity")
                risk_score += 10

        else:
            flags.append("🚨 PDF has NO metadata at all — metadata may have been deliberately stripped")
            risk_score += 25

        # --- Flag 7: File-system date vs. metadata date ---
        file_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
        metadata["file_system_modified"] = file_mtime.isoformat()
        if info:
            mdate = _parse_pdf_date(str(info.get("/ModDate", "")))
            if mdate:
                diff = abs((file_mtime - mdate.replace(tzinfo=None)).total_seconds())
                if diff > 86400 * 30:  # > 30 days
                    flags.append(
                        f"⚠️ File-system modification date ({file_mtime.date()}) differs from "
                        f"PDF metadata date ({mdate.date()}) by > 30 days"
                    )
                    risk_score += 15

    except Exception as e:
        flags.append(f"❌ Error reading PDF: {str(e)}")
        risk_score += 20

    risk_score = min(risk_score, 100)

    if risk_score <= 15:
        verdict = "✅ Low Risk — metadata appears consistent"
    elif risk_score <= 40:
        verdict = "⚠️ Medium Risk — some anomalies detected, manual review recommended"
    else:
        verdict = "🚨 High Risk — multiple metadata red flags, possible document tampering"

    return {
        "file": os.path.basename(filepath),
        "metadata": metadata,
        "flags": flags,
        "risk_score": risk_score,
        "verdict": verdict,
    }


def analyze_docx_metadata(filepath: str) -> Dict:
    """Analyse a DOCX file's metadata for fraud indicators."""
    if DocxDocument is None:
        return {"error": "python-docx not installed"}

    flags: List[str] = []
    risk_score = 0
    metadata = {}

    try:
        doc = DocxDocument(filepath)
        props = doc.core_properties

        metadata = {
            "title": str(props.title or "N/A"),
            "author": str(props.author or "N/A"),
            "last_modified_by": str(props.last_modified_by or "N/A"),
            "created": str(props.created or "N/A"),
            "modified": str(props.modified or "N/A"),
            "revision": str(props.revision or "N/A"),
            "category": str(props.category or "N/A"),
            "num_paragraphs": len(doc.paragraphs),
            "num_tables": len(doc.tables),
        }

        # Flag 1: Author differs from last_modified_by
        if (props.author and props.last_modified_by and
                props.author != props.last_modified_by):
            flags.append(
                f"⚠️ Author ('{props.author}') ≠ Last Modified By ('{props.last_modified_by}') "
                f"— document was edited by a different person"
            )
            risk_score += 10

        # Flag 2: Very high revision count
        try:
            rev = int(props.revision) if props.revision else 0
        except (ValueError, TypeError):
            rev = 0
        if rev > 50:
            flags.append(f"⚠️ Unusually high revision count ({rev}) — extensive editing detected")
            risk_score += 15

        # Flag 3: Missing author
        if not props.author:
            flags.append("⚠️ Author field is empty — metadata may have been stripped")
            risk_score += 10

        # Flag 4: Creation date after modification date
        if props.created and props.modified:
            if props.created > props.modified:
                flags.append("🚨 Creation date is AFTER modification date — metadata tampered")
                risk_score += 30

        # Flag 5: Future dates
        now = datetime.datetime.now()
        for label, dt in [("Creation", props.created), ("Modification", props.modified)]:
            if dt and dt > now:
                flags.append(f"🚨 {label} date is in the future — clock manipulation suspected")
                risk_score += 30

    except Exception as e:
        flags.append(f"❌ Error reading DOCX: {str(e)}")
        risk_score += 20

    risk_score = min(risk_score, 100)

    if risk_score <= 15:
        verdict = "✅ Low Risk — metadata appears consistent"
    elif risk_score <= 40:
        verdict = "⚠️ Medium Risk — anomalies detected, manual review recommended"
    else:
        verdict = "🚨 High Risk — multiple red flags, possible tampering"

    return {
        "file": os.path.basename(filepath),
        "metadata": metadata,
        "flags": flags,
        "risk_score": risk_score,
        "verdict": verdict,
    }


def analyze_file(filepath: str) -> Dict:
    """Route to the correct analyser based on file extension."""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return analyze_pdf_metadata(filepath)
    elif ext == ".docx":
        return analyze_docx_metadata(filepath)
    else:
        return {
            "file": os.path.basename(filepath),
            "error": f"Unsupported file type: {ext}. Supported: .pdf, .docx",
        }
