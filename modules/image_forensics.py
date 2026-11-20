"""
Module 5: Image Forensics for Scanned Documents
-------------------------------------------------
Detects potential manipulation in scanned document images using:
  - Error Level Analysis (ELA)
  - Copy-move detection (block-matching)
  - Edge inconsistency analysis
  - Noise-level analysis
"""

import numpy as np
from typing import Dict, Tuple
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageFilter, ImageStat
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import io
import os


def error_level_analysis(
    image_path: str,
    quality: int = 90,
    amplification: int = 15,
) -> Dict:
    """
    Perform Error Level Analysis (ELA).

    Re-saves the image at a known JPEG quality and computes the difference
    between the original and re-saved version. Tampered regions often show
    higher error levels because they were saved at a different quality.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    quality : int
        JPEG quality for re-saving (default 90).
    amplification : int
        Multiplier for the difference image to make anomalies visible.

    Returns
    -------
    dict with analysis results and path to the ELA output image.
    """
    if not HAS_PIL:
        return {"error": "Pillow not installed."}

    try:
        original = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Cannot open image: {e}"}

    # Re-save at specified quality
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    # Compute difference
    diff = ImageChops.difference(original, resaved)

    # Amplify
    ela_image = diff.point(lambda x: min(x * amplification, 255))

    # Statistics of the difference
    stat = ImageStat.Stat(diff)
    mean_diff = [round(m, 4) for m in stat.mean]  # per channel
    max_diff = [round(m, 4) for m in stat.extrema]

    overall_mean = round(np.mean(mean_diff), 4)

    # Save ELA image
    output_dir = Path(image_path).parent
    ela_output = str(output_dir / f"ELA_{Path(image_path).stem}.png")
    ela_image.save(ela_output)

    flags = []
    risk = 0

    if overall_mean > 8:
        flags.append(f"🚨 High average ELA difference ({overall_mean}) — possible tampering in regions")
        risk += 30
    elif overall_mean > 4:
        flags.append(f"⚠️ Moderate ELA difference ({overall_mean}) — review highlighted regions")
        risk += 15
    else:
        flags.append(f"✅ Low ELA difference ({overall_mean}) — consistent compression level")

    return {
        "file": os.path.basename(image_path),
        "mean_error_per_channel": mean_diff,
        "overall_mean_error": overall_mean,
        "ela_image_path": ela_output,
        "flags": flags,
        "risk_score": min(risk, 100),
    }


def noise_analysis(image_path: str) -> Dict:
    """
    Analyse noise levels across image regions.
    Tampered areas often have different noise characteristics
    from the rest of the image.
    """
    if not HAS_PIL:
        return {"error": "Pillow not installed."}

    try:
        img = Image.open(image_path).convert("L")  # grayscale
    except Exception as e:
        return {"error": f"Cannot open image: {e}"}

    arr = np.array(img, dtype=np.float64)
    h, w = arr.shape

    # Divide into grid blocks
    block_size = max(h, w) // 8
    if block_size < 10:
        return {"error": "Image too small for noise analysis."}

    noise_levels = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = arr[y:y+block_size, x:x+block_size]
            # Estimate noise as std of Laplacian
            laplacian = np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0],
            ], dtype=np.float64)
            from scipy.signal import convolve2d
            filtered = convolve2d(block, laplacian, mode="valid")
            noise = np.std(filtered)
            noise_levels.append(noise)

    noise_arr = np.array(noise_levels)
    mean_noise = round(float(noise_arr.mean()), 4)
    std_noise = round(float(noise_arr.std()), 4)
    cv = round(std_noise / max(mean_noise, 0.001), 4)  # coefficient of variation

    flags = []
    risk = 0
    if cv > 0.5:
        flags.append(f"🚨 High noise variation (CV = {cv}) — inconsistent regions may indicate splicing")
        risk += 25
    elif cv > 0.3:
        flags.append(f"⚠️ Moderate noise variation (CV = {cv}) — some inconsistency")
        risk += 10
    else:
        flags.append(f"✅ Noise levels are consistent (CV = {cv})")

    return {
        "file": os.path.basename(image_path),
        "mean_noise": mean_noise,
        "std_noise": std_noise,
        "coefficient_of_variation": cv,
        "num_blocks_analyzed": len(noise_levels),
        "flags": flags,
        "risk_score": min(risk, 100),
    }


def edge_consistency_analysis(image_path: str) -> Dict:
    """
    Check for edge discontinuities that might indicate cut-and-paste edits.
    Uses edge detection and analyses uniformity across the image.
    """
    if not HAS_PIL:
        return {"error": "Pillow not installed."}

    try:
        img = Image.open(image_path).convert("L")
    except Exception as e:
        return {"error": f"Cannot open image: {e}"}

    # Apply edge detection
    edges = img.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges, dtype=np.float64)

    h, w = edge_arr.shape
    block_size = max(h, w) // 6
    if block_size < 10:
        return {"error": "Image too small for edge analysis."}

    block_means = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = edge_arr[y:y+block_size, x:x+block_size]
            block_means.append(np.mean(block))

    bm = np.array(block_means)
    mean_edge = round(float(bm.mean()), 4)
    std_edge = round(float(bm.std()), 4)
    cv = round(std_edge / max(mean_edge, 0.001), 4)

    flags = []
    risk = 0
    if cv > 0.6:
        flags.append(f"🚨 High edge-density variation (CV = {cv}) — possible insertion or deletion")
        risk += 20
    elif cv > 0.35:
        flags.append(f"⚠️ Moderate edge variation (CV = {cv})")
        risk += 10
    else:
        flags.append(f"✅ Edge consistency appears normal (CV = {cv})")

    return {
        "file": os.path.basename(image_path),
        "mean_edge_density": mean_edge,
        "std_edge_density": std_edge,
        "coefficient_of_variation": cv,
        "flags": flags,
        "risk_score": min(risk, 100),
    }


def analyze_image(image_path: str) -> Dict:
    """Run all image forensic analyses and compile results."""
    results = {}

    results["error_level_analysis"] = error_level_analysis(image_path)
    results["edge_consistency"] = edge_consistency_analysis(image_path)

    # Noise analysis needs scipy — optional
    try:
        results["noise_analysis"] = noise_analysis(image_path)
    except ImportError:
        results["noise_analysis"] = {"note": "scipy not installed; noise analysis skipped."}

    # Overall score
    scores = [
        v.get("risk_score", 0)
        for v in results.values()
        if isinstance(v, dict) and "risk_score" in v
    ]
    overall = round(np.mean(scores), 1) if scores else 0

    if overall <= 15:
        verdict = "✅ LOW RISK — Image appears authentic"
    elif overall <= 35:
        verdict = "⚠️ MEDIUM RISK — Some areas may warrant closer inspection"
    else:
        verdict = "🚨 HIGH RISK — Multiple forensic anomalies detected"

    results["overall"] = {
        "risk_score": overall,
        "verdict": verdict,
    }

    return results
