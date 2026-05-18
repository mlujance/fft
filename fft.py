# ============================================================
# File: binary_analysis_pdf_advanced.py
# Path: ./binary_analysis_pdf_advanced.py
# ============================================================
# Professional Binary Pattern Analysis Report
#
# Features:
# - Reads binary sequences from TXT/CSV-like files
# - Cleans input automatically, keeping only 0 and 1 values
# - Computes core binary statistics
# - Detects run-length structure for groups of 0s and 1s
# - Builds readable run-length matrices/tables
# - Computes transition matrix: 0->0, 0->1, 1->0, 1->1
# - Performs FFT spectral analysis
# - Performs PSD analysis using Welch
# - Performs autocorrelation and periodicity estimation
# - Performs wavelet analysis using PyWavelets
# - Generates a polished technical PDF report
#
# Install requirements:
#   pip install numpy matplotlib scipy pywavelets reportlab pandas
#
# Run example:
#   python binary_analysis_pdf_advanced.py --input bits.csv
#
# Optional:
#   python binary_analysis_pdf_advanced.py --input bits.csv --output analysis_report/binary_analysis_report.pdf
# ============================================================

import os
import re
import math
import argparse
import datetime as dt
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch, correlate, find_peaks
import pywt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)


# ============================================================
# Global styling
# ============================================================

DEFAULT_OUTPUT_DIR = "analysis_report"
DEFAULT_IMG_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "figures")
DEFAULT_OUTPUT_PDF = os.path.join(DEFAULT_OUTPUT_DIR, "binary_analysis_report.pdf")

REPORT_TITLE = "Binary Pattern Analysis Report"
REPORT_SUBTITLE = "FFT, Periodicity, Entropy & Run-Length Diagnostics"

PRIMARY = colors.HexColor("#1F2937")
SECONDARY = colors.HexColor("#4B5563")
ACCENT = colors.HexColor("#2563EB")
LIGHT_BG = colors.HexColor("#F3F4F6")
VERY_LIGHT_BG = colors.HexColor("#F9FAFB")
BORDER = colors.HexColor("#D1D5DB")
SUCCESS = colors.HexColor("#047857")
WARNING = colors.HexColor("#B45309")
DANGER = colors.HexColor("#B91C1C")


# ============================================================
# Utility functions
# ============================================================

def ensure_dirs(output_pdf: str) -> str:
    output_dir = os.path.dirname(output_pdf) or DEFAULT_OUTPUT_DIR
    img_dir = os.path.join(output_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    return img_dir


def debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[DEBUG] {message}")


def read_binary_file(path: str, debug: bool = False) -> np.ndarray:
    """
    Reads a file and extracts binary values.

    Supported inputs:
    - Plain text: 010101001
    - CSV-like: 0,1,0,1,1
    - One bit per line
    - Files containing other text: only standalone 0/1 tokens or continuous 0101 groups are extracted
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    debug_print(debug, f"Raw input length: {len(raw)} characters")

    # First try to detect comma/space/newline-separated tokens.
    tokens = re.findall(r"(?<!\d)[01](?!\d)", raw)

    # Also detect continuous binary strings such as 0101010101.
    continuous_groups = re.findall(r"[01]{2,}", raw)

    if continuous_groups:
        joined_groups = "".join(continuous_groups)
        if len(joined_groups) > len(tokens):
            bits = [int(ch) for ch in joined_groups]
        else:
            bits = [int(x) for x in tokens]
    else:
        bits = [int(x) for x in tokens]

    if not bits:
        raise ValueError("No binary values were found. The file must contain 0 and 1 values.")

    arr = np.array(bits, dtype=np.int8)
    debug_print(debug, f"Extracted bits: {len(arr)}")
    debug_print(debug, f"First 32 bits: {''.join(map(str, arr[:32]))}")
    return arr


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def binary_entropy(bits: np.ndarray) -> float:
    n = len(bits)
    if n == 0:
        return 0.0
    p1 = np.mean(bits)
    p0 = 1.0 - p1
    entropy = 0.0
    for p in (p0, p1):
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_float(value: float, digits: int = 4) -> str:
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value:.{digits}f}"


# ============================================================
# Run-length analysis
# ============================================================

def compute_runs(bits: np.ndarray):
    """
    Returns a list of runs:
    [
        {"value": 0 or 1, "start": index, "end": index, "length": length},
        ...
    ]
    """
    if len(bits) == 0:
        return []

    runs = []
    current_value = int(bits[0])
    start = 0

    for i in range(1, len(bits)):
        value = int(bits[i])
        if value != current_value:
            runs.append({
                "value": current_value,
                "start": start,
                "end": i - 1,
                "length": i - start,
            })
            current_value = value
            start = i

    runs.append({
        "value": current_value,
        "start": start,
        "end": len(bits) - 1,
        "length": len(bits) - start,
    })
    return runs


def run_length_summary(runs):
    by_value = {0: [], 1: []}
    for r in runs:
        by_value[r["value"]].append(r["length"])

    summary = {}
    for value in (0, 1):
        lengths = by_value[value]
        if lengths:
            summary[value] = {
                "count": len(lengths),
                "mean": float(np.mean(lengths)),
                "median": float(np.median(lengths)),
                "max": int(np.max(lengths)),
                "min": int(np.min(lengths)),
            }
        else:
            summary[value] = {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "max": 0,
                "min": 0,
            }
    return summary


def build_run_length_matrix(runs, max_exact_length: int = 12):
    """
    Builds a compact table:
      Run length | Count of 0-runs | Count of 1-runs
         1       |       ...       |       ...
         2       |       ...       |       ...
         ...
         12+     |       ...       |       ...
    """
    counters = {0: Counter(), 1: Counter()}

    for r in runs:
        value = r["value"]
        length = r["length"]
        bucket = length if length <= max_exact_length else f"{max_exact_length}+"
        counters[value][bucket] += 1

    rows = [["Run length", "Count of 0-runs", "Count of 1-runs"]]
    for length in range(1, max_exact_length + 1):
        rows.append([
            str(length),
            str(counters[0].get(length, 0)),
            str(counters[1].get(length, 0)),
        ])

    overflow_key = f"{max_exact_length}+"
    overflow_0 = counters[0].get(overflow_key, 0)
    overflow_1 = counters[1].get(overflow_key, 0)
    if overflow_0 or overflow_1:
        rows.append([overflow_key, str(overflow_0), str(overflow_1)])

    return rows


def top_runs_table(runs, top_n: int = 10):
    sorted_runs = sorted(runs, key=lambda r: r["length"], reverse=True)[:top_n]
    rows = [["Rank", "Value", "Start index", "End index", "Length"]]
    for i, r in enumerate(sorted_runs, start=1):
        rows.append([str(i), str(r["value"]), str(r["start"]), str(r["end"]), str(r["length"])])
    return rows


# ============================================================
# Transition analysis
# ============================================================

def transition_matrix(bits: np.ndarray):
    matrix = np.zeros((2, 2), dtype=int)
    if len(bits) < 2:
        return matrix

    for a, b in zip(bits[:-1], bits[1:]):
        matrix[int(a), int(b)] += 1
    return matrix


def transition_matrix_table(matrix: np.ndarray):
    total = matrix.sum()
    rows = [["From / To", "0", "1", "Row total"]]
    for i in (0, 1):
        row_total = matrix[i].sum()
        rows.append([
            str(i),
            f"{matrix[i, 0]} ({format_pct(safe_div(matrix[i, 0], total))})" if total else "0",
            f"{matrix[i, 1]} ({format_pct(safe_div(matrix[i, 1], total))})" if total else "0",
            str(row_total),
        ])
    rows.append(["Column total", str(matrix[:, 0].sum()), str(matrix[:, 1].sum()), str(total)])
    return rows


# ============================================================
# Pattern recurrence analysis
# ============================================================

def count_binary_blocks(bits: np.ndarray, block_size: int = 4, top_n: int = 12):
    if len(bits) < block_size:
        return []
    counter = Counter()
    for i in range(0, len(bits) - block_size + 1):
        block = "".join(map(str, bits[i:i + block_size]))
        counter[block] += 1
    return counter.most_common(top_n)


def top_blocks_table(bits: np.ndarray, block_size: int = 4, top_n: int = 12):
    rows = [["Rank", f"Block size {block_size}", "Occurrences", "Frequency"]]
    total_windows = max(0, len(bits) - block_size + 1)
    for i, (block, count) in enumerate(count_binary_blocks(bits, block_size, top_n), start=1):
        rows.append([str(i), block, str(count), format_pct(safe_div(count, total_windows))])
    if len(rows) == 1:
        rows.append(["-", "N/A", "0", "0.00%"])
    return rows


def recurrence_matrix(bits: np.ndarray, window_size: int = 16, max_windows: int = 80):
    """
    Creates a simple recurrence matrix based on Hamming similarity between binary windows.
    Useful for seeing repeated local structures.
    """
    if len(bits) < window_size:
        return None

    windows = []
    step = max(1, (len(bits) - window_size) // max_windows)
    for i in range(0, len(bits) - window_size + 1, step):
        windows.append(bits[i:i + window_size])
        if len(windows) >= max_windows:
            break

    n = len(windows)
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            mat[i, j] = 1.0 - np.mean(np.abs(windows[i] - windows[j]))
    return mat


# ============================================================
# Frequency and periodicity analysis
# ============================================================

def fft_analysis(bits: np.ndarray):
    """
    Uses centered signal to remove DC component.
    Returns frequency axis, magnitude spectrum and dominant frequency.
    """
    n = len(bits)
    if n < 4:
        return None

    signal = bits.astype(float) - np.mean(bits)
    spectrum = np.fft.rfft(signal)
    magnitude = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n, d=1.0)

    if len(magnitude) > 1:
        # Ignore frequency 0 because signal is centered, but keep robust behavior.
        idx = int(np.argmax(magnitude[1:]) + 1)
        dominant_freq = float(freqs[idx])
        dominant_mag = float(magnitude[idx])
        dominant_period = float(1.0 / dominant_freq) if dominant_freq > 0 else None
    else:
        dominant_freq = None
        dominant_mag = None
        dominant_period = None

    return {
        "freqs": freqs,
        "magnitude": magnitude,
        "dominant_freq": dominant_freq,
        "dominant_mag": dominant_mag,
        "dominant_period": dominant_period,
    }


def psd_welch_analysis(bits: np.ndarray):
    n = len(bits)
    if n < 8:
        return None

    signal = bits.astype(float) - np.mean(bits)
    nperseg = min(256, n)
    freqs, psd = welch(signal, fs=1.0, nperseg=nperseg)
    return {"freqs": freqs, "psd": psd, "nperseg": nperseg}


def autocorrelation_analysis(bits: np.ndarray):
    n = len(bits)
    if n < 4:
        return None

    signal = bits.astype(float) - np.mean(bits)
    corr = correlate(signal, signal, mode="full")
    corr = corr[corr.size // 2:]

    if corr[0] != 0:
        corr = corr / corr[0]

    lags = np.arange(len(corr))

    # Ignore lag 0. Find meaningful positive peaks.
    peaks, properties = find_peaks(corr[1:], height=0.1)
    peaks = peaks + 1

    if len(peaks) > 0:
        best_peak = int(peaks[np.argmax(corr[peaks])])
        best_value = float(corr[best_peak])
    else:
        best_peak = None
        best_value = None

    return {
        "lags": lags,
        "corr": corr,
        "dominant_lag": best_peak,
        "dominant_corr": best_value,
    }


# ============================================================
# Wavelet analysis
# ============================================================

def wavelet_analysis(bits: np.ndarray, wavelet_name: str = "morl"):
    n = len(bits)
    if n < 16:
        return None

    signal = bits.astype(float) - np.mean(bits)
    max_scale = min(64, max(8, n // 2))
    scales = np.arange(1, max_scale + 1)
    coeffs, freqs = pywt.cwt(signal, scales, wavelet_name)
    power = np.abs(coeffs)

    scale_energy = np.mean(power, axis=1)
    best_scale_idx = int(np.argmax(scale_energy))
    best_scale = int(scales[best_scale_idx])

    return {
        "coeffs": coeffs,
        "power": power,
        "scales": scales,
        "freqs": freqs,
        "best_scale": best_scale,
        "best_scale_energy": float(scale_energy[best_scale_idx]),
    }


# ============================================================
# Plot functions
# ============================================================

def save_signal_overview_plot(bits: np.ndarray, img_dir: str) -> str:
    path = os.path.join(img_dir, "signal_overview.png")
    max_points = min(len(bits), 500)

    plt.figure(figsize=(10, 3.2))
    plt.step(np.arange(max_points), bits[:max_points], where="post")
    plt.title("Binary Signal Overview")
    plt.xlabel("Sample index")
    plt.ylabel("Bit value")
    plt.ylim(-0.2, 1.2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_fft_plot(fft_result, img_dir: str) -> str:
    path = os.path.join(img_dir, "fft_spectrum.png")
    freqs = fft_result["freqs"]
    magnitude = fft_result["magnitude"]

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitude)
    plt.title("FFT Magnitude Spectrum")
    plt.xlabel("Frequency [cycles/sample]")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_psd_plot(psd_result, img_dir: str) -> str:
    path = os.path.join(img_dir, "psd_welch.png")
    freqs = psd_result["freqs"]
    psd = psd_result["psd"]

    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, psd + 1e-12)
    plt.title("Power Spectral Density - Welch Method")
    plt.xlabel("Frequency [cycles/sample]")
    plt.ylabel("PSD")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_autocorrelation_plot(ac_result, img_dir: str) -> str:
    path = os.path.join(img_dir, "autocorrelation.png")
    lags = ac_result["lags"]
    corr = ac_result["corr"]
    max_lag = min(len(lags), 500)

    plt.figure(figsize=(10, 4))
    plt.plot(lags[:max_lag], corr[:max_lag])
    plt.title("Autocorrelation")
    plt.xlabel("Lag [samples]")
    plt.ylabel("Normalized correlation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_wavelet_plot(wavelet_result, img_dir: str) -> str:
    path = os.path.join(img_dir, "wavelet_scalogram.png")
    power = wavelet_result["power"]
    scales = wavelet_result["scales"]

    plt.figure(figsize=(10, 4.5))
    plt.imshow(
        power,
        aspect="auto",
        origin="lower",
        extent=[0, power.shape[1], scales[0], scales[-1]],
    )
    plt.title("Wavelet Scalogram")
    plt.xlabel("Sample index")
    plt.ylabel("Scale")
    plt.colorbar(label="Magnitude")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_recurrence_plot(rec_mat, img_dir: str) -> str:
    path = os.path.join(img_dir, "pattern_recurrence_matrix.png")

    plt.figure(figsize=(6.5, 6))
    plt.imshow(rec_mat, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
    plt.title("Pattern Recurrence Matrix")
    plt.xlabel("Window index")
    plt.ylabel("Window index")
    plt.colorbar(label="Hamming similarity")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_run_length_bar_plot(runs, img_dir: str, max_length: int = 20) -> str:
    path = os.path.join(img_dir, "run_length_distribution.png")
    counters = {0: Counter(), 1: Counter()}

    for r in runs:
        length = min(r["length"], max_length)
        counters[r["value"]][length] += 1

    x = np.arange(1, max_length + 1)
    zero_counts = np.array([counters[0].get(i, 0) for i in x])
    one_counts = np.array([counters[1].get(i, 0) for i in x])

    width = 0.38
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, zero_counts, width=width, label="0-runs")
    plt.bar(x + width / 2, one_counts, width=width, label="1-runs")
    plt.title("Run-Length Distribution")
    plt.xlabel(f"Run length, values >= {max_length} grouped at {max_length}")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


# ============================================================
# Interpretation helpers
# ============================================================

def interpret_balance(p1: float) -> str:
    deviation = abs(p1 - 0.5)
    if deviation < 0.03:
        return "The sequence is well balanced between zeros and ones."
    if deviation < 0.10:
        return "The sequence shows a mild bias toward one of the two values."
    return "The sequence shows a strong imbalance between zeros and ones."


def interpret_entropy(entropy: float) -> str:
    if entropy > 0.95:
        return "Entropy is high, suggesting a near-balanced distribution of symbols."
    if entropy > 0.75:
        return "Entropy is moderate, suggesting some structure or symbol bias may exist."
    return "Entropy is low, suggesting strong bias or repetitive structure."


def interpret_periodicity(ac_result, fft_result) -> str:
    parts = []
    if fft_result and fft_result.get("dominant_period"):
        parts.append(
            f"FFT suggests a dominant period of approximately {fft_result['dominant_period']:.2f} samples."
        )
    if ac_result and ac_result.get("dominant_lag"):
        parts.append(
            f"Autocorrelation highlights lag {ac_result['dominant_lag']} as the strongest non-zero repeated structure."
        )
    if not parts:
        return "No strong periodicity could be estimated from the available data."
    return " ".join(parts)


def interpret_runs(run_summary) -> str:
    one_count = run_summary[1]["count"]
    zero_count = run_summary[0]["count"]
    one_max = run_summary[1]["max"]
    zero_max = run_summary[0]["max"]

    if one_count == 0:
        return "No groups of ones were detected. The sequence contains only zeros."
    if zero_count == 0:
        return "No groups of zeros were detected. The sequence contains only ones."

    return (
        f"The sequence contains {one_count} groups of ones and {zero_count} groups of zeros. "
        f"The longest run of ones has length {one_max}, while the longest run of zeros has length {zero_max}."
    )


# ============================================================
# ReportLab helpers
# ============================================================

def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        textColor=PRIMARY,
        spaceAfter=12,
    ))

    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        textColor=SECONDARY,
        spaceAfter=20,
    ))

    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=20,
        textColor=PRIMARY,
        spaceBefore=12,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="SubsectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=PRIMARY,
        spaceBefore=8,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="BodyTextCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.6,
        leading=13,
        textColor=colors.HexColor("#111827"),
        spaceAfter=7,
    ))

    styles.add(ParagraphStyle(
        name="SmallMuted",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=SECONDARY,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="MetricCard",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=PRIMARY,
    ))

    return styles


def styled_table(data, col_widths=None, font_size=8.5, header_bg=PRIMARY):
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#111827")),
        ("BACKGROUND", (0, 1), (-1, -1), VERY_LIGHT_BG),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, VERY_LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def image_block(path: str, caption: str, styles, width_cm: float = 16.0):
    elements = []
    img = Image(path, width=width_cm * cm, height=None)
    img._restrictSize(width_cm * cm, 11 * cm)
    elements.append(img)
    elements.append(Paragraph(caption, styles["SmallMuted"]))
    elements.append(Spacer(1, 0.25 * cm))
    return KeepTogether(elements)


def metric_cards(metrics, styles):
    """
    Creates a two-column metric card table.
    metrics: list of tuples (label, value, interpretation)
    """
    rows = []
    for i in range(0, len(metrics), 2):
        row = []
        for item in metrics[i:i + 2]:
            label, value, note = item
            card = (
                f"<b>{label}</b><br/>"
                f"<font size='13'><b>{value}</b></font><br/>"
                f"<font color='#4B5563'>{note}</font>"
            )
            row.append(Paragraph(card, styles["MetricCard"]))
        if len(row) == 1:
            row.append("")
        rows.append(row)

    table = Table(rows, colWidths=[8.0 * cm, 8.0 * cm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), VERY_LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SECONDARY)
    page_text = f"{REPORT_TITLE}  |  Page {doc.page}"
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, page_text)
    canvas.setStrokeColor(BORDER)
    canvas.line(1.5 * cm, 1.35 * cm, A4[0] - 1.5 * cm, 1.35 * cm)
    canvas.restoreState()


# ============================================================
# Main report generation
# ============================================================

def generate_report(input_path: str, output_pdf: str, debug: bool = False) -> None:
    img_dir = ensure_dirs(output_pdf)
    styles = build_styles()

    bits = read_binary_file(input_path, debug=debug)
    n = len(bits)
    ones = int(np.sum(bits))
    zeros = int(n - ones)
    p1 = safe_div(ones, n)
    p0 = safe_div(zeros, n)
    entropy = binary_entropy(bits)
    bias = p1 - 0.5

    runs = compute_runs(bits)
    run_summary = run_length_summary(runs)
    run_matrix_rows = build_run_length_matrix(runs, max_exact_length=12)
    top_runs_rows = top_runs_table(runs, top_n=10)

    t_matrix = transition_matrix(bits)
    transition_rows = transition_matrix_table(t_matrix)

    fft_result = fft_analysis(bits)
    psd_result = psd_welch_analysis(bits)
    ac_result = autocorrelation_analysis(bits)
    wavelet_result = wavelet_analysis(bits)
    rec_mat = recurrence_matrix(bits, window_size=min(16, max(4, n // 20)), max_windows=80)

    # Save figures
    signal_plot = save_signal_overview_plot(bits, img_dir)
    run_plot = save_run_length_bar_plot(runs, img_dir)
    fft_plot = save_fft_plot(fft_result, img_dir) if fft_result else None
    psd_plot = save_psd_plot(psd_result, img_dir) if psd_result else None
    ac_plot = save_autocorrelation_plot(ac_result, img_dir) if ac_result else None
    wavelet_plot = save_wavelet_plot(wavelet_result, img_dir) if wavelet_result else None
    rec_plot = save_recurrence_plot(rec_mat, img_dir) if rec_mat is not None else None

    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.7 * cm,
        title=REPORT_TITLE,
        author="Binary Analysis Script",
    )

    story = []

    # Cover page
    story.append(Spacer(1, 2.0 * cm))
    story.append(Paragraph(REPORT_TITLE, styles["ReportTitle"]))
    story.append(Paragraph(REPORT_SUBTITLE, styles["ReportSubtitle"]))
    story.append(Spacer(1, 0.7 * cm))

    cover_data = [
        ["Input file", os.path.basename(input_path)],
        ["Generated at", generated_at],
        ["Total samples", f"{n:,}"],
        ["Detected values", "Binary sequence: 0 / 1"],
    ]
    story.append(styled_table(cover_data, col_widths=[4.2 * cm, 11.8 * cm], font_size=9, header_bg=ACCENT))
    story.append(Spacer(1, 1.0 * cm))

    executive_summary = (
        f"This report analyzes a binary sequence containing <b>{n:,}</b> samples. "
        f"The sequence contains <b>{ones:,}</b> ones and <b>{zeros:,}</b> zeros. "
        f"The binary entropy is <b>{entropy:.4f}</b> bits/symbol. "
        f"{interpret_balance(p1)} {interpret_entropy(entropy)} "
        f"{interpret_runs(run_summary)} {interpret_periodicity(ac_result, fft_result)}"
    )
    story.append(Paragraph("Executive Summary", styles["SectionTitle"]))
    story.append(Paragraph(executive_summary, styles["BodyTextCustom"]))
    story.append(PageBreak())

    # Core metrics
    story.append(Paragraph("1. Core Metrics", styles["SectionTitle"]))
    story.append(Paragraph(
        "This section summarizes the most important high-level characteristics of the binary sequence.",
        styles["BodyTextCustom"],
    ))

    metrics = [
        ("Total samples", f"{n:,}", "Number of binary observations analyzed."),
        ("Ones", f"{ones:,} ({format_pct(p1)})", "Share of symbols equal to 1."),
        ("Zeros", f"{zeros:,} ({format_pct(p0)})", "Share of symbols equal to 0."),
        ("Binary entropy", f"{entropy:.4f}", "Maximum is 1.0 for a perfectly balanced binary source."),
        ("Bias vs 50/50", f"{bias:+.4f}", "Positive means more ones; negative means more zeros."),
        ("Total runs", f"{len(runs):,}", "Number of consecutive groups of equal values."),
        ("Groups of ones", f"{run_summary[1]['count']:,}", "Number of separated blocks made of one or more 1 values."),
        ("Groups of zeros", f"{run_summary[0]['count']:,}", "Number of separated blocks made of one or more 0 values."),
    ]
    story.append(metric_cards(metrics, styles))
    story.append(Spacer(1, 0.4 * cm))
    story.append(image_block(
        signal_plot,
        "Figure 1. First samples of the binary sequence. The plot is capped for readability when the input is long.",
        styles,
    ))

    # Run-length structure
    story.append(Paragraph("2. Run-Length Structure", styles["SectionTitle"]))
    story.append(Paragraph(
        "Run-length analysis counts consecutive groups of identical values. "
        "This is often more readable than a raw distribution plot, especially when long runs are rare or unevenly distributed.",
        styles["BodyTextCustom"],
    ))

    run_summary_rows = [
        ["Value", "Groups", "Mean length", "Median length", "Min length", "Max length"],
        [
            "0",
            str(run_summary[0]["count"]),
            format_float(run_summary[0]["mean"], 2),
            format_float(run_summary[0]["median"], 2),
            str(run_summary[0]["min"]),
            str(run_summary[0]["max"]),
        ],
        [
            "1",
            str(run_summary[1]["count"]),
            format_float(run_summary[1]["mean"], 2),
            format_float(run_summary[1]["median"], 2),
            str(run_summary[1]["min"]),
            str(run_summary[1]["max"]),
        ],
    ]
    story.append(Paragraph("Run Summary", styles["SubsectionTitle"]))
    story.append(styled_table(run_summary_rows, col_widths=[2 * cm, 2.6 * cm, 3 * cm, 3 * cm, 2.6 * cm, 2.6 * cm]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Run-Length Matrix", styles["SubsectionTitle"]))
    story.append(styled_table(run_matrix_rows, col_widths=[4.0 * cm, 5.5 * cm, 5.5 * cm]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Longest Runs", styles["SubsectionTitle"]))
    story.append(styled_table(top_runs_rows, col_widths=[1.5 * cm, 2 * cm, 3.2 * cm, 3.2 * cm, 3 * cm]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(image_block(
        run_plot,
        "Figure 2. Run-length distribution for zeros and ones. Very long runs are grouped into the last bucket for readability.",
        styles,
    ))

    # Transitions and recurring blocks
    story.append(Paragraph("3. Transitions & Repeated Blocks", styles["SectionTitle"]))
    story.append(Paragraph(
        "The transition matrix shows how often the sequence moves from one value to another. "
        "Repeated block analysis highlights short binary patterns that appear frequently.",
        styles["BodyTextCustom"],
    ))
    story.append(Paragraph("Transition Matrix", styles["SubsectionTitle"]))
    story.append(styled_table(transition_rows, col_widths=[3.2 * cm, 4.2 * cm, 4.2 * cm, 3.2 * cm]))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Most Frequent 4-Bit Blocks", styles["SubsectionTitle"]))
    story.append(styled_table(top_blocks_table(bits, block_size=4, top_n=12), col_widths=[1.5 * cm, 4 * cm, 4 * cm, 4 * cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Most Frequent 8-Bit Blocks", styles["SubsectionTitle"]))
    story.append(styled_table(top_blocks_table(bits, block_size=8, top_n=12), col_widths=[1.5 * cm, 4 * cm, 4 * cm, 4 * cm]))
    story.append(Spacer(1, 0.4 * cm))

    if rec_plot:
        story.append(image_block(
            rec_plot,
            "Figure 3. Pattern recurrence matrix. Brighter areas indicate higher similarity between local binary windows.",
            styles,
            width_cm=13.0,
        ))

    # Frequency and periodicity
    story.append(Paragraph("4. Frequency & Periodicity Analysis", styles["SectionTitle"]))
    story.append(Paragraph(
        "FFT and autocorrelation are complementary. FFT highlights dominant frequencies, while autocorrelation highlights repeated structures in sample-lag space.",
        styles["BodyTextCustom"],
    ))

    spectral_rows = [["Metric", "Value", "Interpretation"]]
    if fft_result:
        spectral_rows.append([
            "Dominant FFT frequency",
            format_float(fft_result["dominant_freq"], 6),
            "Cycles per sample. Higher values indicate faster alternation.",
        ])
        spectral_rows.append([
            "Estimated FFT period",
            format_float(fft_result["dominant_period"], 2),
            "Approximate samples per repeated cycle.",
        ])
    else:
        spectral_rows.append(["Dominant FFT frequency", "N/A", "Sequence too short for robust FFT analysis."])

    if ac_result:
        spectral_rows.append([
            "Dominant autocorrelation lag",
            str(ac_result["dominant_lag"]) if ac_result["dominant_lag"] else "N/A",
            "Strongest non-zero lag detected from autocorrelation peaks.",
        ])
        spectral_rows.append([
            "Dominant autocorrelation value",
            format_float(ac_result["dominant_corr"], 4),
            "Normalized correlation at the dominant lag.",
        ])

    story.append(styled_table(spectral_rows, col_widths=[5.0 * cm, 4.0 * cm, 7.0 * cm]))
    story.append(Spacer(1, 0.4 * cm))

    if fft_plot:
        story.append(image_block(
            fft_plot,
            "Figure 4. FFT magnitude spectrum after removing the mean/DC component.",
            styles,
        ))
    if psd_plot:
        story.append(image_block(
            psd_plot,
            f"Figure 5. Welch power spectral density. Segment size used: {psd_result['nperseg']} samples.",
            styles,
        ))
    if ac_plot:
        story.append(image_block(
            ac_plot,
            "Figure 6. Normalized autocorrelation. Peaks at non-zero lags suggest repeated structure.",
            styles,
        ))

    # Wavelet analysis
    story.append(Paragraph("5. Wavelet Analysis", styles["SectionTitle"]))
    story.append(Paragraph(
        "Wavelet analysis helps detect structures that vary over the sequence. "
        "Unlike FFT, which summarizes global frequency content, wavelets preserve approximate location information.",
        styles["BodyTextCustom"],
    ))

    if wavelet_result:
        wavelet_rows = [
            ["Metric", "Value", "Interpretation"],
            ["Wavelet", "Morlet", "Good general-purpose continuous wavelet for oscillatory structures."],
            ["Strongest scale", str(wavelet_result["best_scale"]), "Scale with the highest average wavelet magnitude."],
            ["Scale energy", format_float(wavelet_result["best_scale_energy"], 4), "Average magnitude at the strongest scale."],
        ]
        story.append(styled_table(wavelet_rows, col_widths=[4.0 * cm, 4.0 * cm, 8.0 * cm]))
        story.append(Spacer(1, 0.4 * cm))
        story.append(image_block(
            wavelet_plot,
            "Figure 7. Wavelet scalogram. Strong horizontal bands may indicate persistent scale-level structures.",
            styles,
        ))
    else:
        story.append(Paragraph(
            "Wavelet analysis was skipped because the sequence is too short for a meaningful scalogram.",
            styles["BodyTextCustom"],
        ))

    # Conclusions
    story.append(Paragraph("6. Interpretation & Conclusions", styles["SectionTitle"]))
    conclusions = [
        interpret_balance(p1),
        interpret_entropy(entropy),
        interpret_runs(run_summary),
        interpret_periodicity(ac_result, fft_result),
    ]

    if rec_mat is not None:
        mean_rec = float(np.mean(rec_mat))
        if mean_rec > 0.75:
            conclusions.append(
                "The recurrence matrix shows high average similarity between windows, suggesting strong repeated local structure."
            )
        elif mean_rec > 0.55:
            conclusions.append(
                "The recurrence matrix shows moderate similarity between windows, suggesting partial repeated structure."
            )
        else:
            conclusions.append(
                "The recurrence matrix shows low average similarity between windows, suggesting less obvious local repetition."
            )

    for c in conclusions:
        story.append(Paragraph(f"• {c}", styles["BodyTextCustom"]))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "Recommended next step: compare this report against a known baseline sequence. "
        "The most valuable conclusions usually come from comparing entropy, run-lengths, dominant lags and recurrence structure across multiple captures or generated samples.",
        styles["BodyTextCustom"],
    ))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"[OK] Report generated: {output_pdf}")
    print(f"[OK] Figures saved in: {img_dir}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a professional PDF report for binary pattern analysis."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input TXT/CSV file containing binary values.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PDF,
        help=f"Output PDF path. Default: {DEFAULT_OUTPUT_PDF}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_report(args.input, args.output, debug=args.debug)


if __name__ == "__main__":
    main()
