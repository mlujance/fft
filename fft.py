# ============================================================
# File: binary_analysis_pdf_advanced.py
# Path: ./binary_analysis_pdf_advanced.py
# ============================================================

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch, correlate
import pywt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors


# ============================================================
# Paths
# ============================================================
OUTPUT_DIR = "analysis_report"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
PDF_PATH = os.path.join(OUTPUT_DIR, "binary_analysis_report.pdf")


# ============================================================
# Helpers
# ============================================================
def setup_dirs():
    os.makedirs(IMG_DIR, exist_ok=True)


def save_plot(filename: str):
    path = os.path.join(IMG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def load_bits(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv", ".txt"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        bits = [int(c) for c in text if c in ("0", "1")]
        bits = np.array(bits, dtype=np.uint8)
    else:
        with open(path, "rb") as f:
            raw = f.read()
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))

    if bits.size == 0:
        raise ValueError("No valid bits were found in the input file.")

    return bits


def spectral_entropy(magnitude: np.ndarray) -> float:
    total = np.sum(magnitude)
    if total <= 0:
        return 0.0
    p = magnitude / total
    return float(-np.sum(p * np.log2(p + 1e-12)))


def rolling_binary_entropy(bits: np.ndarray, window: int = 128) -> np.ndarray:
    if len(bits) < window:
        p = np.mean(bits)
        if p in (0, 1):
            return np.array([0.0])
        return np.array([-(p * np.log2(p) + (1 - p) * np.log2(1 - p))])

    out = []
    for i in range(len(bits) - window + 1):
        w = bits[i:i + window]
        p = np.mean(w)
        if p in (0, 1):
            out.append(0.0)
        else:
            out.append(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
    return np.array(out)


def compute_run_lengths(bits: np.ndarray):
    zero_runs = []
    one_runs = []

    current = bits[0]
    run_len = 1

    for b in bits[1:]:
        if b == current:
            run_len += 1
        else:
            if current == 0:
                zero_runs.append(run_len)
            else:
                one_runs.append(run_len)
            current = b
            run_len = 1

    if current == 0:
        zero_runs.append(run_len)
    else:
        one_runs.append(run_len)

    return zero_runs, one_runs


def transition_matrix(bits: np.ndarray) -> np.ndarray:
    mat = np.zeros((2, 2), dtype=int)
    if len(bits) < 2:
        return mat
    for a, b in zip(bits[:-1], bits[1:]):
        mat[a, b] += 1
    return mat


def binary_2d(bits: np.ndarray, width: int = 256) -> np.ndarray:
    rows = len(bits) // width
    if rows == 0:
        return bits.reshape(1, -1)
    return bits[:rows * width].reshape(rows, width)


def mutual_information_binary(bits: np.ndarray, max_lag: int = 64) -> np.ndarray:
    """
    Mutual information I(X_t ; X_{t+lag}) for binary sequence.
    Units: bits.
    """
    mi_vals = []

    for lag in range(1, max_lag + 1):
        x = bits[:-lag]
        y = bits[lag:]

        if len(x) == 0:
            mi_vals.append(0.0)
            continue

        joint = np.zeros((2, 2), dtype=float)
        for a, b in zip(x, y):
            joint[a, b] += 1.0
        joint /= np.sum(joint)

        px = np.sum(joint, axis=1)
        py = np.sum(joint, axis=0)

        mi = 0.0
        for i in range(2):
            for j in range(2):
                if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
        mi_vals.append(mi)

    return np.array(mi_vals)


def block_frequency(bits: np.ndarray, block_size: int = 8):
    n_blocks = len(bits) // block_size
    if n_blocks == 0:
        return {}, block_size

    trimmed = bits[:n_blocks * block_size].reshape(n_blocks, block_size)

    counts = {}
    for row in trimmed:
        key = "".join(map(str, row.tolist()))
        counts[key] = counts.get(key, 0) + 1

    return counts, block_size


def simple_change_score(bits: np.ndarray, window: int = 256) -> np.ndarray:
    """
    Measures how much local bit density changes from one window to the next.
    """
    if len(bits) < 2 * window:
        return np.array([])

    means = []
    for i in range(0, len(bits) - window + 1, window):
        means.append(np.mean(bits[i:i + window]))
    means = np.array(means)

    if len(means) < 2:
        return np.array([])

    return np.abs(np.diff(means))


def walsh_hadamard_transform(signal: np.ndarray):
    """
    Fast Walsh-Hadamard Transform requires power-of-two length.
    We truncate to nearest lower power of two.
    """
    n = len(signal)
    p = 2 ** int(np.floor(np.log2(n)))
    x = signal[:p].astype(float).copy()

    h = 1
    while h < p:
        for i in range(0, p, h * 2):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2

    return x, p


# ============================================================
# Wavelet analysis
# ============================================================
def cwt_analysis(signal: np.ndarray, fs: float = 1.0, wavelet_name: str = "morl"):
    max_scale = min(128, max(8, len(signal) // 8))
    scales = np.arange(1, max_scale + 1)
    coef, freqs = pywt.cwt(signal, scales, wavelet_name, sampling_period=1 / fs)
    return coef, freqs, scales


def dwt_analysis(signal: np.ndarray, wavelet_name: str = "db4", level: int = None):
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(len(signal), wavelet.dec_len)
    if max_level < 1:
        max_level = 1

    if level is None:
        level = min(5, max_level)
    else:
        level = min(level, max_level)

    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    return coeffs, level


def dwt_energy_by_level(coeffs):
    energies = [float(np.sum(np.square(c))) for c in coeffs]
    labels = ["A{}".format(len(coeffs) - 1)] + ["D{}".format(i) for i in range(len(coeffs) - 1, 0, -1)]
    return labels, energies


def wavelet_packet_energy(signal: np.ndarray, wavelet_name: str = "db4", maxlevel: int = 3):
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet_name, mode="symmetric", maxlevel=maxlevel)
    nodes = wp.get_level(maxlevel, order="freq")
    labels = [n.path for n in nodes]
    energies = np.array([np.sum(np.square(n.data)) for n in nodes], dtype=float)
    return labels, energies


# ============================================================
# Plot generation
# ============================================================
def generate_plots(bits: np.ndarray, fs: float, cwt_wavelet: str, dwt_wavelet: str):
    signal = bits.astype(float) - np.mean(bits)

    paths = {}
    metrics = {}

    # --------------------------------------------------------
    # 1) Raw signal
    # --------------------------------------------------------
    n_show = min(500, len(bits))
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(n_show), bits[:n_show], label="Bits (0/1)")
    plt.title("Binary Signal (first 500 samples)")
    plt.xlabel("Sample index")
    plt.ylabel("Bit value")
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["signal"] = save_plot("signal.png")

    # --------------------------------------------------------
    # 2) FFT
    # --------------------------------------------------------
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1 / fs)
    mag = np.abs(fft_vals)
    pos = freqs > 0

    fft_freqs = freqs[pos]
    fft_mag = mag[pos]

    dominant_freq = float(fft_freqs[np.argmax(fft_mag)]) if len(fft_freqs) else 0.0
    spec_entropy = spectral_entropy(fft_mag) if len(fft_mag) else 0.0

    metrics["dominant_frequency"] = dominant_freq
    metrics["spectral_entropy"] = spec_entropy

    plt.figure(figsize=(10, 4))
    plt.plot(fft_freqs, fft_mag, label="FFT magnitude")
    if len(fft_freqs):
        plt.axvline(dominant_freq, linestyle="--", label=f"Dominant frequency = {dominant_freq:.6f}")
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency [cycles/sample]")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["fft"] = save_plot("fft.png")

    # --------------------------------------------------------
    # 3) PSD
    # --------------------------------------------------------
    nperseg = min(256, len(signal))
    psd_f, psd_p = welch(signal, fs=fs, nperseg=nperseg)

    plt.figure(figsize=(10, 4))
    plt.semilogy(psd_f, psd_p, label="Welch PSD")
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency [cycles/sample]")
    plt.ylabel("Power spectral density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["psd"] = save_plot("psd.png")

    # --------------------------------------------------------
    # 4) Autocorrelation
    # --------------------------------------------------------
    corr = correlate(signal, signal, mode="full")
    corr = corr[corr.size // 2:]
    if corr[0] != 0:
        corr = corr / corr[0]

    periodicity = int(np.argmax(corr[1:]) + 1) if len(corr) > 1 else 0
    metrics["periodicity_samples"] = periodicity

    lag_show = min(500, len(corr))
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(lag_show), corr[:lag_show], label="Normalized autocorrelation")
    if periodicity > 0 and periodicity < lag_show:
        plt.axvline(periodicity, linestyle="--", label=f"Detected periodicity = {periodicity}")
    plt.title("Autocorrelation")
    plt.xlabel("Lag [samples]")
    plt.ylabel("Correlation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["autocorr"] = save_plot("autocorr.png")

    # --------------------------------------------------------
    # 5) CWT scalogram
    # --------------------------------------------------------
    cwt_coef, cwt_freqs, scales = cwt_analysis(signal, fs=fs, wavelet_name=cwt_wavelet)

    plt.figure(figsize=(10, 5))
    extent = [0, cwt_coef.shape[1], cwt_freqs[-1], cwt_freqs[0]]
    plt.imshow(np.abs(cwt_coef), aspect="auto", cmap="inferno", extent=extent)
    plt.title(f"CWT Scalogram ({cwt_wavelet})")
    plt.xlabel("Sample index")
    plt.ylabel("Pseudo-frequency [cycles/sample]")
    cbar = plt.colorbar()
    cbar.set_label("Wavelet magnitude")
    paths["cwt_scalogram"] = save_plot("cwt_scalogram.png")

    # --------------------------------------------------------
    # 6) DWT coefficients by level
    # --------------------------------------------------------
    coeffs, level = dwt_analysis(signal, wavelet_name=dwt_wavelet)

    plt.figure(figsize=(10, 2.2 * len(coeffs)))
    for i, c in enumerate(coeffs, start=1):
        ax = plt.subplot(len(coeffs), 1, i)
        ax.plot(c, label=f"Coeff set {i}")
        if i == 1:
            ax.set_title(f"DWT Coefficients by Level ({dwt_wavelet})")
        ax.set_xlabel("Coefficient index")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend()
    paths["dwt_coeffs"] = save_plot("dwt_coeffs.png")

    # --------------------------------------------------------
    # 7) DWT energy by level
    # --------------------------------------------------------
    dwt_labels, dwt_energies = dwt_energy_by_level(coeffs)
    metrics["dwt_energy_labels"] = dwt_labels
    metrics["dwt_energy_values"] = dwt_energies

    plt.figure(figsize=(10, 4))
    plt.bar(dwt_labels, dwt_energies, label="Energy by level")
    plt.title(f"DWT Energy Distribution ({dwt_wavelet})")
    plt.xlabel("Wavelet sub-band")
    plt.ylabel("Energy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    paths["dwt_energy"] = save_plot("dwt_energy.png")

    # --------------------------------------------------------
    # 8) Wavelet packet energy
    # --------------------------------------------------------
    wp_labels, wp_energies = wavelet_packet_energy(signal, wavelet_name=dwt_wavelet, maxlevel=3)
    metrics["wp_labels"] = wp_labels
    metrics["wp_energies"] = wp_energies.tolist()

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(wp_energies)), wp_energies, label="Wavelet packet energy")
    plt.xticks(range(len(wp_labels)), wp_labels, rotation=45)
    plt.title(f"Wavelet Packet Energy Map ({dwt_wavelet}, level 3)")
    plt.xlabel("Wavelet packet node")
    plt.ylabel("Energy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    paths["wavelet_packet_energy"] = save_plot("wavelet_packet_energy.png")

    # --------------------------------------------------------
    # 9) Local entropy
    # --------------------------------------------------------
    local_ent = rolling_binary_entropy(bits, window=min(128, max(16, len(bits) // 20)))
    metrics["local_entropy_mean"] = float(np.mean(local_ent)) if len(local_ent) else 0.0
    metrics["local_entropy_min"] = float(np.min(local_ent)) if len(local_ent) else 0.0
    metrics["local_entropy_max"] = float(np.max(local_ent)) if len(local_ent) else 0.0

    plt.figure(figsize=(10, 4))
    plt.plot(local_ent, label="Local binary entropy")
    plt.title("Local Entropy")
    plt.xlabel("Window position")
    plt.ylabel("Entropy [bits]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["local_entropy"] = save_plot("local_entropy.png")

    # --------------------------------------------------------
    # 10) Mutual information vs lag
    # --------------------------------------------------------
    max_lag = min(128, max(8, len(bits) // 20))
    mi_vals = mutual_information_binary(bits, max_lag=max_lag)
    metrics["mi_max"] = float(np.max(mi_vals)) if len(mi_vals) else 0.0
    metrics["mi_argmax"] = int(np.argmax(mi_vals) + 1) if len(mi_vals) else 0

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, len(mi_vals) + 1), mi_vals, label="Mutual information")
    plt.title("Mutual Information vs Lag")
    plt.xlabel("Lag [samples]")
    plt.ylabel("Mutual information [bits]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["mutual_information"] = save_plot("mutual_information.png")

    # --------------------------------------------------------
    # 11) Runs histogram
    # --------------------------------------------------------
    zero_runs, one_runs = compute_run_lengths(bits)
    metrics["avg_zero_run"] = float(np.mean(zero_runs)) if zero_runs else 0.0
    metrics["avg_one_run"] = float(np.mean(one_runs)) if one_runs else 0.0

    max_run = 1
    if zero_runs:
        max_run = max(max_run, max(zero_runs))
    if one_runs:
        max_run = max(max_run, max(one_runs))

    bins = np.arange(1, max_run + 2) - 0.5

    plt.figure(figsize=(10, 4))
    if zero_runs:
        plt.hist(zero_runs, bins=bins, alpha=0.6, label="0-runs")
    if one_runs:
        plt.hist(one_runs, bins=bins, alpha=0.6, label="1-runs")
    plt.title("Run-Length Distribution")
    plt.xlabel("Run length [samples]")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["runs"] = save_plot("runs.png")

    # --------------------------------------------------------
    # 12) Transition matrix
    # --------------------------------------------------------
    tm = transition_matrix(bits)
    metrics["transition_matrix"] = tm.tolist()

    plt.figure(figsize=(5, 5))
    plt.imshow(tm, cmap="Blues")
    plt.title("Transition Matrix")
    plt.xlabel("Next bit")
    plt.ylabel("Current bit")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(tm[i, j]), ha="center", va="center", color="black")
    cbar = plt.colorbar()
    cbar.set_label("Transition count")
    paths["transition_matrix"] = save_plot("transition_matrix.png")

    # --------------------------------------------------------
    # 13) Binary 2D image
    # --------------------------------------------------------
    img2d = binary_2d(bits, width=256)
    plt.figure(figsize=(10, 5))
    plt.imshow(img2d, aspect="auto", cmap="gray_r")
    plt.title("Binary 2D Visualization")
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    cbar = plt.colorbar()
    cbar.set_label("Bit value")
    paths["binary_2d"] = save_plot("binary_2d.png")

    # --------------------------------------------------------
    # 14) Change score by window
    # --------------------------------------------------------
    change_scores = simple_change_score(bits, window=min(256, max(32, len(bits) // 20)))
    metrics["change_score_max"] = float(np.max(change_scores)) if len(change_scores) else 0.0

    if len(change_scores):
        plt.figure(figsize=(10, 4))
        plt.plot(change_scores, label="Window-to-window density change")
        plt.title("Change Score by Window")
        plt.xlabel("Window transition index")
        plt.ylabel("Absolute change in mean bit value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        paths["change_score"] = save_plot("change_score.png")

    # --------------------------------------------------------
    # 15) Walsh-Hadamard spectrum
    # --------------------------------------------------------
    wht, p = walsh_hadamard_transform(signal)
    wht_mag = np.abs(wht)

    plt.figure(figsize=(10, 4))
    plt.plot(wht_mag, label="Walsh-Hadamard magnitude")
    plt.title("Walsh-Hadamard Spectrum")
    plt.xlabel("Walsh index")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["walsh_hadamard"] = save_plot("walsh_hadamard.png")

    # Global stats
    metrics["mean"] = float(np.mean(bits))
    metrics["variance"] = float(np.var(bits))
    metrics["sampling_frequency"] = float(fs)
    metrics["cwt_wavelet"] = cwt_wavelet
    metrics["dwt_wavelet"] = dwt_wavelet
    metrics["num_bits"] = int(len(bits))

    return metrics, paths


# ============================================================
# Text interpretation
# ============================================================
def interpret_frequency(metrics):
    f = metrics["dominant_frequency"]
    fs = metrics["sampling_frequency"]

    nyquist = fs / 2.0
    if nyquist <= 0:
        return "Sampling frequency is invalid, so frequency interpretation is not reliable."

    ratio = f / nyquist if nyquist > 0 else 0.0

    if ratio > 0.9:
        return (
            f"Dominant frequency = {f:.6f} cycles/sample, very close to the Nyquist limit ({nyquist:.6f}). "
            "This usually indicates rapid alternation, often consistent with short repeating patterns such as 1010-like behavior."
        )
    elif ratio > 0.4:
        return (
            f"Dominant frequency = {f:.6f} cycles/sample. "
            "This suggests medium-to-fast oscillatory structure rather than long homogeneous blocks."
        )
    else:
        return (
            f"Dominant frequency = {f:.6f} cycles/sample. "
            "This suggests slower structure, longer runs, or broader repeating blocks."
        )


def interpret_dwt_energy(metrics):
    labels = metrics["dwt_energy_labels"]
    vals = np.array(metrics["dwt_energy_values"], dtype=float)
    if len(vals) == 0:
        return "DWT energy could not be computed."

    idx = int(np.argmax(vals))
    return (
        f"The largest DWT energy is in sub-band {labels[idx]}. "
        "Higher-energy detail bands indicate stronger fine-scale changes, while higher-energy approximation bands indicate broader low-frequency structure."
    )


def interpret_mutual_information(metrics):
    return (
        f"Maximum mutual information is {metrics['mi_max']:.6f} bits at lag {metrics['mi_argmax']}. "
        "Higher mutual information means stronger dependence between bits separated by that lag."
    )


def interpret_entropy(metrics):
    return (
        f"Local entropy ranges from {metrics['local_entropy_min']:.6f} to {metrics['local_entropy_max']:.6f} bits, "
        f"with mean {metrics['local_entropy_mean']:.6f}. "
        "For binary data, entropy near 1 bit indicates locally balanced and less predictable content; "
        "entropy near 0 indicates strong local regularity."
    )


# ============================================================
# PDF
# ============================================================
def add_section(story, title, body, styles):
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(body, styles["Body"]))
    story.append(Spacer(1, 0.35 * cm))


def add_figure(story, title, image_path, caption, styles, width=16):
    story.append(Paragraph(title, styles["Heading3"]))
    story.append(Spacer(1, 0.1 * cm))
    story.append(Image(image_path, width=width * cm, height=width * 0.58 * cm))
    story.append(Spacer(1, 0.12 * cm))
    story.append(Paragraph(caption, styles["Caption"]))
    story.append(Spacer(1, 0.45 * cm))


def build_pdf(metrics, paths, source_file):
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#222222"),
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Caption",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#444444"),
        spaceAfter=6,
    ))

    story = []

    story.append(Paragraph("Binary Signal Analysis Report", styles["Title"]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph(
        f"Source file: <b>{os.path.basename(source_file)}</b><br/>"
        f"Number of bits: <b>{metrics['num_bits']}</b><br/>"
        f"Sampling frequency: <b>{metrics['sampling_frequency']}</b> samples/unit",
        styles["Body"]
    ))
    story.append(Spacer(1, 0.4 * cm))

    add_section(
        story,
        "Executive Summary",
        (
            f"Mean bit value = {metrics['mean']:.6f}; variance = {metrics['variance']:.6f}; "
            f"dominant frequency = {metrics['dominant_frequency']:.6f} cycles/sample; "
            f"detected autocorrelation periodicity = {metrics['periodicity_samples']} samples. "
            "This report combines spectral, wavelet, structural, and information-theoretic analyses."
        ),
        styles
    )

    add_section(story, "Frequency Interpretation", interpret_frequency(metrics), styles)
    add_section(story, "Entropy Interpretation", interpret_entropy(metrics), styles)
    add_section(story, "Mutual Information Interpretation", interpret_mutual_information(metrics), styles)
    add_section(story, "Wavelet Energy Interpretation", interpret_dwt_energy(metrics), styles)

    add_figure(
        story,
        "1. Raw Binary Signal",
        paths["signal"],
        "Shows the first 500 samples. Y-axis units are bit value (0 or 1). "
        "Long flat segments indicate runs; rapid toggling indicates high local alternation.",
        styles
    )

    add_figure(
        story,
        "2. FFT Spectrum",
        paths["fft"],
        "X-axis: frequency in cycles/sample. Y-axis: magnitude. "
        "Higher peaks indicate stronger global periodic components. "
        "The maximum meaningful frequency is the Nyquist limit, fs/2.",
        styles
    )

    add_figure(
        story,
        "3. Power Spectral Density",
        paths["psd"],
        "X-axis: frequency in cycles/sample. Y-axis: power spectral density. "
        "This shows how signal power is distributed over frequency. "
        "Sharp peaks indicate dominant oscillatory content.",
        styles
    )

    add_figure(
        story,
        "4. Autocorrelation",
        paths["autocorr"],
        "X-axis: lag in samples. Y-axis: normalized correlation. "
        "Peaks away from lag 0 suggest repetition or periodicity. "
        "Values close to 1 indicate strong similarity.",
        styles
    )

    story.append(PageBreak())

    add_figure(
        story,
        "5. CWT Scalogram",
        paths["cwt_scalogram"],
        f"Continuous wavelet transform using '{metrics['cwt_wavelet']}'. "
        "X-axis: sample index. Y-axis: pseudo-frequency in cycles/sample. "
        "Color indicates wavelet magnitude. Bright ridges highlight localized oscillatory structures.",
        styles
    )

    add_figure(
        story,
        "6. DWT Coefficients by Level",
        paths["dwt_coeffs"],
        f"Discrete wavelet transform using '{metrics['dwt_wavelet']}'. "
        "Each subplot corresponds to approximation/detail coefficients at a different scale. "
        "Large-amplitude details indicate changes or edges at that scale.",
        styles
    )

    add_figure(
        story,
        "7. DWT Energy by Level",
        paths["dwt_energy"],
        "Bar chart of wavelet energy per sub-band. "
        "Higher detail-band energy indicates more fine-scale variation; "
        "higher approximation energy indicates broader low-frequency structure.",
        styles
    )

    add_figure(
        story,
        "8. Wavelet Packet Energy",
        paths["wavelet_packet_energy"],
        "Wavelet packet decomposition refines the frequency partition beyond standard DWT. "
        "Each bar represents energy in a narrower sub-band, useful for locating structured activity.",
        styles
    )

    add_figure(
        story,
        "9. Local Entropy",
        paths["local_entropy"],
        "X-axis: window position. Y-axis: entropy in bits. "
        "For binary signals, 0 bits means highly predictable/structured locally; "
        "1 bit means locally balanced and less predictable.",
        styles
    )

    add_figure(
        story,
        "10. Mutual Information vs Lag",
        paths["mutual_information"],
        "X-axis: lag in samples. Y-axis: mutual information in bits. "
        "Higher values indicate stronger dependence between bits separated by that lag.",
        styles
    )

    story.append(PageBreak())

    add_figure(
        story,
        "11. Run-Length Distribution",
        paths["runs"],
        "X-axis: run length in samples. Y-axis: count. "
        "Longer runs indicate extended stable regions; short runs indicate frequent switching.",
        styles
    )

    add_figure(
        story,
        "12. Transition Matrix",
        paths["transition_matrix"],
        "Matrix of counts for 0→0, 0→1, 1→0, and 1→1 transitions. "
        "Asymmetry can indicate bias or serial dependence.",
        styles
    )

    add_figure(
        story,
        "13. Binary 2D Visualization",
        paths["binary_2d"],
        "The bitstream reshaped into an image. "
        "Structured textures, bands, or repeated motifs can reveal hidden regularities not obvious in 1D plots.",
        styles
    )

    if "change_score" in paths:
        add_figure(
            story,
            "14. Change Score by Window",
            paths["change_score"],
            "Measures absolute change in local bit density between neighboring windows. "
            "Large peaks suggest transitions between different regions or regimes in the sequence.",
            styles
        )

    add_figure(
        story,
        "15. Walsh-Hadamard Spectrum",
        paths["walsh_hadamard"],
        "Alternative spectral view especially useful for binary-like signals and Boolean structure. "
        "Large components may indicate structured correlations not captured as clearly in Fourier space.",
        styles
    )

    doc.build(story)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Advanced binary analysis with PDF report output.")
    parser.add_argument("file", help="Input file (.csv/.txt with 0/1 chars, or raw binary)")
    parser.add_argument("--fs", type=float, default=1.0, help="Sampling frequency (default: 1.0)")
    parser.add_argument("--cwt-wavelet", type=str, default="morl", help="CWT wavelet name (default: morl)")
    parser.add_argument("--dwt-wavelet", type=str, default="db4", help="DWT wavelet name (default: db4)")
    args = parser.parse_args()

    setup_dirs()

    bits = load_bits(args.file)
    metrics, paths = generate_plots(bits, fs=args.fs, cwt_wavelet=args.cwt_wavelet, dwt_wavelet=args.dwt_wavelet)
    build_pdf(metrics, paths, source_file=args.file)

    print("Analysis complete.")
    print(f"PDF report: {PDF_PATH}")
    print(f"Images folder: {IMG_DIR}")


if __name__ == "__main__":
    main()