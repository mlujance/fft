# ============================================================
# File: binary_analysis_pdf_compact.py
# Path: ./binary_analysis_pdf_compact.py
# ============================================================

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch, correlate, find_peaks
import pywt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors


# ============================================================
# Theme / Template
# ============================================================
REPORT_THEME = {
    "title": "Binary Signal Analysis Report",
    "subtitle": "Compact multiscale analysis of periodicity, structure and local regimes",
    "author": "Automated Binary Analyzer",
    "primary_color": colors.HexColor("#163A5F"),
    "accent_color": colors.HexColor("#2F6690"),
    "text_color": colors.HexColor("#222222"),
    "muted_color": colors.HexColor("#5C6770"),
    "background_light": colors.HexColor("#F4F7FA"),
    "figure_width_cm": 16.2,
    "base_font": "Helvetica",
    "title_font": "Helvetica-Bold",
    "caption_font": "Helvetica-Oblique",
}

# ============================================================
# Paths
# ============================================================
OUTPUT_DIR = "analysis_report"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
PDF_PATH = os.path.join(OUTPUT_DIR, "binary_analysis_report.pdf")


# ============================================================
# IO / Helpers
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


def safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def spectral_entropy(magnitude: np.ndarray) -> float:
    magnitude = np.asarray(magnitude, dtype=float)
    total = np.sum(magnitude)
    if total <= 0:
        return 0.0
    p = magnitude / total
    return float(-np.sum(p * np.log2(p + 1e-12)))


def spectral_flatness(power: np.ndarray) -> float:
    power = np.asarray(power, dtype=float)
    power = power[power > 0]
    if power.size == 0:
        return 0.0
    gm = np.exp(np.mean(np.log(power + 1e-12)))
    am = np.mean(power)
    return float(gm / (am + 1e-12))


def rolling_mean(bits: np.ndarray, window: int = 128, step: int = 1) -> np.ndarray:
    if len(bits) < window:
        return np.array([np.mean(bits, dtype=float)])
    vals = []
    for i in range(0, len(bits) - window + 1, step):
        vals.append(np.mean(bits[i:i + window], dtype=float))
    return np.asarray(vals, dtype=float)


def rolling_binary_entropy(bits: np.ndarray, window: int = 128, step: int = 1) -> np.ndarray:
    if len(bits) < window:
        p = np.mean(bits)
        if p in (0, 1):
            return np.array([0.0])
        return np.array([-(p * np.log2(p) + (1 - p) * np.log2(1 - p))])

    out = []
    for i in range(0, len(bits) - window + 1, step):
        w = bits[i:i + window]
        p = np.mean(w)
        if p in (0, 1):
            out.append(0.0)
        else:
            out.append(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
    return np.array(out, dtype=float)


def compute_run_lengths(bits: np.ndarray):
    if len(bits) == 0:
        return [], []

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


def transition_probabilities(bits: np.ndarray) -> np.ndarray:
    mat = transition_matrix(bits).astype(float)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


def binary_2d(bits: np.ndarray, width: int = 256) -> np.ndarray:
    rows = len(bits) // width
    if rows == 0:
        return bits.reshape(1, -1)
    return bits[:rows * width].reshape(rows, width)


def mutual_information_binary(bits: np.ndarray, max_lag: int = 64) -> np.ndarray:
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

    return np.array(mi_vals, dtype=float)


def top_autocorr_peaks(corr: np.ndarray, n_peaks: int = 4, min_distance: int = 2):
    if len(corr) < 3:
        return [], []

    peaks, _ = find_peaks(corr[1:], distance=min_distance)
    peaks = peaks + 1
    if peaks.size == 0:
        return [], []

    vals = corr[peaks]
    order = np.argsort(vals)[::-1][:n_peaks]
    return peaks[order].tolist(), vals[order].tolist()


# ============================================================
# Approximate Lempel-Ziv complexity
# ============================================================
def lempel_ziv_complexity(bits: np.ndarray) -> int:
    s = "".join(str(int(b)) for b in bits)
    n = len(s)
    if n == 0:
        return 0
    if n == 1:
        return 1

    i, l, k, k_max, c = 0, 1, 1, 1, 1

    while True:
        if l + k > n or i + k > n:
            c += 1
            break

        if s[i:i + k] == s[l:l + k]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l >= n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1

    return int(c)


def normalized_lz_complexity(bits: np.ndarray) -> float:
    n = len(bits)
    if n < 2:
        return 0.0
    c = lempel_ziv_complexity(bits)
    return float(c * np.log2(n) / n)


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
    max_level = max(1, max_level)

    if level is None:
        level = min(5, max_level)
    else:
        level = min(level, max_level)

    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    return coeffs, level


def dwt_energy_by_level(coeffs):
    energies = [float(np.sum(np.square(c))) for c in coeffs]
    labels = [f"A{len(coeffs) - 1}"] + [f"D{i}" for i in range(len(coeffs) - 1, 0, -1)]
    return labels, energies


def multi_cwt_energy(signal: np.ndarray, fs: float, wavelets):
    out = {}
    for w in wavelets:
        try:
            coef, freqs, scales = cwt_analysis(signal, fs=fs, wavelet_name=w)
            energy = np.sum(np.abs(coef) ** 2, axis=1)
            out[w] = {
                "scales": scales,
                "energy": energy,
                "dominant_scale": int(scales[np.argmax(energy)]),
                "dominant_pseudofreq": float(freqs[np.argmax(energy)]) if len(freqs) else 0.0
            }
        except Exception:
            continue
    return out


# ============================================================
# Matrix Profile (compact implementation)
# ============================================================
def z_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    std = np.std(x)
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def naive_matrix_profile(ts: np.ndarray, m: int, exclusion_ratio: float = 0.5):
    ts = np.asarray(ts, dtype=float)
    n = len(ts)
    k = n - m + 1
    if k <= 2:
        return np.array([]), None, None, None

    profile = np.full(k, np.inf)
    profile_idx = np.full(k, -1, dtype=int)
    exclusion = max(1, int(m * exclusion_ratio))

    subseqs = np.array([z_norm(ts[i:i + m]) for i in range(k)], dtype=float)

    for i in range(k):
        d = np.sqrt(np.sum((subseqs - subseqs[i]) ** 2, axis=1))
        left = max(0, i - exclusion)
        right = min(k, i + exclusion + 1)
        d[left:right] = np.inf

        j = int(np.argmin(d))
        profile[i] = d[j]
        profile_idx[i] = j

    motif_idx = int(np.argmin(profile))
    discord_idx = int(np.argmax(profile))
    return profile, profile_idx, motif_idx, discord_idx


# ============================================================
# Higher-level evidence fusion
# ============================================================
def evaluate_periodicity_evidence(metrics: dict) -> dict:
    score = 0.0
    reasons = []

    dom_freq = metrics.get("dominant_frequency", 0.0)
    fs = metrics.get("sampling_frequency", 1.0)
    nyquist = fs / 2.0 if fs > 0 else 0.0
    periodicity = metrics.get("periodicity_samples", 0)
    mi_max = metrics.get("mi_max", 0.0)
    mi_argmax = metrics.get("mi_argmax", 0)
    autocorr_vals = metrics.get("autocorr_peak_values", [])

    if nyquist > 0 and dom_freq > 0:
        ratio = dom_freq / nyquist
        if ratio > 0.9:
            score += 1.5
            reasons.append("FFT peak is very close to Nyquist, consistent with very fast alternation.")
        elif ratio > 0.45:
            score += 1.0
            reasons.append("FFT shows a substantial non-zero oscillatory component.")

    if periodicity > 0:
        score += 1.0
        reasons.append(f"Autocorrelation identifies a main repetition candidate at lag {periodicity}.")

    if autocorr_vals:
        best_corr = max(autocorr_vals)
        if best_corr > 0.5:
            score += 1.2
            reasons.append("Secondary autocorrelation peaks are strong.")
        elif best_corr > 0.2:
            score += 0.6
            reasons.append("Secondary autocorrelation peaks are moderate.")

    if mi_max > 0.05:
        score += 0.8
        reasons.append(f"Mutual information is clear at lag {mi_argmax}.")
    elif mi_max > 0.02:
        score += 0.3
        reasons.append("Mutual information suggests weak-to-moderate lag dependence.")

    if "matrix_profile_min" in metrics and "matrix_profile_max" in metrics:
        mp_min = metrics["matrix_profile_min"]
        mp_max = metrics["matrix_profile_max"]
        if mp_min < 1.0:
            score += 1.0
            reasons.append("Matrix Profile identifies at least one repeated motif-like subsequence.")
        if mp_max > mp_min * 2.5:
            score += 0.5
            reasons.append("Matrix Profile clearly separates repeated and anomalous local regimes.")

    if score >= 4.0:
        level = "strong"
    elif score >= 2.3:
        level = "moderate"
    elif score >= 1.2:
        level = "weak"
    else:
        level = "inconclusive"

    return {"score": score, "level": level, "reasons": reasons}


def evaluate_structure_evidence(metrics: dict) -> dict:
    score = 0.0
    reasons = []

    flatness = metrics.get("spectral_flatness", 0.0)
    lz_norm = metrics.get("lz_complexity_normalized", 0.0)
    local_ent_mean = metrics.get("local_entropy_mean", 0.0)
    max_run = metrics.get("max_run", 0)

    if flatness < 0.25:
        score += 1.3
        reasons.append("Spectral flatness is low, indicating peaky rather than broadband behavior.")
    elif flatness < 0.5:
        score += 0.6
        reasons.append("Spectral flatness is intermediate, suggesting mixed structure.")

    if lz_norm < 0.6:
        score += 1.3
        reasons.append("Normalized Lempel-Ziv complexity is low, compatible with redundancy or repetition.")
    elif lz_norm < 0.85:
        score += 0.6
        reasons.append("Lempel-Ziv complexity is not maximal, leaving room for structure.")

    if local_ent_mean < 0.75:
        score += 1.0
        reasons.append("Mean local entropy is clearly below 1 bit, so the sequence is not locally uniform.")
    elif local_ent_mean < 0.92:
        score += 0.4
        reasons.append("Mean local entropy is moderately below the ideal balanced limit.")

    if max_run >= 8:
        score += 0.8
        reasons.append("Long runs exist, indicating local persistence.")
    elif max_run >= 5:
        score += 0.3
        reasons.append("Run length provides mild evidence of local persistence.")

    if score >= 4.0:
        level = "strong"
    elif score >= 2.2:
        level = "moderate"
    elif score >= 1.1:
        level = "weak"
    else:
        level = "inconclusive"

    return {"score": score, "level": level, "reasons": reasons}


def evaluate_randomness_like_behavior(metrics: dict) -> dict:
    score = 0.0
    reasons = []

    flatness = metrics.get("spectral_flatness", 0.0)
    lz_norm = metrics.get("lz_complexity_normalized", 0.0)
    local_ent_mean = metrics.get("local_entropy_mean", 0.0)
    mi_max = metrics.get("mi_max", 0.0)

    if flatness > 0.7:
        score += 1.2
        reasons.append("Spectral flatness is high, compatible with broadband behavior.")
    elif flatness > 0.45:
        score += 0.5
        reasons.append("Spectral flatness is moderately high.")

    if lz_norm > 0.9:
        score += 1.2
        reasons.append("Lempel-Ziv complexity is high, compatible with novelty and lower compressibility.")
    elif lz_norm > 0.75:
        score += 0.5
        reasons.append("Lempel-Ziv complexity is moderately high.")

    if local_ent_mean > 0.95:
        score += 1.0
        reasons.append("Local entropy is close to 1 bit, compatible with locally balanced unpredictability.")
    elif local_ent_mean > 0.85:
        score += 0.4
        reasons.append("Local entropy is relatively high.")

    if mi_max < 0.01:
        score += 0.8
        reasons.append("Mutual information is very low, suggesting limited lag dependence.")
    elif mi_max < 0.03:
        score += 0.3
        reasons.append("Mutual information is small.")

    if score >= 3.5:
        level = "strong"
    elif score >= 2.0:
        level = "moderate"
    elif score >= 1.0:
        level = "weak"
    else:
        level = "inconclusive"

    return {"score": score, "level": level, "reasons": reasons}


def generate_final_conclusions(metrics: dict) -> dict:
    periodicity = evaluate_periodicity_evidence(metrics)
    structure = evaluate_structure_evidence(metrics)
    randomness_like = evaluate_randomness_like_behavior(metrics)

    if periodicity["level"] in ("strong", "moderate") and structure["level"] in ("strong", "moderate"):
        headline = (
            "The binary shows meaningful evidence of repeated or structured organization, "
            "and the periodicity indicators are supported by multiple independent analyses."
        )
    elif randomness_like["level"] in ("strong", "moderate") and structure["level"] in ("weak", "inconclusive"):
        headline = (
            "The binary is more compatible with noise-like or weakly structured behavior than with strong repetitive organization, "
            "although this is not a formal randomness certification."
        )
    else:
        headline = (
            "The evidence is mixed: some metrics suggest organization or dependence, "
            "but not strongly enough to claim a single simple regime."
        )

    bullets = [
        f"Periodicity evidence: {periodicity['level']} (score={periodicity['score']:.2f}).",
        *periodicity["reasons"][:3],
        f"Structure evidence: {structure['level']} (score={structure['score']:.2f}).",
        *structure["reasons"][:3],
        f"Noise-like compatibility: {randomness_like['level']} (score={randomness_like['score']:.2f}).",
        *randomness_like["reasons"][:3],
    ]

    caution = (
        "These are inferential conclusions, not definitive labels. "
        "A binary file can mix headers, payloads, compression, encryption, framing and padding, "
        "so local structure does not imply a single generating mechanism."
    )

    return {
        "headline": headline,
        "bullets": bullets,
        "caution": caution,
    }


# ============================================================
# Plot generation
# ============================================================
def generate_plots(bits: np.ndarray, fs: float, cwt_wavelet: str, dwt_wavelet: str):
    signal = bits.astype(float) - np.mean(bits)

    paths = {}
    metrics = {}

    # 1) Raw signal
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

    # 2) FFT
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1 / fs)
    mag = np.abs(fft_vals)
    pos = freqs > 0
    fft_freqs = freqs[pos]
    fft_mag = mag[pos]

    metrics["dominant_frequency"] = float(fft_freqs[np.argmax(fft_mag)]) if len(fft_freqs) else 0.0
    metrics["spectral_entropy"] = spectral_entropy(fft_mag)

    plt.figure(figsize=(10, 4))
    plt.plot(fft_freqs, fft_mag, label="FFT magnitude")
    if len(fft_freqs):
        plt.axvline(metrics["dominant_frequency"], linestyle="--",
                    label=f"Dominant frequency = {metrics['dominant_frequency']:.6f}")
    plt.title("FFT Spectrum")
    plt.xlabel("Frequency [cycles/sample]")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["fft"] = save_plot("fft.png")

    # 3) PSD
    nperseg = min(256, len(signal))
    psd_f, psd_p = welch(signal, fs=fs, nperseg=nperseg)
    metrics["spectral_flatness"] = spectral_flatness(psd_p)

    plt.figure(figsize=(10, 4))
    plt.semilogy(psd_f, psd_p, label="Welch PSD")
    plt.title("Power Spectral Density")
    plt.xlabel("Frequency [cycles/sample]")
    plt.ylabel("Power spectral density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["psd"] = save_plot("psd.png")

    # 4) Autocorrelation
    corr = correlate(signal, signal, mode="full")
    corr = corr[corr.size // 2:]
    if corr[0] != 0:
        corr = corr / corr[0]

    metrics["periodicity_samples"] = int(np.argmax(corr[1:]) + 1) if len(corr) > 1 else 0
    peak_lags, peak_vals = top_autocorr_peaks(corr, n_peaks=4)
    metrics["autocorr_peak_lags"] = peak_lags
    metrics["autocorr_peak_values"] = peak_vals

    lag_show = min(500, len(corr))
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(lag_show), corr[:lag_show], label="Normalized autocorrelation")
    if 0 < metrics["periodicity_samples"] < lag_show:
        plt.axvline(metrics["periodicity_samples"], linestyle="--",
                    label=f"Detected periodicity = {metrics['periodicity_samples']}")
    plt.title("Autocorrelation")
    plt.xlabel("Lag [samples]")
    plt.ylabel("Correlation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["autocorr"] = save_plot("autocorr.png")

    # 5) Mutual information
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

    # 6) Local entropy
    local_window = min(128, max(16, len(bits) // 20))
    local_ent = rolling_binary_entropy(bits, window=local_window, step=1)
    metrics["local_entropy_mean"] = safe_mean(local_ent)
    metrics["local_entropy_min"] = float(np.min(local_ent)) if len(local_ent) else 0.0
    metrics["local_entropy_max"] = float(np.max(local_ent)) if len(local_ent) else 0.0

    plt.figure(figsize=(10, 4))
    plt.plot(local_ent, label="Local entropy")
    plt.title("Local Entropy")
    plt.xlabel("Window position")
    plt.ylabel("Entropy [bits]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["local_entropy"] = save_plot("local_entropy.png")

    # 7) Local bias
    bias_vals = rolling_mean(bits, window=local_window, step=1) - 0.5
    metrics["local_bias_min"] = float(np.min(bias_vals)) if len(bias_vals) else 0.0
    metrics["local_bias_max"] = float(np.max(bias_vals)) if len(bias_vals) else 0.0

    plt.figure(figsize=(10, 4))
    plt.plot(bias_vals, label="Local bias (mean - 0.5)")
    plt.axhline(0.0, linestyle="--", label="Balanced reference")
    plt.title("Local Bit Bias")
    plt.xlabel("Window position")
    plt.ylabel("Bias")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["local_bias"] = save_plot("local_bias.png")

    # 8) Runs
    zero_runs, one_runs = compute_run_lengths(bits)
    all_runs = zero_runs + one_runs
    metrics["avg_zero_run"] = safe_mean(zero_runs)
    metrics["avg_one_run"] = safe_mean(one_runs)
    metrics["max_run"] = int(max(all_runs)) if all_runs else 0

    max_run = max(1, metrics["max_run"])
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

    # 9) Transition probabilities
    tp = transition_probabilities(bits)
    metrics["transition_probabilities"] = tp.tolist()

    plt.figure(figsize=(5, 5))
    plt.imshow(tp, cmap="viridis", vmin=0, vmax=1)
    plt.title("Transition Probability Matrix")
    plt.xlabel("Next bit")
    plt.ylabel("Current bit")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{tp[i, j]:.3f}", ha="center", va="center", color="white")
    cbar = plt.colorbar()
    cbar.set_label("Transition probability")
    paths["transition_probabilities"] = save_plot("transition_probabilities.png")

    # 10) Binary 2D
    img2d = binary_2d(bits, width=256)
    plt.figure(figsize=(10, 5))
    plt.imshow(img2d, aspect="auto", cmap="gray_r")
    plt.title("Binary 2D Visualization")
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    cbar = plt.colorbar()
    cbar.set_label("Bit value")
    paths["binary_2d"] = save_plot("binary_2d.png")

    # 11) CWT main
    cwt_coef, cwt_freqs, scales = cwt_analysis(signal, fs=fs, wavelet_name=cwt_wavelet)
    cwt_energy_scale = np.sum(np.abs(cwt_coef) ** 2, axis=1)
    metrics["cwt_dominant_scale"] = int(scales[np.argmax(cwt_energy_scale)])
    metrics["cwt_dominant_pseudofreq"] = float(cwt_freqs[np.argmax(cwt_energy_scale)]) if len(cwt_freqs) else 0.0

    plt.figure(figsize=(10, 5))
    extent = [0, cwt_coef.shape[1], cwt_freqs[-1], cwt_freqs[0]]
    plt.imshow(np.abs(cwt_coef), aspect="auto", cmap="inferno", extent=extent)
    plt.title(f"CWT Scalogram ({cwt_wavelet})")
    plt.xlabel("Sample index")
    plt.ylabel("Pseudo-frequency [cycles/sample]")
    cbar = plt.colorbar()
    cbar.set_label("Wavelet magnitude")
    paths["cwt_scalogram"] = save_plot("cwt_scalogram.png")

    # 12) CWT comparison
    cwt_compare = multi_cwt_energy(signal, fs=fs, wavelets=["morl", "mexh", "gaus1"])
    metrics["multi_cwt_summary"] = {
        k: {
            "dominant_scale": v["dominant_scale"],
            "dominant_pseudofreq": v["dominant_pseudofreq"]
        } for k, v in cwt_compare.items()
    }

    if cwt_compare:
        plt.figure(figsize=(10, 4))
        for w, d in cwt_compare.items():
            plt.plot(d["scales"], d["energy"], label=w)
        plt.title("CWT Energy by Scale Across Wavelet Families")
        plt.xlabel("Scale")
        plt.ylabel("Energy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        paths["multi_cwt_energy"] = save_plot("multi_cwt_energy.png")

    # 13) DWT energy
    coeffs, _ = dwt_analysis(signal, wavelet_name=dwt_wavelet)
    dwt_labels, dwt_energies = dwt_energy_by_level(coeffs)
    metrics["dwt_energy_labels"] = dwt_labels
    metrics["dwt_energy_values"] = dwt_energies

    plt.figure(figsize=(10, 4))
    plt.bar(dwt_labels, dwt_energies, label=f"{dwt_wavelet} energy")
    plt.title(f"DWT Energy Distribution ({dwt_wavelet})")
    plt.xlabel("Wavelet sub-band")
    plt.ylabel("Energy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    paths["dwt_energy"] = save_plot("dwt_energy.png")

    # 14) LZ complexity
    lz_sample_len = min(len(bits), 50000)
    bits_for_lz = bits[:lz_sample_len]
    metrics["lz_complexity"] = lempel_ziv_complexity(bits_for_lz)
    metrics["lz_complexity_normalized"] = normalized_lz_complexity(bits_for_lz)

    # 15) Matrix Profile on local-density series
    mp_window_bits = min(256, max(64, len(bits) // 200))
    mp_step = max(1, mp_window_bits // 8)
    density_series = rolling_mean(bits, window=mp_window_bits, step=mp_step)

    if len(density_series) >= 200:
        max_density_points = 1000
        if len(density_series) > max_density_points:
            idx = np.linspace(0, len(density_series) - 1, max_density_points).astype(int)
            density_series_mp = density_series[idx]
        else:
            density_series_mp = density_series.copy()

        m = max(20, min(80, len(density_series_mp) // 20))
        profile, _, motif_idx, discord_idx = naive_matrix_profile(density_series_mp, m=m)

        if profile.size > 0:
            metrics["matrix_profile_window"] = int(m)
            metrics["matrix_profile_min"] = float(np.min(profile))
            metrics["matrix_profile_max"] = float(np.max(profile))
            metrics["matrix_profile_motif_index"] = int(motif_idx)
            metrics["matrix_profile_discord_index"] = int(discord_idx)

            plt.figure(figsize=(10, 4))
            plt.plot(profile, label="Matrix profile")
            plt.axvline(motif_idx, linestyle="--", label=f"Motif idx = {motif_idx}")
            plt.axvline(discord_idx, linestyle="--", label=f"Discord idx = {discord_idx}")
            plt.title("Matrix Profile on Local-Density Series")
            plt.xlabel("Subsequence index")
            plt.ylabel("Nearest-neighbor distance")
            plt.grid(True, alpha=0.3)
            plt.legend()
            paths["matrix_profile"] = save_plot("matrix_profile.png")

            plt.figure(figsize=(10, 4))
            plt.plot(density_series_mp, label="Local-density series")
            if 0 <= motif_idx < len(density_series_mp) - m:
                plt.axvspan(motif_idx, motif_idx + m, alpha=0.25, label="Motif")
            if 0 <= discord_idx < len(density_series_mp) - m:
                plt.axvspan(discord_idx, discord_idx + m, alpha=0.25, label="Discord")
            plt.title("Motif and Discord Regions")
            plt.xlabel("Reduced window index")
            plt.ylabel("Local mean bit value")
            plt.grid(True, alpha=0.3)
            plt.legend()
            paths["matrix_profile_regions"] = save_plot("matrix_profile_regions.png")

    # Final global metrics
    metrics["mean"] = float(np.mean(bits))
    metrics["variance"] = float(np.var(bits))
    metrics["sampling_frequency"] = float(fs)
    metrics["cwt_wavelet"] = cwt_wavelet
    metrics["dwt_wavelet"] = dwt_wavelet
    metrics["num_bits"] = int(len(bits))
    metrics["bit_bias"] = abs(metrics["mean"] - 0.5)

    return metrics, paths


# ============================================================
# Interpretation text
# ============================================================
def interpret_frequency(metrics):
    f = metrics["dominant_frequency"]
    fs = metrics["sampling_frequency"]
    nyquist = fs / 2.0

    if nyquist <= 0:
        return "Sampling frequency is invalid, so frequency interpretation is not reliable."

    period = (1.0 / f) if f > 1e-12 else None

    if period is not None:
        base = f"Dominant frequency = {f:.6f} cycles/sample, equivalent to an approximate period of {period:.3f} samples. "
    else:
        base = "No meaningful non-zero dominant oscillatory component was isolated. "

    ratio = f / nyquist if nyquist > 0 else 0.0
    if ratio > 0.9:
        return base + "This is very close to Nyquist, so the sequence is compatible with very rapid alternation."
    if ratio > 0.4:
        return base + "This suggests medium-to-fast oscillatory structure rather than very long homogeneous blocks."
    return base + "This suggests slower large-scale organization or longer stable segments."


def interpret_entropy(metrics):
    return (
        f"Local entropy ranges from {metrics['local_entropy_min']:.6f} to {metrics['local_entropy_max']:.6f} bits, "
        f"with mean {metrics['local_entropy_mean']:.6f}. "
        "For binary data, 1 bit is the local upper limit and indicates a locally balanced distribution."
    )


def interpret_mutual_information(metrics):
    mi_max = metrics["mi_max"]
    lag = metrics["mi_argmax"]

    if mi_max > 0.05:
        strength = "clear"
    elif mi_max > 0.02:
        strength = "moderate"
    elif mi_max > 0.005:
        strength = "weak"
    else:
        strength = "minimal"

    return (
        f"Maximum mutual information is {mi_max:.6f} bits at lag {lag}. "
        f"This is {strength} evidence of lagged dependency."
    )


def interpret_dwt_energy(metrics):
    labels = metrics["dwt_energy_labels"]
    vals = np.array(metrics["dwt_energy_values"], dtype=float)
    if len(vals) == 0:
        return "DWT energy could not be computed."

    idx = int(np.argmax(vals))
    total = np.sum(vals) + 1e-12
    frac = vals[idx] / total
    band = labels[idx]

    if band.startswith("A"):
        tendency = "Energy is concentrated in broader low-frequency structure."
    else:
        tendency = "Energy is concentrated in detail content, compatible with transitions or fine-scale changes."

    return f"The largest DWT energy is in sub-band {band}, representing about {frac:.2%} of total DWT energy. {tendency}"


def interpret_matrix_profile(metrics):
    if "matrix_profile_window" not in metrics:
        return "Matrix Profile was not computed because the reduced local-density series was too short."

    return (
        f"Matrix Profile window = {metrics['matrix_profile_window']}; "
        f"motif index = {metrics['matrix_profile_motif_index']}; "
        f"discord index = {metrics['matrix_profile_discord_index']}; "
        f"profile min = {metrics['matrix_profile_min']:.6f}; "
        f"profile max = {metrics['matrix_profile_max']:.6f}. "
        "Low values indicate repeated local subsequences; high values indicate anomalous local regimes."
    )


# ============================================================
# PDF helpers
# ============================================================
def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CustomBody",
        parent=styles["BodyText"],
        fontName=REPORT_THEME["base_font"],
        fontSize=9.4,
        leading=13,
        textColor=REPORT_THEME["text_color"],
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name="CustomCaption",
        parent=styles["BodyText"],
        fontName=REPORT_THEME["caption_font"],
        fontSize=8.3,
        leading=10.8,
        textColor=REPORT_THEME["muted_color"],
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="CustomTitle",
        parent=styles["Title"],
        fontName=REPORT_THEME["title_font"],
        fontSize=20,
        leading=24,
        textColor=REPORT_THEME["primary_color"],
        spaceAfter=10,
    ))

    styles.add(ParagraphStyle(
        name="CustomSubTitle",
        parent=styles["BodyText"],
        fontName=REPORT_THEME["base_font"],
        fontSize=10.2,
        leading=13,
        textColor=REPORT_THEME["muted_color"],
        spaceAfter=10,
    ))

    return styles


def add_section(story, title, body, styles):
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(body, styles["CustomBody"]))
    story.append(Spacer(1, 0.30 * cm))


def add_bullet_section(story, title, bullets, styles):
    story.append(Paragraph(title, styles["Heading2"]))
    story.append(Spacer(1, 0.15 * cm))
    for b in bullets:
        story.append(Paragraph(f"• {b}", styles["CustomBody"]))
    story.append(Spacer(1, 0.30 * cm))


def add_figure(story, title, image_path, caption, styles, width=None):
    if width is None:
        width = REPORT_THEME["figure_width_cm"]

    story.append(Paragraph(title, styles["Heading3"]))
    story.append(Spacer(1, 0.08 * cm))
    story.append(Image(image_path, width=width * cm, height=width * 0.58 * cm))
    story.append(Spacer(1, 0.10 * cm))
    story.append(Paragraph(caption, styles["CustomCaption"]))
    story.append(Spacer(1, 0.35 * cm))


def make_metrics_table(metrics, styles):
    rows = [
        ["Metric", "Value"],
        ["Bits", f"{metrics['num_bits']}"],
        ["Mean", f"{metrics['mean']:.6f}"],
        ["Variance", f"{metrics['variance']:.6f}"],
        ["Dominant frequency", f"{metrics['dominant_frequency']:.6f} cycles/sample"],
        ["Autocorr periodicity", f"{metrics['periodicity_samples']} samples"],
        ["Mutual information max", f"{metrics['mi_max']:.6f} bits @ lag {metrics['mi_argmax']}"],
        ["Local entropy mean", f"{metrics['local_entropy_mean']:.6f} bits"],
        ["Spectral flatness", f"{metrics['spectral_flatness']:.6f}"],
        ["Normalized LZ complexity", f"{metrics['lz_complexity_normalized']:.6f}"],
        ["Max run length", f"{metrics['max_run']}"],
    ]

    table = Table(rows, colWidths=[6.2 * cm, 9.4 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), REPORT_THEME["primary_color"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), REPORT_THEME["title_font"]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), REPORT_THEME["base_font"]),
        ("BACKGROUND", (0, 1), (-1, -1), REPORT_THEME["background_light"]),
        ("TEXTCOLOR", (0, 1), (-1, -1), REPORT_THEME["text_color"]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4D0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


# ============================================================
# PDF build
# ============================================================
def build_pdf(metrics, paths, source_file):
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm
    )

    styles = build_styles()
    conclusions = generate_final_conclusions(metrics)

    story = []

    # Cover
    story.append(Paragraph(REPORT_THEME["title"], styles["CustomTitle"]))
    story.append(Paragraph(REPORT_THEME["subtitle"], styles["CustomSubTitle"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"<b>Source file:</b> {os.path.basename(source_file)}<br/>"
        f"<b>Author:</b> {REPORT_THEME['author']}<br/>"
        f"<b>Sampling frequency:</b> {metrics['sampling_frequency']} samples/unit<br/>"
        f"<b>Main CWT wavelet:</b> {metrics['cwt_wavelet']}<br/>"
        f"<b>Main DWT wavelet:</b> {metrics['dwt_wavelet']}",
        styles["CustomBody"]
    ))
    story.append(Spacer(1, 0.35 * cm))
    story.append(make_metrics_table(metrics, styles))
    story.append(Spacer(1, 0.5 * cm))

    add_section(story, "Executive Summary", conclusions["headline"], styles)
    add_bullet_section(story, "Evidence Summary", conclusions["bullets"], styles)
    add_section(story, "Interpretation Caution", conclusions["caution"], styles)

    story.append(PageBreak())

    add_section(story, "Frequency Interpretation", interpret_frequency(metrics), styles)
    add_section(story, "Entropy Interpretation", interpret_entropy(metrics), styles)
    add_section(story, "Mutual Information Interpretation", interpret_mutual_information(metrics), styles)
    add_section(story, "Wavelet Interpretation", interpret_dwt_energy(metrics), styles)
    add_section(story, "Matrix Profile Interpretation", interpret_matrix_profile(metrics), styles)

    add_figure(
        story, "1. Raw Binary Signal", paths["signal"],
        "The first 500 samples. Long flats indicate persistence; frequent toggling indicates local alternation.",
        styles
    )
    add_figure(
        story, "2. FFT Spectrum", paths["fft"],
        "Frequency is shown in cycles/sample. Peaks indicate global periodic structure; the upper meaningful limit is Nyquist (fs/2).",
        styles
    )
    add_figure(
        story, "3. Power Spectral Density", paths["psd"],
        "Shows how power is distributed over frequency. A peaky PSD suggests concentrated structure; a flatter PSD suggests broader or more noise-like behavior.",
        styles
    )
    add_figure(
        story, "4. Autocorrelation", paths["autocorr"],
        "Peaks away from lag 0 indicate repeated structure or periodicity candidates.",
        styles
    )

    story.append(PageBreak())

    add_figure(
        story, "5. Mutual Information vs Lag", paths["mutual_information"],
        "Higher values indicate statistical dependence between bits separated by a given lag, even when dependence is not purely linear.",
        styles
    )
    add_figure(
        story, "6. Local Entropy", paths["local_entropy"],
        "For binary data, 1 bit is the upper local limit. Lower values indicate local predictability, bias, persistence or repeated micro-structure.",
        styles
    )
    add_figure(
        story, "7. Local Bit Bias", paths["local_bias"],
        "Shows local excess of ones or zeros. Values above zero indicate more ones; below zero indicate more zeros.",
        styles
    )
    add_figure(
        story, "8. Run-Length Distribution", paths["runs"],
        "Shows how long identical-bit runs persist. Longer runs indicate local stability; short runs indicate frequent switching.",
        styles
    )

    story.append(PageBreak())

    add_figure(
        story, "9. Transition Probability Matrix", paths["transition_probabilities"],
        "Diagonal dominance indicates persistence; large off-diagonal probabilities indicate frequent alternation.",
        styles, width=10.5
    )
    add_figure(
        story, "10. Binary 2D Visualization", paths["binary_2d"],
        "The bitstream reshaped as an image. Repeated textures or bands often reveal hidden regularities not obvious in 1D.",
        styles
    )
    add_figure(
        story, "11. Main CWT Scalogram", paths["cwt_scalogram"],
        "Continuous wavelet transform highlights localized time-frequency structure and transient periodic behavior.",
        styles
    )

    if "multi_cwt_energy" in paths:
        add_figure(
            story, "12. CWT Energy Across Wavelet Families", paths["multi_cwt_energy"],
            "Compares scale-energy profiles for Morlet, Mexican Hat and Gaussian wavelets. Robust structure tends to appear consistently across families.",
            styles
        )

    add_figure(
        story, "13. DWT Energy Distribution", paths["dwt_energy"],
        "Shows how discrete wavelet energy is distributed across approximation and detail bands.",
        styles
    )

    if "matrix_profile" in paths:
        add_figure(
            story, "14. Matrix Profile", paths["matrix_profile"],
            "Low values correspond to repeated motifs; high values correspond to anomalous subsequences.",
            styles
        )

    if "matrix_profile_regions" in paths:
        add_figure(
            story, "15. Motif and Discord Regions", paths["matrix_profile_regions"],
            "Highlights the most repeated and most anomalous local regions on the reduced density series.",
            styles
        )

    doc.build(story)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compact advanced binary analysis with PDF report output."
    )
    parser.add_argument("file", help="Input file (.csv/.txt with 0/1 chars, or raw binary)")
    parser.add_argument("--fs", type=float, default=1.0, help="Sampling frequency (default: 1.0)")
    parser.add_argument("--cwt-wavelet", type=str, default="morl", help="Main CWT wavelet name (default: morl)")
    parser.add_argument("--dwt-wavelet", type=str, default="db4", help="Main DWT wavelet name (default: db4)")
    args = parser.parse_args()

    setup_dirs()

    bits = load_bits(args.file)
    metrics, paths = generate_plots(
        bits,
        fs=args.fs,
        cwt_wavelet=args.cwt_wavelet,
        dwt_wavelet=args.dwt_wavelet
    )
    build_pdf(metrics, paths, source_file=args.file)

    print("Analysis complete.")
    print(f"PDF report: {PDF_PATH}")
    print(f"Images folder: {IMG_DIR}")


if __name__ == "__main__":
    main()