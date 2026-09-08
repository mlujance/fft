"""
Binary Signal Analyzer - standalone analysis backend.

This file contains the complete analysis/report pipeline required by
front_end_22.py. It has no runtime dependency on another project backend.

Responsibilities:
- Load TXT/CSV bit text and raw BIN/DAT-style files.
- Perform spectral, temporal, statistical and multiscale analysis.
- Generate analysis figures.
- Search optional reference files as exact raw-byte sequences.
- Build the PDF report, including the reference-occurrence appendix.

Public API used by front_end_22.py:
    OUTPUT_DIR, IMG_DIR, PDF_PATH
    setup_dirs()
    load_bits()
    generate_plots()
    search_reference_binaries()
    build_original_report_with_appendix()
"""

import os
import math
import mmap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
import pywt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle,
)


REPORT_THEME = {
    "title": "Binary Signal Analysis Report",
    "subtitle": "Quantitative spectral, statistical and multiscale analysis",
    "author": "Binary Signal Analyzer",
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

OUTPUT_DIR = "analysis_report"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
PDF_PATH = os.path.join(OUTPUT_DIR, "binary_analysis_report.pdf")


def setup_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)


def _save_plot(filename):
    path = os.path.join(IMG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def load_bits(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in (".csv", ".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
        bits = np.fromiter(
            (1 if char == "1" else 0 for char in text if char in ("0", "1")),
            dtype=np.uint8,
        )
    else:
        with open(path, "rb") as handle:
            raw = handle.read()
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))

    if bits.size == 0:
        raise ValueError("No valid bits were found in the input file.")

    return bits


def _binary_entropy_from_p(p):
    p = float(p)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def spectral_entropy(power):
    power = np.asarray(power, dtype=float)
    total = float(np.sum(power))
    if total <= 0.0:
        return 0.0
    p = power / total
    return float(-np.sum(p * np.log2(p + 1e-15)))


def spectral_flatness(power):
    power = np.asarray(power, dtype=float)
    power = power[power > 0.0]
    if power.size == 0:
        return 0.0
    geometric = float(np.exp(np.mean(np.log(power + 1e-15))))
    arithmetic = float(np.mean(power))
    return geometric / (arithmetic + 1e-15)


def transition_matrix(bits):
    matrix = np.zeros((2, 2), dtype=np.int64)
    if len(bits) < 2:
        return matrix
    encoded = bits[:-1].astype(np.int8) * 2 + bits[1:].astype(np.int8)
    counts = np.bincount(encoded, minlength=4)
    matrix[0, 0] = counts[0]
    matrix[0, 1] = counts[1]
    matrix[1, 0] = counts[2]
    matrix[1, 1] = counts[3]
    return matrix


def transition_probabilities(bits):
    matrix = transition_matrix(bits).astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def longest_runs_with_positions(bits, top_n=15):
    if len(bits) == 0:
        return []

    changes = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends_exclusive = np.concatenate((changes, [len(bits)]))
    lengths = ends_exclusive - starts
    values = bits[starts]

    order = np.lexsort((starts, -lengths))
    out = []
    for idx in order[:top_n]:
        start = int(starts[idx])
        end_exclusive = int(ends_exclusive[idx])
        out.append({
            "value": int(values[idx]),
            "length": int(lengths[idx]),
            "start": start,
            "end": end_exclusive - 1,
        })
    return out


def run_length_distributions(bits):
    if len(bits) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    changes = np.flatnonzero(bits[1:] != bits[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends_exclusive = np.concatenate((changes, [len(bits)]))
    lengths = ends_exclusive - starts
    values = bits[starts]

    return lengths[values == 0], lengths[values == 1]


def rolling_mean_fast(bits, window=256):
    n = len(bits)
    if n <= window:
        return np.array([float(np.mean(bits))]), np.array([n // 2], dtype=int)

    window = int(max(8, min(window, n)))
    max_points = 5000
    step = max(1, (n - window + 1) // max_points)
    starts = np.arange(0, n - window + 1, step, dtype=np.int64)

    csum = np.concatenate(([0], np.cumsum(bits, dtype=np.int64)))
    means = (csum[starts + window] - csum[starts]) / float(window)
    centers = starts + window // 2
    return means.astype(float), centers


def mutual_information_binary(bits, max_lag=64):
    n = len(bits)
    if n < 2:
        return np.array([], dtype=float)

    max_lag = min(int(max_lag), n - 1)
    out = np.zeros(max_lag, dtype=float)

    for lag in range(1, max_lag + 1):
        a = bits[:-lag].astype(np.int8)
        b = bits[lag:].astype(np.int8)
        encoded = a * 2 + b
        joint = np.bincount(encoded, minlength=4).reshape(2, 2).astype(float)
        joint /= joint.sum()

        px = joint.sum(axis=1)
        py = joint.sum(axis=0)

        mi = 0.0
        for i in range(2):
            for j in range(2):
                value = joint[i, j]
                if value > 0.0 and px[i] > 0.0 and py[j] > 0.0:
                    mi += value * math.log2(value / (px[i] * py[j]))
        out[lag - 1] = mi

    return out


def autocorrelation_fft(signal, max_lag=4096):
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    n = len(x)

    if n == 0:
        return np.array([], dtype=float)
    if np.allclose(x, 0.0):
        return np.ones(1, dtype=float)

    fft_len = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=fft_len)
    corr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_len)[:n]
    corr /= np.arange(n, 0, -1, dtype=float)

    if corr[0] != 0:
        corr /= corr[0]

    return corr[: min(n, int(max_lag) + 1)]


def normalized_lz_complexity(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size < 2:
        return 0.0

    max_len = 200000
    if bits.size > max_len:
        idx = np.linspace(0, bits.size - 1, max_len, dtype=np.int64)
        bits = bits[idx]

    s = "".join("1" if b else "0" for b in bits)
    n = len(s)
    dictionary = set()
    i = 0
    phrases = 0

    while i < n:
        length = 1
        while i + length <= n and s[i:i + length] in dictionary:
            length += 1
        dictionary.add(s[i:i + length])
        phrases += 1
        i += length

    return float(phrases * math.log2(max(n, 2)) / n)


def binary_2d(bits, width=256):
    width = int(max(8, width))
    rows = len(bits) // width
    if rows == 0:
        return bits.reshape(1, -1)
    return bits[: rows * width].reshape(rows, width)


def _matrix_profile_compact(bits, subseq_len=64, max_points=1200):
    x = np.asarray(bits, dtype=float)
    if len(x) < subseq_len * 3:
        return np.array([]), None, None, None, None

    if len(x) > max_points + subseq_len:
        idx = np.linspace(0, len(x) - 1, max_points + subseq_len, dtype=np.int64)
        x = x[idx]
        source_scale = (len(bits) - 1) / max(1, len(x) - 1)
    else:
        source_scale = 1.0

    m = min(subseq_len, max(8, len(x) // 8))
    k = len(x) - m + 1
    if k < 3:
        return np.array([]), None, None, None, source_scale

    shape = (k, m)
    strides = (x.strides[0], x.strides[0])
    subseqs = np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides
    ).copy()
    means = subseqs.mean(axis=1, keepdims=True)
    stds = subseqs.std(axis=1, keepdims=True)
    stds[stds < 1e-12] = 1.0
    z = (subseqs - means) / stds

    profile = np.full(k, np.inf, dtype=float)
    neighbour = np.full(k, -1, dtype=int)
    exclusion = max(1, m // 2)

    for i in range(k):
        d = np.sqrt(np.sum((z - z[i]) ** 2, axis=1))
        left = max(0, i - exclusion)
        right = min(k, i + exclusion + 1)
        d[left:right] = np.inf
        j = int(np.argmin(d))
        profile[i] = float(d[j])
        neighbour[i] = j

    finite = np.isfinite(profile)
    if not np.any(finite):
        return profile, neighbour, None, None, source_scale

    motif = int(np.argmin(profile))
    discord = int(np.argmax(np.where(finite, profile, -np.inf)))
    return profile, neighbour, motif, discord, source_scale


def _cwt_window_summary(signal, fs, wavelet_name):
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    boundaries = np.linspace(0, n, 4, dtype=int)
    labels = ("Beginning", "Middle", "End")
    rows = []
    figures = {}

    for i, label in enumerate(labels):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        window = signal[start:end]

        max_samples = 8192
        if len(window) > max_samples:
            sample_idx = np.linspace(
                0, len(window) - 1, max_samples, dtype=np.int64
            )
            work = window[sample_idx]
        else:
            work = window

        if len(work) < 8:
            rows.append({
                "window": label,
                "start": start,
                "end": max(start, end - 1),
                "mean_energy": 0.0,
                "max_energy": 0.0,
                "dominant_scale": 0,
                "dominant_pseudofreq": 0.0,
            })
            continue

        max_scale = min(96, max(8, len(work) // 16))
        scales = np.arange(1, max_scale + 1)
        coef, freqs = pywt.cwt(
            work,
            scales,
            wavelet_name,
            sampling_period=1.0 / float(fs),
        )
        power = np.abs(coef) ** 2
        scale_energy = np.mean(power, axis=1)
        dom_idx = int(np.argmax(scale_energy))

        rows.append({
            "window": label,
            "start": start,
            "end": end - 1,
            "mean_energy": float(np.mean(power)),
            "max_energy": float(np.max(power)),
            "dominant_scale": int(scales[dom_idx]),
            "dominant_pseudofreq": float(freqs[dom_idx]),
        })

        plt.figure(figsize=(10, 4))
        plt.imshow(
            power,
            aspect="auto",
            origin="lower",
            extent=[start, max(start + 1, end - 1), scales[0], scales[-1]],
        )
        plt.title(f"CWT Scalogram - {label}")
        plt.xlabel("Approximate sample index")
        plt.ylabel("Wavelet scale")
        plt.colorbar(label="Energy")
        figures[label.lower()] = _save_plot(f"cwt_{label.lower()}.png")

    return rows, figures


def _dwt_analysis(signal, wavelet_name):
    signal = np.asarray(signal, dtype=float)
    wavelet = pywt.Wavelet(wavelet_name)
    max_level = pywt.dwt_max_level(len(signal), wavelet.dec_len)

    if max_level < 1:
        return [], [], 0

    level = min(5, max_level)
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    labels = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
    energies = [float(np.sum(np.square(c))) for c in coeffs]
    return labels, energies, level


@dataclass
class ReferenceMatch:
    slot: int
    filename: str
    size: int
    offsets: list
    hex_preview: str = ""
    bit_preview: str = ""

    @property
    def found(self):
        return bool(self.offsets)


def find_all_byte_occurrences(data, pattern):
    if not pattern or len(pattern) > len(data):
        return []

    offsets = []
    position = data.find(pattern)

    while position >= 0:
        offsets.append(int(position))
        position = data.find(pattern, position + 1)

    return offsets


def _preview_reference(raw, max_bytes=64):
    preview = raw[:max_bytes]
    hex_preview = " ".join(f"{byte:02X}" for byte in preview)
    bit_preview = " ".join(f"{byte:08b}" for byte in preview)

    if len(raw) > max_bytes:
        hex_preview += " ..."
        bit_preview += " ..."

    return hex_preview, bit_preview


def search_reference_binaries(main_path, reference_paths):
    refs = [
        (slot, path)
        for slot, path in enumerate(reference_paths, start=1)
        if path
    ]
    if not refs:
        return []

    main_size = os.path.getsize(main_path)
    matches = []

    if main_size == 0:
        for slot, path in refs:
            raw = Path(path).read_bytes()
            hex_preview, bit_preview = _preview_reference(raw)
            matches.append(
                ReferenceMatch(
                    slot=slot,
                    filename=os.path.basename(path),
                    size=len(raw),
                    offsets=[],
                    hex_preview=hex_preview,
                    bit_preview=bit_preview,
                )
            )
        return matches

    with open(main_path, "rb") as main_handle:
        with mmap.mmap(
            main_handle.fileno(),
            length=0,
            access=mmap.ACCESS_READ,
        ) as mapped:
            for slot, path in refs:
                raw = Path(path).read_bytes()
                if not raw:
                    raise ValueError(
                        f"Reference {slot} is empty and cannot be searched."
                    )

                offsets = find_all_byte_occurrences(mapped, raw)
                hex_preview, bit_preview = _preview_reference(raw)

                matches.append(
                    ReferenceMatch(
                        slot=slot,
                        filename=os.path.basename(path),
                        size=len(raw),
                        offsets=offsets,
                        hex_preview=hex_preview,
                        bit_preview=bit_preview,
                    )
                )

    return matches


def generate_plots(bits, fs=1.0, cwt_wavelet="morl", dwt_wavelet="db4"):
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size == 0:
        raise ValueError("Cannot analyze an empty bit sequence.")
    if fs <= 0:
        raise ValueError("Sampling frequency must be greater than zero.")

    setup_dirs()

    n = len(bits)
    x = bits.astype(float)
    centered = x - np.mean(x)
    paths = {}
    metrics = {}

    metrics["num_bits"] = int(n)
    metrics["num_ones"] = int(np.sum(bits))
    metrics["num_zeros"] = int(n - metrics["num_ones"])
    metrics["ones_ratio"] = float(np.mean(bits))
    metrics["zeros_ratio"] = 1.0 - metrics["ones_ratio"]
    metrics["binary_entropy"] = _binary_entropy_from_p(metrics["ones_ratio"])
    metrics["sampling_frequency"] = float(fs)
    metrics["cwt_wavelet"] = str(cwt_wavelet)
    metrics["dwt_wavelet"] = str(dwt_wavelet)

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(n), bits, linewidth=0.7, label="Bits (0/1)")
    plt.title(f"Binary Signal (complete input: {n:,} bits)")
    plt.xlabel("Sample index")
    plt.ylabel("Bit value")
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["signal"] = _save_plot("signal.png")

    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))
    fft_values = np.fft.rfft(centered)
    magnitude = np.abs(fft_values)

    if len(magnitude) > 1:
        positive_mag = magnitude.copy()
        positive_mag[0] = 0.0
        top_count = min(3, len(positive_mag) - 1)
        top_idx = np.argpartition(positive_mag, -top_count)[-top_count:]
        top_idx = top_idx[np.argsort(positive_mag[top_idx])[::-1]]
        top_freqs = [
            {
                "frequency": float(freqs[i]),
                "magnitude": float(magnitude[i]),
                "bin": int(i),
            }
            for i in top_idx
            if magnitude[i] > 0
        ]
    else:
        top_freqs = []

    metrics["top_frequencies"] = top_freqs
    metrics["dominant_frequency"] = (
        top_freqs[0]["frequency"] if top_freqs else 0.0
    )
    metrics["spectral_entropy_fft"] = spectral_entropy(magnitude ** 2)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitude, linewidth=0.9)
    plt.title("FFT Magnitude Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    paths["fft"] = _save_plot("fft.png")

    nperseg = min(4096, max(8, n))
    welch_freqs, psd = welch(
        centered,
        fs=float(fs),
        nperseg=nperseg,
        detrend="constant",
    )
    metrics["spectral_flatness"] = spectral_flatness(psd)
    metrics["spectral_entropy_psd"] = spectral_entropy(psd)

    plt.figure(figsize=(10, 4))
    plt.semilogy(welch_freqs, psd + 1e-18, linewidth=0.9)
    plt.title("Welch Power Spectral Density")
    plt.xlabel("Frequency")
    plt.ylabel("PSD")
    plt.grid(True, alpha=0.3)
    paths["psd"] = _save_plot("psd.png")

    autocorr = autocorrelation_fft(centered, max_lag=min(4096, n - 1))
    peaks = []
    if len(autocorr) > 2:
        peak_idx, _ = find_peaks(autocorr[1:], distance=2)
        peak_idx = peak_idx + 1
        if len(peak_idx):
            order = np.argsort(autocorr[peak_idx])[::-1][:5]
            peaks = [
                {
                    "lag": int(peak_idx[i]),
                    "value": float(autocorr[peak_idx[i]]),
                }
                for i in order
            ]
    metrics["autocorrelation_peaks"] = peaks

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(autocorr)), autocorr, linewidth=0.9)
    plt.title("Normalized Autocorrelation")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlation")
    plt.grid(True, alpha=0.3)
    paths["autocorrelation"] = _save_plot("autocorrelation.png")

    mi = mutual_information_binary(bits, max_lag=min(64, n - 1))
    metrics["mutual_information"] = mi.tolist()
    if len(mi):
        top_mi_idx = np.argsort(mi)[::-1][:5]
        metrics["mi_peaks"] = [
            {"lag": int(i + 1), "value": float(mi[i])}
            for i in top_mi_idx
        ]
    else:
        metrics["mi_peaks"] = []

    plt.figure(figsize=(10, 4))
    if len(mi):
        plt.plot(np.arange(1, len(mi) + 1), mi, linewidth=0.9)
    plt.title("Mutual Information vs Lag")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Mutual information (bits)")
    plt.grid(True, alpha=0.3)
    paths["mutual_information"] = _save_plot("mutual_information.png")

    window = min(max(64, n // 500 if n >= 500 else n), 2048)
    local_mean, centers = rolling_mean_fast(bits, window=max(8, window))
    local_entropy = np.array(
        [_binary_entropy_from_p(value) for value in local_mean],
        dtype=float,
    )
    metrics["local_window"] = int(max(8, window))
    metrics["local_mean_min"] = float(np.min(local_mean))
    metrics["local_mean_max"] = float(np.max(local_mean))
    metrics["local_entropy_min"] = float(np.min(local_entropy))
    metrics["local_entropy_max"] = float(np.max(local_entropy))

    plt.figure(figsize=(10, 4))
    plt.plot(
        centers,
        local_mean,
        linewidth=0.9,
        label="Local fraction of ones",
    )
    plt.plot(
        centers,
        local_entropy,
        linewidth=0.9,
        label="Local binary entropy",
    )
    plt.title(
        f"Local Bias and Entropy (window={metrics['local_window']})"
    )
    plt.xlabel("Sample index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["local_stats"] = _save_plot("local_stats.png")

    zero_runs, one_runs = run_length_distributions(bits)
    longest = longest_runs_with_positions(bits, top_n=15)
    metrics["zero_run_count"] = int(len(zero_runs))
    metrics["one_run_count"] = int(len(one_runs))
    metrics["zero_run_mean"] = (
        float(np.mean(zero_runs)) if len(zero_runs) else 0.0
    )
    metrics["one_run_mean"] = (
        float(np.mean(one_runs)) if len(one_runs) else 0.0
    )
    metrics["max_zero_run"] = (
        int(np.max(zero_runs)) if len(zero_runs) else 0
    )
    metrics["max_one_run"] = (
        int(np.max(one_runs)) if len(one_runs) else 0
    )
    metrics["longest_runs"] = longest

    plt.figure(figsize=(10, 4))
    if len(zero_runs):
        hist0 = np.bincount(
            np.minimum(zero_runs, 255),
            minlength=256,
        )[1:]
        x0 = np.arange(1, len(hist0) + 1)
        mask0 = hist0 > 0
        plt.plot(
            x0[mask0],
            hist0[mask0],
            linewidth=1.8,
            label="0-runs",
        )
    if len(one_runs):
        hist1 = np.bincount(
            np.minimum(one_runs, 255),
            minlength=256,
        )[1:]
        x1 = np.arange(1, len(hist1) + 1)
        mask1 = hist1 > 0
        plt.plot(
            x1[mask1],
            hist1[mask1],
            linewidth=1.8,
            label="1-runs",
        )
    plt.title("Run-Length Distribution")
    plt.xlabel("Run length (255 includes longer runs)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    paths["runs"] = _save_plot("run_lengths.png")

    counts = transition_matrix(bits)
    probs = transition_probabilities(bits)
    metrics["transition_counts"] = counts.tolist()
    metrics["transition_probabilities"] = probs.tolist()

    plt.figure(figsize=(6, 5))
    plt.imshow(probs, vmin=0.0, vmax=1.0)
    plt.xticks([0, 1], ["Next 0", "Next 1"])
    plt.yticks([0, 1], ["Current 0", "Current 1"])
    plt.title("Transition Probability Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{probs[i, j]:.4f}",
                ha="center",
                va="center",
            )
    plt.colorbar(label="Probability")
    paths["transitions"] = _save_plot("transitions.png")

    image = binary_2d(bits, width=256)
    metrics["binary_2d_shape"] = tuple(int(v) for v in image.shape)

    plt.figure(figsize=(10, 5))
    plt.imshow(image, aspect="auto", interpolation="nearest")
    plt.title("Binary 2D Representation (row width = 256 bits)")
    plt.xlabel("Bit position within row")
    plt.ylabel("Row")
    paths["binary_2d"] = _save_plot("binary_2d.png")

    cwt_rows, cwt_figures = _cwt_window_summary(
        centered,
        fs=float(fs),
        wavelet_name=str(cwt_wavelet),
    )
    metrics["cwt_windows"] = cwt_rows
    for key, value in cwt_figures.items():
        paths[f"cwt_{key}"] = value

    dwt_labels, dwt_energies, dwt_level = _dwt_analysis(
        centered,
        str(dwt_wavelet),
    )
    metrics["dwt_labels"] = dwt_labels
    metrics["dwt_energies"] = dwt_energies
    metrics["dwt_level"] = int(dwt_level)

    plt.figure(figsize=(10, 4))
    if dwt_labels:
        plt.bar(dwt_labels, dwt_energies)
    plt.title(f"DWT Energy by Level ({dwt_wavelet})")
    plt.xlabel("Coefficient band")
    plt.ylabel("Energy")
    plt.grid(True, axis="y", alpha=0.3)
    paths["dwt"] = _save_plot("dwt_energy.png")

    metrics["normalized_lz_complexity"] = normalized_lz_complexity(bits)

    profile, neighbours, motif, discord, source_scale = (
        _matrix_profile_compact(bits)
    )
    metrics["matrix_profile"] = profile.tolist() if len(profile) else []
    metrics["matrix_profile_motif_index"] = (
        int(round(motif * source_scale))
        if motif is not None and source_scale is not None
        else None
    )
    metrics["matrix_profile_discord_index"] = (
        int(round(discord * source_scale))
        if discord is not None and source_scale is not None
        else None
    )

    plt.figure(figsize=(10, 4))
    if len(profile):
        source_positions = (
            np.arange(len(profile), dtype=float) * float(source_scale)
        )
        plt.plot(source_positions, profile, linewidth=0.9)
    plt.title("Compact Matrix Profile")
    plt.xlabel("Approximate source sample index")
    plt.ylabel("Nearest-neighbour distance")
    plt.grid(True, alpha=0.3)
    paths["matrix_profile"] = _save_plot("matrix_profile.png")

    return metrics, paths


def _safe_text(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=REPORT_THEME["title_font"],
            fontSize=22,
            leading=25,
            textColor=REPORT_THEME["primary_color"],
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName=REPORT_THEME["base_font"],
            fontSize=10,
            leading=14,
            textColor=REPORT_THEME["muted_color"],
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName=REPORT_THEME["title_font"],
            fontSize=13,
            leading=16,
            textColor=REPORT_THEME["primary_color"],
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subsection",
            parent=styles["Heading3"],
            fontName=REPORT_THEME["title_font"],
            fontSize=10.5,
            leading=13,
            textColor=REPORT_THEME["accent_color"],
            spaceBefore=5,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodySmall",
            parent=styles["BodyText"],
            fontName=REPORT_THEME["base_font"],
            fontSize=8.5,
            leading=11.5,
            textColor=REPORT_THEME["text_color"],
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontName=REPORT_THEME["caption_font"],
            fontSize=7.5,
            leading=9.5,
            textColor=REPORT_THEME["muted_color"],
            alignment=1,
            spaceBefore=2,
            spaceAfter=6,
        )
    )
    return styles


def _report_table(rows, widths=None, font_size=8):
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    REPORT_THEME["primary_color"],
                ),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.HexColor("#BCC7D1"),
                ),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, REPORT_THEME["background_light"]],
                ),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _figure_flowable(path, caption, styles, width_cm=None):
    if not path or not os.path.exists(path):
        return []

    width_cm = width_cm or REPORT_THEME["figure_width_cm"]
    image = Image(path)
    aspect = image.imageHeight / max(1.0, image.imageWidth)
    image.drawWidth = width_cm * cm
    image.drawHeight = min(11.5 * cm, width_cm * cm * aspect)

    return [
        image,
        Paragraph(_safe_text(caption), styles["Caption"]),
    ]


def _page_header_footer(canvas, document):
    canvas.saveState()

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(REPORT_THEME["muted_color"])
    canvas.drawString(
        1.6 * cm,
        A4[1] - 1.0 * cm,
        "Binary Signal Analyzer",
    )
    canvas.drawRightString(
        A4[0] - 1.6 * cm,
        A4[1] - 1.0 * cm,
        REPORT_THEME["title"],
    )

    canvas.setStrokeColor(colors.HexColor("#D0D7DE"))
    canvas.line(
        1.6 * cm,
        A4[1] - 1.15 * cm,
        A4[0] - 1.6 * cm,
        A4[1] - 1.15 * cm,
    )

    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(
        1.6 * cm,
        0.75 * cm,
        datetime.now().strftime("Issued %Y-%m-%d %H:%M"),
    )
    canvas.drawRightString(
        A4[0] - 1.6 * cm,
        0.75 * cm,
        f"Page {document.page}",
    )

    canvas.restoreState()


def _append_reference_section(story, matches, paths, styles):
    if not matches:
        return

    story.append(PageBreak())
    story.append(
        Paragraph("Reference Binary Occurrences", styles["Section"])
    )
    story.append(
        Paragraph(
            "Optional references are searched as complete exact raw-byte "
            "sequences inside the primary file. Offsets below are zero-based "
            "byte offsets; end offsets are inclusive. Overlapping matches are "
            "retained.",
            styles["BodySmall"],
        )
    )

    summary_rows = [
        ["Reference", "Size (bytes)", "Found", "Occurrences"]
    ]
    for match in matches:
        summary_rows.append(
            [
                f"Reference {match.slot}: {_safe_text(match.filename)}",
                f"{match.size:,}",
                "Yes" if match.found else "No",
                f"{len(match.offsets):,}",
            ]
        )

    story.append(
        _report_table(
            summary_rows,
            widths=[7.2 * cm, 2.5 * cm, 2.0 * cm, 2.8 * cm],
            font_size=7.8,
        )
    )
    story.append(Spacer(1, 0.2 * cm))

    for match in matches:
        story.append(
            Paragraph(
                f"Reference {match.slot}: {_safe_text(match.filename)}",
                styles["Subsection"],
            )
        )
        story.append(
            Paragraph(
                f"<b>Reference size:</b> {match.size:,} bytes &nbsp;&nbsp; "
                f"<b>Occurrences:</b> {len(match.offsets):,}",
                styles["BodySmall"],
            )
        )

        if match.hex_preview:
            story.append(
                Paragraph(
                    f"<b>Hex preview:</b> "
                    f"{_safe_text(match.hex_preview)}",
                    styles["BodySmall"],
                )
            )
        if match.bit_preview:
            story.append(
                Paragraph(
                    f"<b>Bit preview:</b> "
                    f"{_safe_text(match.bit_preview)}",
                    styles["BodySmall"],
                )
            )

        if match.offsets:
            rows = [
                ["#", "Start byte", "End byte", "Start hex", "End hex"]
            ]
            for number, start in enumerate(match.offsets, start=1):
                end = start + match.size - 1
                rows.append(
                    [
                        str(number),
                        f"{start:,}",
                        f"{end:,}",
                        f"0x{start:X}",
                        f"0x{end:X}",
                    ]
                )
            story.append(
                _report_table(
                    rows,
                    widths=[
                        1.2 * cm,
                        3.0 * cm,
                        3.0 * cm,
                        3.2 * cm,
                        3.2 * cm,
                    ],
                    font_size=7.5,
                )
            )
        else:
            story.append(
                Paragraph(
                    "No complete exact occurrence was found.",
                    styles["BodySmall"],
                )
            )

        story.append(Spacer(1, 0.15 * cm))

    if paths.get("signal"):
        story.extend(
            _figure_flowable(
                paths["signal"],
                "Complete binary signal of the primary input.",
                styles,
            )
        )
    if paths.get("binary_2d"):
        story.extend(
            _figure_flowable(
                paths["binary_2d"],
                "Binary 2D representation of the primary input.",
                styles,
            )
        )


def build_original_report_with_appendix(
    metrics,
    paths,
    source_file,
    matches,
):
    setup_dirs()
    styles = _styles()

    document = SimpleDocTemplate(
        PDF_PATH,
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.35 * cm,
        title=REPORT_THEME["title"],
        author=REPORT_THEME["author"],
    )

    story = []

    story.append(
        Paragraph(REPORT_THEME["title"], styles["ReportTitle"])
    )
    story.append(
        Paragraph(REPORT_THEME["subtitle"], styles["ReportSubtitle"])
    )
    story.append(
        Paragraph(
            f"<b>Source file:</b> "
            f"{_safe_text(os.path.basename(source_file))}<br/>"
            f"<b>Generated:</b> "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["BodySmall"],
        )
    )
    story.append(Spacer(1, 0.15 * cm))

    story.append(
        Paragraph("1. General Parameters", styles["Section"])
    )
    general_rows = [
        ["Parameter", "Value"],
        ["Analyzed bits", f"{metrics['num_bits']:,}"],
        [
            "Zeros",
            f"{metrics['num_zeros']:,} "
            f"({metrics['zeros_ratio'] * 100:.6f}%)",
        ],
        [
            "Ones",
            f"{metrics['num_ones']:,} "
            f"({metrics['ones_ratio'] * 100:.6f}%)",
        ],
        [
            "Binary entropy",
            f"{metrics['binary_entropy']:.8f} bits/symbol",
        ],
        [
            "Sampling frequency",
            f"{metrics['sampling_frequency']:.12g}",
        ],
        ["CWT wavelet", _safe_text(metrics["cwt_wavelet"])],
        ["DWT wavelet", _safe_text(metrics["dwt_wavelet"])],
    ]
    story.append(
        _report_table(
            general_rows,
            widths=[6.2 * cm, 9.0 * cm],
        )
    )
    story.extend(
        _figure_flowable(
            paths.get("signal"),
            "Complete binary sequence. The horizontal axis covers "
            "the entire loaded input.",
            styles,
        )
    )

    story.append(
        Paragraph("2. Frequency-Domain Analysis", styles["Section"])
    )
    top_rows = [["Rank", "Frequency", "FFT magnitude", "FFT bin"]]
    for rank, item in enumerate(
        metrics.get("top_frequencies", []),
        start=1,
    ):
        top_rows.append(
            [
                str(rank),
                f"{item['frequency']:.12g}",
                f"{item['magnitude']:.6g}",
                str(item["bin"]),
            ]
        )
    if len(top_rows) == 1:
        top_rows.append(["-", "-", "-", "-"])

    story.append(
        _report_table(
            top_rows,
            widths=[1.3 * cm, 4.5 * cm, 4.5 * cm, 2.2 * cm],
        )
    )
    spectral_rows = [
        ["Metric", "Value"],
        [
            "FFT spectral entropy",
            f"{metrics['spectral_entropy_fft']:.8f}",
        ],
        [
            "PSD spectral entropy",
            f"{metrics['spectral_entropy_psd']:.8f}",
        ],
        [
            "Spectral flatness",
            f"{metrics['spectral_flatness']:.8f}",
        ],
    ]
    story.append(Spacer(1, 0.12 * cm))
    story.append(
        _report_table(
            spectral_rows,
            widths=[7.0 * cm, 6.5 * cm],
        )
    )
    story.extend(
        _figure_flowable(
            paths.get("fft"),
            "FFT magnitude spectrum of the mean-centered binary sequence.",
            styles,
        )
    )
    story.extend(
        _figure_flowable(
            paths.get("psd"),
            "Welch power spectral density estimate.",
            styles,
        )
    )

    story.append(
        Paragraph("3. Temporal Dependence", styles["Section"])
    )
    ac_rows = [
        ["Autocorrelation rank", "Lag", "Normalized value"]
    ]
    for rank, peak in enumerate(
        metrics.get("autocorrelation_peaks", []),
        start=1,
    ):
        ac_rows.append(
            [
                str(rank),
                str(peak["lag"]),
                f"{peak['value']:.8f}",
            ]
        )
    if len(ac_rows) == 1:
        ac_rows.append(["-", "-", "-"])
    story.append(
        _report_table(
            ac_rows,
            widths=[4.8 * cm, 3.5 * cm, 5.2 * cm],
        )
    )

    mi_rows = [
        ["Mutual-information rank", "Lag", "MI (bits)"]
    ]
    for rank, peak in enumerate(
        metrics.get("mi_peaks", []),
        start=1,
    ):
        mi_rows.append(
            [
                str(rank),
                str(peak["lag"]),
                f"{peak['value']:.8f}",
            ]
        )
    if len(mi_rows) == 1:
        mi_rows.append(["-", "-", "-"])
    story.append(Spacer(1, 0.12 * cm))
    story.append(
        _report_table(
            mi_rows,
            widths=[4.8 * cm, 3.5 * cm, 5.2 * cm],
        )
    )
    story.extend(
        _figure_flowable(
            paths.get("autocorrelation"),
            "Normalized autocorrelation versus lag.",
            styles,
        )
    )
    story.extend(
        _figure_flowable(
            paths.get("mutual_information"),
            "Binary mutual information versus lag.",
            styles,
        )
    )

    story.append(
        Paragraph("4. Local Statistics", styles["Section"])
    )
    local_rows = [
        ["Parameter", "Value"],
        [
            "Rolling window",
            f"{metrics['local_window']:,} samples",
        ],
        [
            "Local fraction of ones - minimum",
            f"{metrics['local_mean_min']:.8f}",
        ],
        [
            "Local fraction of ones - maximum",
            f"{metrics['local_mean_max']:.8f}",
        ],
        [
            "Local binary entropy - minimum",
            f"{metrics['local_entropy_min']:.8f}",
        ],
        [
            "Local binary entropy - maximum",
            f"{metrics['local_entropy_max']:.8f}",
        ],
    ]
    story.append(
        _report_table(
            local_rows,
            widths=[8.2 * cm, 5.3 * cm],
        )
    )
    story.extend(
        _figure_flowable(
            paths.get("local_stats"),
            "Rolling local bias and binary entropy.",
            styles,
        )
    )

    story.append(
        Paragraph("5. Run-Length Analysis", styles["Section"])
    )
    run_rows = [
        ["Metric", "0-runs", "1-runs"],
        [
            "Run count",
            f"{metrics['zero_run_count']:,}",
            f"{metrics['one_run_count']:,}",
        ],
        [
            "Mean run length",
            f"{metrics['zero_run_mean']:.6f}",
            f"{metrics['one_run_mean']:.6f}",
        ],
        [
            "Maximum run length",
            str(metrics["max_zero_run"]),
            str(metrics["max_one_run"]),
        ],
    ]
    story.append(
        _report_table(
            run_rows,
            widths=[6.0 * cm, 3.8 * cm, 3.8 * cm],
        )
    )
    story.extend(
        _figure_flowable(
            paths.get("runs"),
            "Run-length distribution. Long runs above 255 are "
            "accumulated at 255 in the plot only.",
            styles,
        )
    )

    longest_rows = [["Rank", "Bit", "Length", "Start", "End"]]
    for rank, run in enumerate(
        metrics.get("longest_runs", []),
        start=1,
    ):
        longest_rows.append(
            [
                str(rank),
                str(run["value"]),
                f"{run['length']:,}",
                f"{run['start']:,}",
                f"{run['end']:,}",
            ]
        )
    story.append(
        Paragraph("15 Longest Runs", styles["Subsection"])
    )
    story.append(
        _report_table(
            longest_rows,
            widths=[
                1.5 * cm,
                1.5 * cm,
                3.0 * cm,
                3.6 * cm,
                3.6 * cm,
            ],
            font_size=7.8,
        )
    )

    story.append(
        Paragraph("6. Transition Statistics", styles["Section"])
    )
    counts = np.asarray(
        metrics["transition_counts"],
        dtype=int,
    )
    probs = np.asarray(
        metrics["transition_probabilities"],
        dtype=float,
    )
    transition_rows = [
        ["Current -> Next", "0", "1"],
        [
            "0 count",
            f"{counts[0,0]:,}",
            f"{counts[0,1]:,}",
        ],
        [
            "1 count",
            f"{counts[1,0]:,}",
            f"{counts[1,1]:,}",
        ],
        [
            "P(next|0)",
            f"{probs[0,0]:.8f}",
            f"{probs[0,1]:.8f}",
        ],
        [
            "P(next|1)",
            f"{probs[1,0]:.8f}",
            f"{probs[1,1]:.8f}",
        ],
    ]
    story.append(
        _report_table(
            transition_rows,
            widths=[5.0 * cm, 4.2 * cm, 4.2 * cm],
        )
    )
    story.extend(
        _figure_flowable(
            paths.get("transitions"),
            "Conditional transition-probability matrix.",
            styles,
            width_cm=11.0,
        )
    )

    story.append(
        Paragraph("7. Binary 2D Representation", styles["Section"])
    )
    shape = metrics.get("binary_2d_shape", ("-", "-"))
    story.append(
        Paragraph(
            f"The one-dimensional bitstream was reshaped to a "
            f"two-dimensional representation of approximately "
            f"{_safe_text(shape)} for visual inspection. This "
            f"transformation is descriptive and does not imply an "
            f"intrinsic 2D data model.",
            styles["BodySmall"],
        )
    )
    story.extend(
        _figure_flowable(
            paths.get("binary_2d"),
            "Binary 2D representation using 256 bits per row "
            "where possible.",
            styles,
        )
    )

    story.append(
        Paragraph("8. Continuous Wavelet Transform", styles["Section"])
    )
    cwt_rows = [[
        "Window",
        "Start",
        "End",
        "Mean energy",
        "Max energy",
        "Dominant scale",
        "Pseudo-frequency",
    ]]
    for item in metrics.get("cwt_windows", []):
        cwt_rows.append(
            [
                item["window"],
                f"{item['start']:,}",
                f"{item['end']:,}",
                f"{item['mean_energy']:.6g}",
                f"{item['max_energy']:.6g}",
                str(item["dominant_scale"]),
                f"{item['dominant_pseudofreq']:.8g}",
            ]
        )
    story.append(
        _report_table(
            cwt_rows,
            widths=[
                2.3 * cm,
                2.0 * cm,
                2.0 * cm,
                2.3 * cm,
                2.3 * cm,
                2.2 * cm,
                2.5 * cm,
            ],
            font_size=6.8,
        )
    )
    for key, label in (
        ("cwt_beginning", "Beginning"),
        ("cwt_middle", "Middle"),
        ("cwt_end", "End"),
    ):
        story.extend(
            _figure_flowable(
                paths.get(key),
                f"CWT scalogram - {label} window.",
                styles,
            )
        )

    story.append(
        Paragraph("9. Discrete Wavelet Transform", styles["Section"])
    )
    dwt_rows = [["Band", "Energy"]]
    for label, energy in zip(
        metrics.get("dwt_labels", []),
        metrics.get("dwt_energies", []),
    ):
        dwt_rows.append([label, f"{energy:.8g}"])
    if len(dwt_rows) == 1:
        dwt_rows.append(["-", "-"])
    story.append(
        _report_table(
            dwt_rows,
            widths=[5.0 * cm, 8.5 * cm],
        )
    )
    story.extend(
        _figure_flowable(
            paths.get("dwt"),
            f"DWT coefficient-band energy using "
            f"{metrics['dwt_wavelet']}.",
            styles,
        )
    )

    story.append(
        Paragraph(
            "10. Additional Quantitative Metrics",
            styles["Section"],
        )
    )
    additional_rows = [
        ["Metric", "Value"],
        [
            "Normalized Lempel-Ziv complexity",
            f"{metrics['normalized_lz_complexity']:.8f}",
        ],
        [
            "Matrix Profile motif candidate",
            "-"
            if metrics["matrix_profile_motif_index"] is None
            else (
                f"Approx. sample "
                f"{metrics['matrix_profile_motif_index']:,}"
            ),
        ],
        [
            "Matrix Profile discord candidate",
            "-"
            if metrics["matrix_profile_discord_index"] is None
            else (
                f"Approx. sample "
                f"{metrics['matrix_profile_discord_index']:,}"
            ),
        ],
    ]
    story.append(
        _report_table(
            additional_rows,
            widths=[7.2 * cm, 6.3 * cm],
        )
    )
    story.extend(
        _figure_flowable(
            paths.get("matrix_profile"),
            "Compact Matrix Profile used to identify repeated and "
            "locally unusual subsequences.",
            styles,
        )
    )

    story.append(
        Paragraph("11. Measurement Notes", styles["Section"])
    )
    story.append(
        Paragraph(
            "All values in this report are quantitative measurements "
            "or direct transform-derived descriptors of the loaded "
            "bitstream. Spectral peaks, correlation peaks, entropy "
            "values, wavelet energies and complexity measures do not "
            "by themselves establish the semantic origin of the data, "
            "prove randomness, identify encryption/compression, or "
            "determine a generating algorithm. Interpretation should "
            "be based on the numerical evidence and independent "
            "knowledge of the source data.",
            styles["BodySmall"],
        )
    )

    _append_reference_section(story, matches, paths, styles)

    document.build(
        story,
        onFirstPage=_page_header_footer,
        onLaterPages=_page_header_footer,
    )

    return os.path.abspath(PDF_PATH)


def build_pdf(metrics, paths, source_file=None):
    return build_original_report_with_appendix(
        metrics,
        paths,
        source_file or "",
        [],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Binary Signal Analyzer backend"
    )
    parser.add_argument("input_file")
    parser.add_argument("--output", default=PDF_PATH)
    parser.add_argument("--fs", type=float, default=1.0)
    parser.add_argument("--cwt", default="morl")
    parser.add_argument("--dwt", default="db4")
    args = parser.parse_args()

    PDF_PATH = os.path.abspath(args.output)
    OUTPUT_DIR = os.path.dirname(PDF_PATH) or os.getcwd()
    IMG_DIR = os.path.join(OUTPUT_DIR, "images")

    setup_dirs()
    input_bits = load_bits(args.input_file)
    analysis_metrics, analysis_paths = generate_plots(
        input_bits,
        fs=args.fs,
        cwt_wavelet=args.cwt,
        dwt_wavelet=args.dwt,
    )
    build_original_report_with_appendix(
        analysis_metrics,
        analysis_paths,
        args.input_file,
        [],
    )
    print(PDF_PATH)
