import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.signal import welch, correlate
import pywt


# ------------------------------
# Load bit sequence
# ------------------------------
def load_bits(path):

    ext = os.path.splitext(path)[1].lower()

    # CSV / TXT
    if ext in [".csv", ".txt"]:

        with open(path, "r") as f:
            text = f.read()

        bits = []

        for c in text:
            if c == "0" or c == "1":
                bits.append(int(c))

        bits = np.array(bits)

    # BINARIO
    else:

        with open(path, "rb") as f:
            raw = f.read()

        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))

    return bits


# ------------------------------
# FFT
# ------------------------------

def compute_fft(signal, fs):

    N = len(signal)

    fft = np.fft.fft(signal)

    freqs = np.fft.fftfreq(N, d=1/fs)

    magnitude = np.abs(fft)

    positive = freqs > 0

    return freqs[positive], magnitude[positive]


# ------------------------------
# Spectral entropy
# ------------------------------

def spectral_entropy(magnitude):

    prob = magnitude / np.sum(magnitude)

    entropy = -np.sum(prob * np.log2(prob + 1e-12))

    return entropy


# ------------------------------
# Autocorrelation
# ------------------------------

def compute_autocorrelation(signal):

    corr = correlate(signal, signal, mode="full")

    corr = corr[corr.size // 2:]

    return corr


# ------------------------------
# Detect periodicity
# ------------------------------

def detect_periodicity(corr):

    peak_index = np.argmax(corr[1:]) + 1

    return peak_index


# ------------------------------
# Wavelet transform
# ------------------------------

def wavelet_transform(signal):

    scales = np.arange(1, 128)

    coef, freqs = pywt.cwt(signal, scales, "morl")

    return coef


# ------------------------------
# Plotting
# ------------------------------

def plot_all(bits, fft_freqs, fft_mag, psd_freqs, psd_power, corr, wavelet):

    plt.figure(figsize=(14,10))

    # Signal
    plt.subplot(3,2,1)
    plt.plot(bits[:500])
    plt.title("Binary Signal (first 500 bits)")

    # FFT
    plt.subplot(3,2,2)
    plt.plot(fft_freqs, fft_mag)
    plt.title("FFT Spectrum")

    # PSD
    plt.subplot(3,2,3)
    plt.semilogy(psd_freqs, psd_power)
    plt.title("Power Spectral Density (Welch)")

    # Autocorrelation
    plt.subplot(3,2,4)
    plt.plot(corr[:500])
    plt.title("Autocorrelation")

    # Wavelet
    plt.subplot(3,2,5)
    plt.imshow(np.abs(wavelet), aspect="auto", cmap="inferno")
    plt.title("Wavelet Transform")

    plt.tight_layout()
    plt.show()


# ------------------------------
# Main
# ------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("file", help="Binary or CSV bit file")
    parser.add_argument("--fs", type=float, default=1.0)

    args = parser.parse_args()

    bits = load_bits(args.file)

    print("Loaded bits:", len(bits))

    signal = bits - np.mean(bits)

    # FFT
    freqs, mag = compute_fft(signal, args.fs)

    dominant_freq = freqs[np.argmax(mag)]

    entropy = spectral_entropy(mag)

    # PSD
    psd_freqs, psd_power = welch(signal)

    # Autocorrelation
    corr = compute_autocorrelation(signal)

    periodicity = detect_periodicity(corr)

    # Wavelet
    wavelet = wavelet_transform(signal)

    print("\n=== SIGNAL FEATURES ===")

    print("Dominant frequency:", dominant_freq)

    print("Spectral entropy:", entropy)

    print("Detected periodicity (samples):", periodicity)

    print("Mean:", np.mean(bits))

    print("Variance:", np.var(bits))

    plot_all(bits, freqs, mag, psd_freqs, psd_power, corr, wavelet)


if __name__ == "__main__":
    main()
