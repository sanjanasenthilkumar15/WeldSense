"""
sensors.py
==========
Sensor reading functions for the BattleEdge inspection pipeline.

  read_weld_image(filepath)   → (224,224,3) normalised array + original BGR
  read_thermal(filepath)      → normalised array + {"peak", "mean", "std"} dict
  audio_to_spectrogram(path)  → mel-spectrogram ndarray (128, time_steps)

All functions are pure (no side-effects) and work with either real sensor
data or the synthetic files produced by data/generate_synthetic.py.
"""

import os
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CAMERA  — visual weld image
# ─────────────────────────────────────────────────────────────────────────────

def read_weld_image(filepath: str):
    """
    Read a weld image and return a model-ready array plus the display version.

    Parameters
    ----------
    filepath : str  path to .jpg / .png image

    Returns
    -------
    img_normalized : np.ndarray  shape (224, 224, 3), float32, values 0-1
    img_display    : np.ndarray  shape (H, W, 3), uint8, RGB colour order
    """
    img = cv2.imread(filepath)
    if img is None:
        raise FileNotFoundError(f"Image not found: {filepath}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize to 224×224 (standard input for most vision models)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)

    # normalise pixel values to 0-1 float32
    img_normalized = img_resized.astype(np.float32) / 255.0

    return img_normalized, img_rgb


# ─────────────────────────────────────────────────────────────────────────────
# 2.  IR SENSOR  — thermal heatmap
# ─────────────────────────────────────────────────────────────────────────────

def read_thermal(filepath: str):
    """
    Read a thermal heatmap PNG and extract three scalar features.

    Parameters
    ----------
    filepath : str  path to thermal PNG

    Returns
    -------
    thermal_norm : np.ndarray  shape (H, W), float32, values 0-1
    features     : dict        {"peak", "mean", "std"}
                    peak → highest temperature proxy  (higher = hotter spot)
                    mean → average temperature        (higher = generally hot)
                    std  → temperature unevenness     (higher = suspect weld)
    """
    thermal = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if thermal is None:
        raise FileNotFoundError(f"Thermal image not found: {filepath}")

    thermal_norm = thermal.astype(np.float32) / 255.0

    peak_temp = float(np.max(thermal_norm))
    mean_temp = float(np.mean(thermal_norm))
    std_temp  = float(np.std(thermal_norm))

    features = {
        "peak": round(peak_temp, 3),
        "mean": round(mean_temp, 3),
        "std":  round(std_temp,  3),
    }

    return thermal_norm, features


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MICROPHONE  — acoustic spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def audio_to_spectrogram(wav_path: str, save_path: str = None):
    """
    Convert a WAV file to a mel-spectrogram 2-D array.

    Uses librosa if available; falls back to numpy STFT otherwise.

    Parameters
    ----------
    wav_path  : str   input .wav file
    save_path : str | None   if given, save a 224×224 PNG of the spectrogram

    Returns
    -------
    mel_db : np.ndarray  shape (128, time_steps), values in dB  (librosa path)
           OR STFT magnitude array (fallback path)
    """
    try:
        import librosa
        import librosa.display
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y, sr = librosa.load(wav_path, sr=16_000, duration=2.0)

        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=128,
            fmax=8_000,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if save_path:
            fig, ax = plt.subplots(figsize=(2.24, 2.24))
            librosa.display.specshow(mel_db, sr=sr, fmax=8_000, ax=ax)
            ax.axis("off")
            fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        return mel_db

    except ImportError:
        # ── Fallback: scipy + numpy STFT ──────────────────────────────────
        import scipy.io.wavfile as wav_io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rate, data = wav_io.read(wav_path)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float32) / 32767.0

        n_fft = 512
        hop   = 256
        frames = [
            np.abs(np.fft.rfft(data[i:i + n_fft] * np.hanning(n_fft), n=n_fft))
            for i in range(0, max(1, len(data) - n_fft), hop)
        ]
        spec = np.array(frames).T  # (freq_bins, time_frames)
        spec_db = 20 * np.log10(spec + 1e-9)

        if save_path:
            fig, ax = plt.subplots(figsize=(2.24, 2.24))
            ax.imshow(spec_db, aspect="auto", origin="lower", cmap="magma")
            ax.axis("off")
            fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        return spec_db


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMG_DIR     = os.path.join(ROOT, "data", "images")
    THERMAL_DIR = os.path.join(ROOT, "data", "thermal")
    AUDIO_DIR   = os.path.join(ROOT, "data", "audio")

    # ── visual ──
    img_files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    if img_files:
        arr, disp = read_weld_image(os.path.join(IMG_DIR, img_files[0]))
        print(f"[visual]  shape={arr.shape}, dtype={arr.dtype}, "
              f"min={arr.min():.3f}, max={arr.max():.3f}")
    else:
        print("[visual]  No images found — run data/generate_synthetic.py first.")

    # ── thermal ──
    th_files = [f for f in os.listdir(THERMAL_DIR) if f.endswith(".png")]
    if th_files:
        th, feat = read_thermal(os.path.join(THERMAL_DIR, th_files[0]))
        print(f"[thermal] shape={th.shape}, features={feat}")
    else:
        print("[thermal] No thermal images found — run data/generate_synthetic.py first.")

    # ── audio ──
    wav_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    if wav_files:
        spec = audio_to_spectrogram(os.path.join(AUDIO_DIR, wav_files[0]))
        print(f"[audio]   spectrogram shape={spec.shape}")
    else:
        print("[audio]   No WAV files found — run data/generate_synthetic.py first.")
