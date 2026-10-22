"""Run the official MTG Cold-Diffusion model for Carnatic singing-voice separation.

Extracts:
1. Singing-voice estimate using the official Cold-Diffusion U-Net, clustering mask, and VAD.
2. Accompaniment residual estimate (mixture minus vocal estimate).
3. Objective evaluation metrics: SI-SDR, SI-SDR improvement, peak, RMS, and finite/clipping checks.
4. Spectrogram and waveform diagnostic plots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm

# Ensure MTG modules can be imported
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import Config
from model import DiffWave
from model.clustering import get_mask
from model.vad import VAD
from utils.signal_processing import (
    compute_signal_from_stft,
    compute_stft,
    get_overlap_window,
    next_power_of_2,
)

TARGET_SR = 22_050
EPSILON = 1e-12


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels."""
    ref = reference.astype(np.float64) - np.mean(reference, dtype=np.float64)
    est = estimate.astype(np.float64) - np.mean(estimate, dtype=np.float64)

    ref_energy = float(ref @ ref)
    if ref_energy <= 0 or not np.isfinite(ref_energy):
        raise ValueError("Reference signal must have non-zero, finite energy.")

    projection = (float(est @ ref) / ref_energy) * ref
    residual = est - projection

    proj_energy = float(projection @ projection)
    res_energy = float(residual @ residual)

    if res_energy <= EPSILON:
        return float("inf")
    return float(10.0 * np.log10(max(proj_energy, EPSILON) / res_energy))


def load_mono(file_path: Path, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load audio file as mono float32, resampled to target_sr."""
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, orig_sr = sf.read(str(file_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if orig_sr != target_sr:
        gcd = np.gcd(orig_sr, target_sr)
        audio = scipy.signal.resample_poly(
            audio, target_sr // gcd, orig_sr // gcd
        ).astype(np.float32)

    return audio, target_sr


def separate_audio(
    model: DiffWave,
    config: Config,
    mixture_audio: np.ndarray,
    chunk_size_sec: float = 3.0,
    clusters: int = 5,
    scheduler: float = 4.0,
) -> np.ndarray:
    """Run official MTG cold-diffusion separation pipeline."""
    mix_tensor = tf.convert_to_tensor(mixture_audio, dtype=tf.float32)
    peak = float(tf.reduce_max(mix_tensor).numpy())
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Mixture must have a positive, finite peak amplitude.")
    norm_mixture = mix_tensor / peak

    output_voc = np.zeros(norm_mixture.shape, dtype=np.float32)
    hop_samples = int((chunk_size_sec * TARGET_SR) / 2)
    runs = math.floor(norm_mixture.shape[0] / hop_samples)

    trim_low = 0
    num_steps = max(1, (runs * 2) - 1)

    print(f"Processing {norm_mixture.shape[0]} samples ({norm_mixture.shape[0]/TARGET_SR:.2f}s) in {num_steps} overlapping hops...")
    for trim in tqdm(range(num_steps), desc="Cold-Diffusion Inference"):
        trim_high = int(trim_low + (hop_samples * 2))
        mix_trim = norm_mixture[trim_low:trim_high]
        if len(mix_trim) == 0:
            break

        # Compute STFT
        mix_mag, mix_phase = compute_stft(mix_trim[None], config)
        new_len = next_power_of_2(mix_mag.shape[1])
        mix_mag_trim = mix_mag[:, :new_len, :]
        mix_phase_trim = mix_phase[:, :new_len, :]

        # DiffWave reverse feature extraction across all cold diffusion steps
        diff_feat = model(mix_mag_trim, mode="train")
        diff_feat = tf.transpose(diff_feat, [1, 0, 2, 3])
        diff_feat_t = tf.squeeze(
            tf.reshape(diff_feat, [1, 8, diff_feat.shape[-2] * diff_feat.shape[-1]]),
            axis=0,
        ).numpy()

        # Normalize feature curves
        normalized_feat = []
        for j in range(diff_feat_t.shape[1]):
            curve = diff_feat_t[:, j]
            norm_curve = curve / (np.max(np.abs(curve)) + 1e-6)
            normalized_feat.append(norm_curve)
        normalized_feat = np.array(normalized_feat, dtype=np.float32)

        # Unsupervised clustering mask
        mask = get_mask(normalized_feat, clusters, scheduler)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        mask = tf.reshape(mask, mix_mag_trim.shape)

        # Apply mask to final feature representation
        diff_feat_last = tf.reshape(diff_feat_t[-1, :], mix_mag_trim.shape)
        output_signal = tf.math.multiply(diff_feat_last, mask)

        # Invert STFT with mixture phase
        output_signal = compute_signal_from_stft(output_signal, mix_phase_trim, config)
        pred_audio = tf.squeeze(output_signal, axis=0).numpy()

        # VAD filtering: silence unvoiced frames
        vad = VAD(pred_audio, sr=TARGET_SR, nFFT=512, win_length=0.025, hop_length=0.01, theshold=0.99)
        if len(vad) > 0 and (np.sum(vad) / len(vad)) < 0.25:
            pred_audio = np.zeros_like(pred_audio)

        # Overlap-add with smooth boundary crossfade
        boundary = "start" if trim == 0 else ("end" if trim == num_steps - 1 else None)
        window = get_overlap_window(pred_audio, boundary=boundary)

        overlap_len = min(len(pred_audio), len(output_voc) - trim_low)
        output_voc[trim_low : trim_low + overlap_len] += pred_audio[:overlap_len] * window[:overlap_len]
        trim_low += hop_samples

    # Scale vocal estimate to match mixture peak scale
    mix_peak = float(np.max(np.abs(mixture_audio)))
    voc_peak = float(np.max(np.abs(output_voc)))
    if voc_peak > 1e-6:
        output_voc = output_voc * (mix_peak / voc_peak)

    return output_voc


def plot_separation_diagnostic(
    signals: Dict[str, np.ndarray],
    sample_rate: int,
    out_path: Path,
    title: str,
) -> None:
    """Generate multi-panel comparison showing waveforms and log spectrograms."""
    num_signals = len(signals)
    fig, axes = plt.subplots(num_signals, 2, figsize=(14, 2.5 * num_signals), squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, (name, sig) in enumerate(signals.items()):
        time_axis = np.arange(len(sig)) / sample_rate
        # Waveform
        axes[i, 0].plot(time_axis, sig, color="teal", alpha=0.8, linewidth=0.6)
        axes[i, 0].set_title(f"{name} — Waveform (Peak: {np.max(np.abs(sig)):.3f}, RMS: {np.sqrt(np.mean(sig**2)):.4f})", fontsize=10)
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].set_ylim(-1.05, 1.05)
        axes[i, 0].grid(True, linestyle=":", alpha=0.6)
        if i == num_signals - 1:
            axes[i, 0].set_xlabel("Time (s)")

        # Spectrogram
        f, t_spec, sxx = scipy.signal.spectrogram(
            sig, fs=sample_rate, nperseg=1024, noverlap=512, window="hann"
        )
        log_sxx = 10.0 * np.log10(np.maximum(sxx, 1e-10))
        im = axes[i, 1].pcolormesh(t_spec, f, log_sxx, shading="gouraud", cmap="magma", vmin=-80, vmax=0)
        axes[i, 1].set_title(f"{name} — Log Spectrogram", fontsize=10)
        axes[i, 1].set_ylabel("Freq (Hz)")
        axes[i, 1].set_ylim(0, 8000)
        if i == num_signals - 1:
            axes[i, 1].set_xlabel("Time (s)")
        plt.colorbar(im, ax=axes[i, 1], orientation="vertical", pad=0.02, aspect=15)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def run_baseline(
    mixture_path: Path,
    vocal_ref_path: Optional[Path],
    mridangam_ref_paths: Optional[List[Path]],
    output_dir: Path,
    checkpoint_dir: Path,
    max_duration_sec: Optional[float] = 30.0,
    clusters: int = 5,
    scheduler: float = 4.0,
) -> Dict[str, Any]:
    """Execute complete MTG baseline inference and evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Config()

    # Load pretrained model
    ckpt_path = checkpoint_dir / "saraga-8.ckpt-1"
    print(f"Loading MTG checkpoint from: {ckpt_path}...")
    model = DiffWave(config.model)
    restore_status = model.restore(str(ckpt_path))
    restore_status.expect_partial()

    # Load audio
    mix, sr = load_mono(mixture_path)
    if max_duration_sec is not None:
        max_samples = int(max_duration_sec * sr)
        mix = mix[:max_samples]

    # Run separation
    pred_voc = separate_audio(
        model=model,
        config=config,
        mixture_audio=mix,
        chunk_size_sec=3.0,
        clusters=clusters,
        scheduler=scheduler,
    )
    restore_status.assert_existing_objects_matched()
    print("MTG DiffWave checkpoint weights restored successfully.")

    # Accompaniment residual estimate
    common_len = min(len(mix), len(pred_voc))
    mix = mix[:common_len]
    pred_voc = pred_voc[:common_len]
    pred_accomp = mix - pred_voc

    # Save generated audio
    rec_stem = mixture_path.stem
    sf.write(str(output_dir / f"{rec_stem}_mixture_22k.wav"), mix, sr, subtype="PCM_16")
    sf.write(str(output_dir / f"{rec_stem}_vocal_estimate.wav"), pred_voc, sr, subtype="PCM_16")
    sf.write(str(output_dir / f"{rec_stem}_accomp_estimate.wav"), pred_accomp, sr, subtype="PCM_16")

    # Metrics dictionary
    metrics = {
        "mixture_path": str(mixture_path),
        "duration_s": float(len(mix) / sr),
        "sample_rate": sr,
        "clusters": clusters,
        "scheduler": scheduler,
        "mix_peak": float(np.max(np.abs(mix))),
        "mix_rms": float(np.sqrt(np.mean(mix**2))),
        "vocal_est_peak": float(np.max(np.abs(pred_voc))),
        "vocal_est_rms": float(np.sqrt(np.mean(pred_voc**2))),
        "accomp_est_peak": float(np.max(np.abs(pred_accomp))),
        "accomp_est_rms": float(np.sqrt(np.mean(pred_accomp**2))),
        "finite_samples": bool(np.isfinite(pred_voc).all() and np.isfinite(pred_accomp).all()),
        "clipping_detected": bool(np.max(np.abs(pred_voc)) > 1.0 or np.max(np.abs(pred_accomp)) > 1.0),
    }

    signals_to_plot = {
        "Mixture": mix,
        "Vocal Estimate": pred_voc,
        "Accompaniment Residual": pred_accomp,
    }

    # Evaluate against vocal reference if provided
    if vocal_ref_path and vocal_ref_path.exists():
        voc_ref, _ = load_mono(vocal_ref_path)
        voc_ref = voc_ref[:common_len]
        sf.write(str(output_dir / f"{rec_stem}_vocal_reference_22k.wav"), voc_ref, sr, subtype="PCM_16")

        sdr_mix_voc = si_sdr(voc_ref, mix)
        sdr_est_voc = si_sdr(voc_ref, pred_voc)
        metrics["vocal_reference"] = {
            "path": str(vocal_ref_path),
            "ref_peak": float(np.max(np.abs(voc_ref))),
            "ref_rms": float(np.sqrt(np.mean(voc_ref**2))),
            "si_sdr_mixture_input_db": float(sdr_mix_voc),
            "si_sdr_vocal_estimate_db": float(sdr_est_voc),
            "si_sdr_improvement_db": float(sdr_est_voc - sdr_mix_voc),
        }
        signals_to_plot["Vocal Ground Truth"] = voc_ref

    # Evaluate accompaniment against mridangam reference if provided
    if mridangam_ref_paths:
        mrid_sigs = [load_mono(p)[0] for p in mridangam_ref_paths if p.exists()]
        if mrid_sigs:
            min_mrid_len = min(len(s) for s in mrid_sigs)
            mrid_ref = sum(s[:min_mrid_len] for s in mrid_sigs)[:common_len]
            sf.write(str(output_dir / f"{rec_stem}_mridangam_reference_22k.wav"), mrid_ref, sr, subtype="PCM_16")

            sdr_mix_mrid = si_sdr(mrid_ref, mix[:len(mrid_ref)])
            sdr_accomp_mrid = si_sdr(mrid_ref, pred_accomp[:len(mrid_ref)])
            metrics["mridangam_reference"] = {
                "paths": [str(p) for p in mridangam_ref_paths],
                "ref_peak": float(np.max(np.abs(mrid_ref))),
                "ref_rms": float(np.sqrt(np.mean(mrid_ref**2))),
                "si_sdr_mixture_vs_mridangam_db": float(sdr_mix_mrid),
                "si_sdr_accomp_vs_mridangam_db": float(sdr_accomp_mrid),
                "si_sdr_improvement_db": float(sdr_accomp_mrid - sdr_mix_mrid),
            }
            signals_to_plot["Mridangam Reference (L+R)"] = mrid_ref

    # Plot
    plot_path = output_dir / f"{rec_stem}_diagnostic_plot.png"
    plot_separation_diagnostic(
        signals=signals_to_plot,
        sample_rate=sr,
        out_path=plot_path,
        title=f"MTG Cold Diffusion Baseline: {mixture_path.name}",
    )
    metrics["plot_path"] = str(plot_path)

    # Save metrics JSON
    metrics_path = output_dir / f"{rec_stem}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- MTG BASELINE EVALUATION RESULTS ---")
    print(f"Recording: {mixture_path.name}")
    print(f"Duration:  {metrics['duration_s']:.2f} s")
    if "vocal_reference" in metrics:
        vr = metrics["vocal_reference"]
        print(f"Vocal SI-SDR Input (Mix):      {vr['si_sdr_mixture_input_db']:+6.2f} dB")
        print(f"Vocal SI-SDR Estimate (MTG):  {vr['si_sdr_vocal_estimate_db']:+6.2f} dB")
        print(f"Vocal SI-SDR Improvement:     {vr['si_sdr_improvement_db']:+6.2f} dB")
    if "mridangam_reference" in metrics:
        mr = metrics["mridangam_reference"]
        print(f"Accompaniment vs Mridangam SI-SDR: {mr['si_sdr_accomp_vs_mridangam_db']:+6.2f} dB (Change: {mr['si_sdr_improvement_db']:+6.2f} dB)")
    print(f"Metrics saved: {metrics_path}")
    print(f"Plot saved:    {plot_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run MTG Carnatic Cold-Diffusion baseline inference.")
    parser.add_argument(
        "--mixture",
        type=Path,
        default=PROJECT_ROOT / "data" / "saraga_mridangam_brova_barama" / "saraga_brova_barama_mixture.mp3",
    )
    parser.add_argument(
        "--vocal-ref",
        type=Path,
        default=PROJECT_ROOT / "data" / "saraga_mridangam_brova_barama" / "saraga_brova_barama_vocal.mp3",
    )
    parser.add_argument(
        "--mridangam-left",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--mridangam-right",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "models" / "mtg" / "saraga-8",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "mtg_baseline",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Max duration in seconds to evaluate (None for full).")
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--scheduler", type=float, default=4.0)

    args = parser.parse_args()

    mrid_refs = []
    if args.mridangam_left and args.mridangam_left.exists():
        mrid_refs.append(args.mridangam_left)
    if args.mridangam_right and args.mridangam_right.exists():
        mrid_refs.append(args.mridangam_right)

    run_baseline(
        mixture_path=args.mixture,
        vocal_ref_path=args.vocal_ref if (args.vocal_ref and args.vocal_ref.exists()) else None,
        mridangam_ref_paths=mrid_refs if mrid_refs else None,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        max_duration_sec=args.duration,
        clusters=args.clusters,
        scheduler=args.scheduler,
    )


if __name__ == "__main__":
    main()
