"""Read-only diagnostics for the existing 3-second MTG Brova Barama baseline.

Does not overwrite outputs/mtg_baseline/. Writes only to outputs/mtg_diagnostics/.
Does not run model inference.
"""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "outputs" / "mtg_baseline"
DIAG_DIR = PROJECT_ROOT / "outputs" / "mtg_diagnostics"
DATA_DIR = PROJECT_ROOT / "data" / "saraga_mridangam_brova_barama"
CKPT_DIR = PROJECT_ROOT / "models" / "mtg" / "saraga-8"
TARGET_SR = 22_050
EPSILON = 1e-12
SEGMENT_SEC = 3.0
WIN = 1024
HOP = 256


def si_sdr_diagnostic(reference: np.ndarray, estimate: np.ndarray) -> Dict[str, Any]:
    """Return a JSON-safe SI-SDR value and explain undefined or infinite cases."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    result: Dict[str, Any] = {
        "value_db": None,
        "status": "invalid_input",
        "reason": None,
        "reference_energy": None,
        "estimate_energy": None,
        "projection_energy": None,
        "residual_energy": None,
    }
    if reference.ndim != 1 or estimate.ndim != 1 or reference.size != estimate.size or reference.size == 0:
        result["reason"] = "Reference and estimate must be non-empty, one-dimensional arrays of equal length."
        return result
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        result["reason"] = "Reference or estimate contains NaN or infinite samples."
        return result

    with np.errstate(over="ignore", invalid="ignore"):
        ref = reference - np.mean(reference)
        est = estimate - np.mean(estimate)
        ref_energy = float(ref @ ref)
        est_energy = float(est @ est)
    if not np.isfinite(ref_energy):
        result["status"] = "numerical_failure"
        result["reason"] = "Reference energy overflowed or became non-finite."
        return result
    result["reference_energy"] = ref_energy
    if not np.isfinite(est_energy):
        result["status"] = "numerical_failure"
        result["reason"] = "Estimate energy overflowed or became non-finite."
        return result
    result["estimate_energy"] = est_energy
    if ref_energy <= 0:
        result["status"] = "invalid_reference"
        result["reason"] = "Reference signal must have non-zero, finite energy after mean removal."
        return result
    if est_energy == 0:
        if np.all(estimate == 0):
            result["status"] = "undefined_silent_estimate"
            result["reason"] = "Silent estimate gives zero projection and zero residual energy (0/0)."
        else:
            result["status"] = "undefined_zero_centered_estimate"
            result["reason"] = "Estimate has no energy after mean removal, so projection and residual are both zero."
        result["projection_energy"] = 0.0
        result["residual_energy"] = 0.0
        return result

    with np.errstate(over="ignore", invalid="ignore"):
        projection = (float(est @ ref) / ref_energy) * ref
        residual = est - projection
        proj_energy = float(projection @ projection)
        res_energy = float(residual @ residual)
    if not np.isfinite(proj_energy) or not np.isfinite(res_energy):
        result["status"] = "numerical_failure"
        result["reason"] = "Projection or residual energy is non-finite."
        return result
    result["projection_energy"] = proj_energy
    result["residual_energy"] = res_energy
    if res_energy <= EPSILON:
        if proj_energy > EPSILON:
            result["status"] = "positive_infinity_perfect_projection"
            result["reason"] = "Residual energy is zero within tolerance while projection energy is non-zero."
        else:
            result["status"] = "numerical_failure"
            result["reason"] = "Projection and residual energies are both below the numerical tolerance."
    elif proj_energy <= EPSILON:
        result["status"] = "negative_infinity_zero_projection"
        result["reason"] = "Projection energy is zero within tolerance while residual energy is non-zero."
    else:
        value_db = float(10.0 * np.log10(proj_energy / res_energy))
        if np.isfinite(value_db):
            result["value_db"] = value_db
            result["status"] = "finite"
            result["reason"] = None
        else:
            result["status"] = "numerical_failure"
            result["reason"] = "SI-SDR calculation produced a non-finite value."
    return result


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Le Roux et al. scale-invariant SDR with mean removal."""
    result = si_sdr_diagnostic(reference, estimate)
    if result["status"] != "finite":
        raise ValueError(f"SI-SDR is {result['status']}: {result['reason']}")
    return float(result["value_db"])


def si_sdr_no_dc(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Same formula without mean subtraction."""
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    ref_energy = float(ref @ ref)
    projection = (float(est @ ref) / ref_energy) * ref
    residual = est - projection
    return float(10.0 * np.log10(max(float(projection @ projection), EPSILON) / max(float(residual @ residual), EPSILON)))


def stats(x: np.ndarray) -> Dict[str, Any]:
    x64 = np.asarray(x, dtype=np.float64)
    return {
        "n_samples": int(x64.size),
        "duration_s": float(x64.size / TARGET_SR),
        "peak": float(np.max(np.abs(x64))),
        "max_signed": float(np.max(x64)),
        "min_signed": float(np.min(x64)),
        "rms": float(np.sqrt(np.mean(x64**2))),
        "mean": float(np.mean(x64)),
        "finite": bool(np.isfinite(x64).all()),
        "n_nan": int(np.isnan(x64).sum()),
        "n_inf": int(np.isinf(x64).sum()),
        "n_zero": int(np.sum(x64 == 0)),
        "zero_fraction": float(np.mean(x64 == 0)),
    }


def load_wav(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), int(sr)


def wav_header(path: Path) -> Dict[str, Any]:
    with wave.open(str(path), "rb") as wf:
        return {
            "nchannels": wf.getnchannels(),
            "sampwidth": wf.getsampwidth(),
            "framerate": wf.getframerate(),
            "nframes": wf.getnframes(),
            "comptype": wf.getcomptype(),
        }


def best_lag(reference: np.ndarray, estimate: np.ndarray, max_lag: int) -> Dict[str, Any]:
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    ref = ref - ref.mean()
    est = est - est.mean()
    corr = signal.correlate(est, ref, mode="full")
    lags = signal.correlation_lags(est.size, ref.size, mode="full")
    mask = np.abs(lags) <= max_lag
    corr = corr[mask]
    lags = lags[mask]
    denom = float(np.linalg.norm(ref) * np.linalg.norm(est) + EPSILON)
    # Use normalized correlation at lag 0 and at best lag on equal-length overlap.
    def ncc_at(lag: int) -> float:
        if lag >= 0:
            a = est[lag:]
            b = ref[: a.size]
        else:
            b = ref[-lag:]
            a = est[: b.size]
        n = min(a.size, b.size)
        a = a[:n] - a[:n].mean()
        b = b[:n] - b[:n].mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPSILON))

    lag0 = ncc_at(0)
    best_idx = int(np.argmax(np.abs(corr)))
    lag = int(lags[best_idx])
    best = ncc_at(lag)
    return {
        "lag0_normalized_corr": lag0,
        "best_lag_samples": lag,
        "best_lag_ms": float(1000.0 * lag / TARGET_SR),
        "best_lag_normalized_corr": best,
        "max_lag_searched_samples": int(max_lag),
    }


def shift_for_sisdr(reference: np.ndarray, estimate: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    if lag == 0:
        n = min(reference.size, estimate.size)
        return reference[:n], estimate[:n]
    if lag > 0:
        est = estimate[lag:]
        ref = reference[: est.size]
    else:
        ref = reference[-lag:]
        est = estimate[: ref.size]
    n = min(ref.size, est.size)
    return ref[:n], est[:n]


def official_next_power_of_2(n: int) -> int:
    n = n - 1
    lg = int(np.log(n) / np.log(2))
    return 1 << lg


def stft_truncation_notes(n_samples: int) -> Dict[str, Any]:
    n_frames = 1 + (n_samples - WIN) // HOP
    even_frames = n_frames - (n_frames % 2)
    truncated = official_next_power_of_2(even_frames)
    approx_kept_samples = truncated * HOP
    return {
        "input_samples": n_samples,
        "stft_frames_approx": n_frames,
        "even_frames": even_frames,
        "official_next_power_of_2_kept_frames": truncated,
        "approx_kept_samples_after_frame_trim": approx_kept_samples,
        "approx_kept_seconds": float(approx_kept_samples / TARGET_SR),
        "local_hop_samples": int((3.0 * TARGET_SR) / 2),
        "official_hop_if_using_pred_len": int(approx_kept_samples // 2),
    }


def inspect_checkpoint() -> Dict[str, Any]:
    data_path = CKPT_DIR / "saraga-8.ckpt-1.data-00000-of-00001"
    index_path = CKPT_DIR / "saraga-8.ckpt-1.index"
    pointer_path = CKPT_DIR / "checkpoint"
    info: Dict[str, Any] = {
        "pointer": pointer_path.read_text(encoding="utf-8") if pointer_path.exists() else None,
        "data_bytes": data_path.stat().st_size if data_path.exists() else None,
        "index_bytes": index_path.stat().st_size if index_path.exists() else None,
        "expected_zenodo_zip_about_gb": 1.47,
        "data_gb": float(data_path.stat().st_size / (1024**3)) if data_path.exists() else None,
    }
    if not index_path.exists():
        return info
    raw = index_path.read_bytes()
    strings = []
    current = []
    for b in raw:
        if 32 <= b < 127:
            current.append(chr(b))
        else:
            if len(current) >= 8:
                strings.append("".join(current))
            current = []
    if len(current) >= 8:
        strings.append("".join(current))
    interesting = [
        s
        for s in strings
        if any(k in s.lower() for k in ("model", "unet", "conv", "dense", "kernel", "gamma", "beta"))
    ]
    info["index_printable_string_count"] = len(strings)
    info["index_sample_variable_like_strings"] = interesting[:40]
    info["contains_model_prefix"] = any("model" in s.lower() for s in strings)
    info["contains_net_or_unet"] = any(("unet" in s.lower() or "/net" in s.lower() or "net/" in s.lower()) for s in strings)
    return info


def try_load_source(path: Path, duration_s: float = SEGMENT_SEC) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        info = sf.info(str(path))
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        n_take = int(duration_s * sr)
        audio = audio[:n_take]
        mono = np.mean(audio, axis=1)
        return {
            "path": str(path),
            "exists": True,
            "readable": True,
            "format": info.format,
            "subtype": info.subtype,
            "samplerate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "duration_s": float(info.frames / info.samplerate) if info.samplerate else None,
            "segment_samples": int(mono.size),
            "segment_peak": float(np.max(np.abs(mono))),
            "segment_rms": float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))),
        }
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "readable": False,
            "error": f"{type(exc).__name__}: {exc}",
            "size_bytes": path.stat().st_size,
        }


def envelope(x: np.ndarray, win: int = 512) -> np.ndarray:
    mag = np.abs(x.astype(np.float64))
    kernel = np.ones(win) / win
    return np.convolve(mag, kernel, mode="same")


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "mixture": BASELINE_DIR / "saraga_brova_barama_mixture_mixture_22k.wav",
        "vocal_estimate": BASELINE_DIR / "saraga_brova_barama_mixture_vocal_estimate.wav",
        "accomp_estimate": BASELINE_DIR / "saraga_brova_barama_mixture_accomp_estimate.wav",
        "vocal_reference": BASELINE_DIR / "saraga_brova_barama_mixture_vocal_reference_22k.wav",
    }
    report: Dict[str, Any] = {
        "baseline_dir": str(BASELINE_DIR),
        "diagnostic_dir": str(DIAG_DIR),
        "wav_headers": {k: wav_header(p) for k, p in files.items()},
        "checkpoint": inspect_checkpoint(),
        "stft_truncation_for_3s": stft_truncation_notes(int(SEGMENT_SEC * TARGET_SR)),
        "source_mp3": {
            "mixture": try_load_source(DATA_DIR / "saraga_brova_barama_mixture.mp3"),
            "vocal": try_load_source(DATA_DIR / "saraga_brova_barama_vocal.mp3"),
            "mridangam_left": try_load_source(DATA_DIR / "saraga_brova_barama_mridangam_left.mp3"),
            "mridangam_right": try_load_source(DATA_DIR / "saraga_brova_barama_mridangam_right.mp3"),
        },
        "accompaniment_reference_available": False,
        "accompaniment_note": (
            "No isolated accompaniment ground-truth stem is present. "
            "Mridangam left/right are percussion parts with bleed, not a full accompaniment reference. "
            "Mixture-minus-vocals is not treated as ground truth."
        ),
    }

    signals = {}
    for name, path in files.items():
        audio, sr = load_wav(path)
        if sr != TARGET_SR:
            report.setdefault("sample_rate_mismatches", {})[name] = sr
        signals[name] = audio
        report.setdefault("signal_stats", {})[name] = stats(audio)

    n = min(len(s) for s in signals.values())
    for k in list(signals):
        signals[k] = signals[k][:n]
    mix = signals["mixture"]
    est = signals["vocal_estimate"]
    ref = signals["vocal_reference"]
    acc = signals["accomp_estimate"]

    residual = mix - est
    residual_match = float(np.max(np.abs(residual - acc[:n])))

    mix_sdr = si_sdr(ref, mix)
    est_sdr = si_sdr(ref, est)
    mix_sdr_nodc = si_sdr_no_dc(ref, mix)
    est_sdr_nodc = si_sdr_no_dc(ref, est)

    max_lag = 2048  # ~93 ms at 22.05 kHz; STFT group delay is typically within a window
    lag_est = best_lag(ref, est, max_lag)
    lag_mix = best_lag(ref, mix, max_lag)
    lag_est_vs_mix = best_lag(mix, est, max_lag)

    ref_aln, est_aln = shift_for_sisdr(ref, est, lag_est["best_lag_samples"])
    ref_aln_m, mix_aln = shift_for_sisdr(ref, mix, lag_mix["best_lag_samples"])
    # Use the SAME lag for both mixture and estimate so a delay is not hidden for one and not the other.
    shared_lag = lag_est_vs_mix["best_lag_samples"]
    mix_s, est_s = shift_for_sisdr(mix, est, shared_lag)
    n_shared = min(mix_s.size, est_s.size, ref.size)
    # Align estimate to mixture, then compare both to the unshifted reference cropped to overlap with mixture.
    if shared_lag >= 0:
        ref_shared = ref[:n_shared]
        mix_shared = mix[:n_shared]
        est_shared = est[shared_lag : shared_lag + n_shared]
    else:
        ref_shared = ref[-shared_lag : -shared_lag + n_shared]
        mix_shared = mix[-shared_lag : -shared_lag + n_shared]
        est_shared = est[:n_shared]

    report["reconstruction_checks"] = {
        "accomp_equals_mix_minus_vocals_max_abs_diff": residual_match,
        "equal_lengths": {k: int(v.size) for k, v in signals.items()},
        "pcm16_quantization_note": "WAV files are PCM_16, so residual match may be a few LSBs from quantization.",
    }
    report["si_sdr"] = {
        "formula": "10*log10(||proj||^2 / ||est-proj||^2), proj = <est,ref>/<ref,ref> * ref, after mean removal",
        "same_region_samples": n,
        "mixture_vs_vocal_ref_db": mix_sdr,
        "estimate_vs_vocal_ref_db": est_sdr,
        "change_db": est_sdr - mix_sdr,
        "without_dc_removal": {
            "mixture_vs_vocal_ref_db": mix_sdr_nodc,
            "estimate_vs_vocal_ref_db": est_sdr_nodc,
            "change_db": est_sdr_nodc - mix_sdr_nodc,
        },
        "lag_search": {
            "estimate_vs_ref": lag_est,
            "mixture_vs_ref": lag_mix,
            "estimate_vs_mixture": lag_est_vs_mix,
        },
        "after_best_lag_estimate_to_ref": {
            "lag_samples": lag_est["best_lag_samples"],
            "mixture_unshifted_on_same_crop_not_recomputed": True,
            "estimate_vs_vocal_ref_db": si_sdr(ref_aln, est_aln),
            "n_samples": int(ref_aln.size),
        },
        "after_best_lag_mixture_to_ref": {
            "lag_samples": lag_mix["best_lag_samples"],
            "mixture_vs_vocal_ref_db": si_sdr(ref_aln_m, mix_aln),
            "n_samples": int(ref_aln_m.size),
        },
        "after_shared_estimate_to_mixture_lag": {
            "lag_samples": shared_lag,
            "mixture_vs_vocal_ref_db": si_sdr(ref_shared, mix_shared),
            "estimate_vs_vocal_ref_db": si_sdr(ref_shared, est_shared),
            "change_db": si_sdr(ref_shared, est_shared) - si_sdr(ref_shared, mix_shared),
            "n_samples": int(n_shared),
            "note": "Lag is estimate-vs-mixture delay; reference is cropped consistently, not independently time-warped.",
        },
    }

    # Sliding 0.5 s SI-SDR windows to see if the score is uniformly poor.
    win = int(0.5 * TARGET_SR)
    hop_w = int(0.25 * TARGET_SR)
    sliding = []
    for start in range(0, n - win + 1, hop_w):
        sl = slice(start, start + win)
        mix_result = si_sdr_diagnostic(ref[sl], mix[sl])
        est_result = si_sdr_diagnostic(ref[sl], est[sl])
        mix_db = mix_result["value_db"]
        est_db = est_result["value_db"]
        change_db = est_db - mix_db if mix_db is not None and est_db is not None else None
        sliding.append(
            {
                "start_s": float(start / TARGET_SR),
                "end_s": float((start + win) / TARGET_SR),
                "mix_db": mix_db,
                "mix_status": mix_result["status"],
                "mix_reason": mix_result["reason"],
                "est_db": est_db,
                "est_status": est_result["status"],
                "est_reason": est_result["reason"],
                "est_projection_energy": est_result["projection_energy"],
                "est_residual_energy": est_result["residual_energy"],
                "change_db": change_db,
                "change_status": "finite" if change_db is not None else "undefined",
                "change_reason": None if change_db is not None else (
                    "Change requires finite mixture and estimate SI-SDR values."
                ),
            }
        )
    report["sliding_0p5s_si_sdr"] = sliding

    # Compare resampled source MP3 segment vs saved 22k mixture/reference if readable.
    src_mix = report["source_mp3"]["mixture"]
    src_voc = report["source_mp3"]["vocal"]
    if src_mix and src_mix.get("readable") and src_voc and src_voc.get("readable"):
        mix_src, sr_m = sf.read(str(DATA_DIR / "saraga_brova_barama_mixture.mp3"), dtype="float32", always_2d=True)
        voc_src, sr_v = sf.read(str(DATA_DIR / "saraga_brova_barama_vocal.mp3"), dtype="float32", always_2d=True)
        mix_src = np.mean(mix_src, axis=1)
        voc_src = np.mean(voc_src, axis=1)
        n_m = int(SEGMENT_SEC * sr_m)
        n_v = int(SEGMENT_SEC * sr_v)
        mix_src = mix_src[:n_m]
        voc_src = voc_src[:n_v]
        if sr_m != TARGET_SR:
            g = math.gcd(sr_m, TARGET_SR)
            mix_rs = signal.resample_poly(mix_src, TARGET_SR // g, sr_m // g).astype(np.float32)
        else:
            mix_rs = mix_src
        if sr_v != TARGET_SR:
            g = math.gcd(sr_v, TARGET_SR)
            voc_rs = signal.resample_poly(voc_src, TARGET_SR // g, sr_v // g).astype(np.float32)
        else:
            voc_rs = voc_src
        nn = min(len(mix_rs), len(voc_rs), n)
        mix_rs, voc_rs = mix_rs[:nn], voc_rs[:nn]
        report["source_vs_saved"] = {
            "resampled_source_mixture_vs_saved_max_abs": float(np.max(np.abs(mix_rs[:nn] - mix[:nn]))),
            "resampled_source_vocal_vs_saved_max_abs": float(np.max(np.abs(voc_rs[:nn] - ref[:nn]))),
            "source_si_sdr_mixture_vs_vocal_db": si_sdr(voc_rs, mix_rs),
            "source_durations": {
                "mixture_s": float(sf.info(str(DATA_DIR / "saraga_brova_barama_mixture.mp3")).frames / sr_m),
                "vocal_s": float(sf.info(str(DATA_DIR / "saraga_brova_barama_vocal.mp3")).frames / sr_v),
            },
            "duration_difference_s": abs(
                sf.info(str(DATA_DIR / "saraga_brova_barama_mixture.mp3")).frames / sr_m
                - sf.info(str(DATA_DIR / "saraga_brova_barama_vocal.mp3")).frames / sr_v
            ),
        }
        lag_src = best_lag(voc_rs, mix_rs, max_lag)
        report["source_vs_saved"]["mixture_vs_vocal_lag"] = lag_src

    # Plots
    t = np.arange(n) / TARGET_SR
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("MTG 3s Brova Barama diagnostics (existing baseline files)")
    axes[0].plot(t, mix, lw=0.6, label="mixture")
    axes[0].plot(t, ref, lw=0.6, label="vocal ref", alpha=0.85)
    axes[0].plot(t, est, lw=0.6, label="vocal estimate", alpha=0.85)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("Waveforms")
    axes[1].plot(t, envelope(mix), lw=0.8, label="mix")
    axes[1].plot(t, envelope(ref), lw=0.8, label="ref")
    axes[1].plot(t, envelope(est), lw=0.8, label="est")
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("smoothed |x|")
    axes[1].set_title("Amplitude envelopes")
    starts = [row["start_s"] for row in sliding]
    axes[2].plot(starts, [row["mix_db"] for row in sliding], marker="o", label="mix SI-SDR")
    axes[2].plot(
        starts,
        [np.nan if row["est_db"] is None else row["est_db"] for row in sliding],
        marker="o",
        label="est SI-SDR",
    )
    axes[2].axhline(0.0, color="gray", lw=0.8)
    axes[2].legend(loc="upper right")
    axes[2].set_ylabel("dB")
    axes[2].set_title("Sliding 0.5 s SI-SDR vs vocal reference")
    axes[3].plot(
        starts,
        [np.nan if row["change_db"] is None else row["change_db"] for row in sliding],
        marker="o",
        color="tab:red",
    )
    axes[3].axhline(0.0, color="gray", lw=0.8)
    axes[3].set_ylabel("dB")
    axes[3].set_xlabel("window start (s)")
    axes[3].set_title("Sliding SI-SDR change (estimate minus mixture)")
    fig.tight_layout()
    plot_path = DIAG_DIR / "brova_barama_3s_sisdr_alignment_diagnostics_strict.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    json_path = DIAG_DIR / "brova_barama_3s_diagnostics.json"
    report["plot_path"] = str(plot_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print(json.dumps({
        "json_path": str(json_path),
        "plot_path": str(plot_path),
        "si_sdr": report["si_sdr"],
        "signal_stats": report["signal_stats"],
        "stft_truncation_for_3s": report["stft_truncation_for_3s"],
        "source_mp3_readable": {
            k: bool(v and v.get("readable")) for k, v in report["source_mp3"].items()
        },
        "checkpoint_data_gb": report["checkpoint"].get("data_gb"),
    }, indent=2))


if __name__ == "__main__":
    main()
