"""Run a coverage-corrected MTG inference diagnostic without touching baseline outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MTG_SOURCE = PROJECT_ROOT / "src" / "mtg"
if str(MTG_SOURCE) not in sys.path:
    sys.path.insert(0, str(MTG_SOURCE))

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
SEGMENT_SECONDS = 3.0
CHUNK_SECONDS = 3.0
CLUSTERS = 5
SCHEDULER = 4.0
EPSILON = 1e-12

DATA_DIR = PROJECT_ROOT / "data" / "saraga_mridangam_brova_barama"
BASELINE_DIR = PROJECT_ROOT / "outputs" / "mtg_baseline"
DIAGNOSTIC_DIR = PROJECT_ROOT / "outputs" / "mtg_diagnostics"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "mtg" / "saraga-8"
OUTPUT_WAV = DIAGNOSTIC_DIR / "brova_barama_3s_coverage_corrected_verified_vocal.wav"
OUTPUT_JSON = DIAGNOSTIC_DIR / "brova_barama_3s_coverage_corrected_verified_metrics.json"


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64) - np.mean(reference, dtype=np.float64)
    estimate = estimate.astype(np.float64) - np.mean(estimate, dtype=np.float64)
    reference_energy = float(reference @ reference)
    if reference_energy <= 0 or not np.isfinite(reference_energy):
        raise ValueError("Reference signal must have non-zero, finite energy.")
    projection = (float(estimate @ reference) / reference_energy) * reference
    residual = estimate - projection
    projection_energy = float(projection @ projection)
    residual_energy = float(residual @ residual)
    if residual_energy <= EPSILON:
        return float("inf")
    return float(10.0 * np.log10(max(projection_energy, EPSILON) / residual_energy))


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sample_rate != TARGET_SR:
        divisor = int(np.gcd(sample_rate, TARGET_SR))
        audio = scipy.signal.resample_poly(
            audio, TARGET_SR // divisor, sample_rate // divisor
        ).astype(np.float32)
    return audio.astype(np.float32), TARGET_SR


def predict_chunk(
    model: DiffWave,
    config: Config,
    chunk: np.ndarray,
) -> np.ndarray:
    magnitude, phase = compute_stft(chunk[None], config)
    frame_count = int(magnitude.shape[1])
    kept_frames = next_power_of_2(frame_count)
    magnitude = magnitude[:, :kept_frames, :]
    phase = phase[:, :kept_frames, :]

    features = model(magnitude, mode="train")
    features = tf.transpose(features, [1, 0, 2, 3])
    features = tf.squeeze(
        tf.reshape(
            features,
            [1, config.model.iter, features.shape[-2] * features.shape[-1]],
        ),
        axis=0,
    ).numpy()

    normalized_features = np.stack(
        [
            row / (np.max(np.abs(row)) + 1e-6)
            for row in features.T
        ],
        axis=0,
    ).astype(np.float32)
    mask = get_mask(normalized_features, CLUSTERS, SCHEDULER)
    mask = tf.reshape(tf.convert_to_tensor(mask, dtype=tf.float32), magnitude.shape)
    final_feature = tf.reshape(features[-1, :], magnitude.shape)
    reconstructed = compute_signal_from_stft(final_feature * mask, phase, config)
    prediction = tf.squeeze(reconstructed, axis=0).numpy()

    vad = VAD(
        prediction,
        sr=TARGET_SR,
        nFFT=512,
        win_length=0.025,
        hop_length=0.01,
        theshold=0.99,
    )
    if len(vad) > 0 and np.sum(vad) / len(vad) < 0.25:
        prediction = np.zeros_like(prediction)
    return prediction.astype(np.float32)


def describe(audio: np.ndarray) -> dict[str, Any]:
    audio64 = audio.astype(np.float64)
    nonzero = np.flatnonzero(audio != 0)
    last_nonzero = int(nonzero[-1]) if nonzero.size else -1
    return {
        "samples": int(audio.size),
        "duration_seconds": float(audio.size / TARGET_SR),
        "sample_rate": TARGET_SR,
        "channels": 1,
        "peak": float(np.max(np.abs(audio64))),
        "rms": float(np.sqrt(np.mean(audio64**2))),
        "finite": bool(np.isfinite(audio).all()),
        "exact_zero_samples": int(np.count_nonzero(audio == 0)),
        "trailing_exact_zero_samples": int(audio.size - last_nonzero - 1),
        "trailing_exact_zero_seconds": float(
            (audio.size - last_nonzero - 1) / TARGET_SR
        ),
    }


def main() -> None:
    if OUTPUT_WAV.exists() or OUTPUT_JSON.exists():
        raise FileExistsError(
            f"Refusing to overwrite diagnostic results: {OUTPUT_WAV} or {OUTPUT_JSON}"
        )

    mixture_path = DATA_DIR / "saraga_brova_barama_mixture.mp3"
    vocal_reference_path = DATA_DIR / "saraga_brova_barama_vocal.mp3"
    baseline_mixture_path = BASELINE_DIR / "saraga_brova_barama_mixture_mixture_22k.wav"
    baseline_estimate_path = BASELINE_DIR / "saraga_brova_barama_mixture_vocal_estimate.wav"
    baseline_metrics_path = BASELINE_DIR / "saraga_brova_barama_mixture_metrics.json"
    checkpoint_path = CHECKPOINT_DIR / "saraga-8.ckpt-1"

    mixture, sample_rate = load_mono(mixture_path)
    reference, _ = load_mono(vocal_reference_path)
    target_samples = int(SEGMENT_SECONDS * sample_rate)
    mixture = mixture[:target_samples]
    reference = reference[:target_samples]
    if len(mixture) != target_samples or len(reference) != target_samples:
        raise ValueError("Source mixture/reference do not contain the full evaluation interval.")

    baseline_mixture, baseline_sr = sf.read(
        str(baseline_mixture_path), dtype="float32", always_2d=True
    )
    baseline_estimate, estimate_sr = sf.read(
        str(baseline_estimate_path), dtype="float32", always_2d=True
    )
    if (
        baseline_sr != TARGET_SR
        or estimate_sr != TARGET_SR
        or baseline_mixture.shape != (target_samples, 1)
        or baseline_estimate.shape != (target_samples, 1)
    ):
        raise ValueError("Preserved baseline WAVs do not match the expected 3-second mono interval.")

    mixture_peak = float(np.max(mixture))
    if not np.isfinite(mixture_peak) or mixture_peak <= 0:
        raise ValueError("Mixture must have a positive, finite peak amplitude.")
    normalized_mixture = mixture / mixture_peak

    config = Config()
    model = DiffWave(config.model)
    restore_status = model.restore(str(checkpoint_path))
    restore_status.expect_partial()

    chunk_samples = int(CHUNK_SECONDS * sample_rate)
    output = np.zeros(target_samples, dtype=np.float64)
    weight_sum = np.zeros(target_samples, dtype=np.float64)
    chunk_trace: list[dict[str, Any]] = []
    cursor = 0

    for chunk_index in tqdm(range(100), desc="Coverage-corrected MTG inference"):
        if cursor >= target_samples:
            break

        end = min(cursor + chunk_samples, target_samples)
        original_chunk = normalized_mixture[cursor:end]
        chunk = np.pad(
            original_chunk,
            (0, chunk_samples - original_chunk.size),
        )
        prediction = predict_chunk(model, config, chunk)
        if chunk_index == 0:
            restore_status.assert_existing_objects_matched()

        if prediction.size == 0:
            raise RuntimeError(f"Model returned an empty chunk at offset {cursor}.")
        if prediction.size // 2 == 0:
            raise RuntimeError(f"Model prediction is too short to advance at offset {cursor}.")

        is_final = cursor + prediction.size >= target_samples
        boundary = "start" if chunk_index == 0 else ("end" if is_final else None)
        window = get_overlap_window(prediction, boundary=boundary)
        write_length = min(prediction.size, target_samples - cursor)
        output[cursor : cursor + write_length] += (
            prediction[:write_length] * window[:write_length]
        )
        weight_sum[cursor : cursor + write_length] += window[:write_length]

        step = prediction.size // 2
        chunk_trace.append(
            {
                "chunk_index": chunk_index,
                "start_sample": cursor,
                "input_samples_before_padding": int(original_chunk.size),
                "input_samples_to_model": int(chunk.size),
                "right_padded_samples": int(chunk.size - original_chunk.size),
                "reconstructed_samples": int(prediction.size),
                "written_samples": int(write_length),
                "overlap_add_boundary": boundary,
                "next_start_sample": cursor + step,
            }
        )
        cursor += step
        if is_final:
            break
    else:
        raise RuntimeError("Chunk loop exceeded its safety limit without covering the segment.")

    if np.any(weight_sum <= 0):
        uncovered = int(np.count_nonzero(weight_sum <= 0))
        raise RuntimeError(f"Overlap-add left {uncovered} samples without prediction coverage.")

    prediction_peak = float(np.max(np.abs(output)))
    if prediction_peak > 1e-6:
        output *= float(np.max(np.abs(mixture))) / prediction_peak
    output = output.astype(np.float32)

    sf.write(str(OUTPUT_WAV), output, sample_rate, subtype="PCM_16")
    saved_output, output_sr = sf.read(str(OUTPUT_WAV), dtype="float32")
    if output_sr != TARGET_SR or saved_output.shape != (target_samples,):
        raise RuntimeError("Saved estimate does not have the exact target duration and format.")

    baseline_estimate_mono = baseline_estimate[:, 0]
    mixture_sisdr = si_sdr(reference, mixture)
    baseline_sisdr = si_sdr(reference, baseline_estimate_mono)
    corrected_sisdr = si_sdr(reference, saved_output)
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))

    report = {
        "method": "official MTG feature extraction, clustering mask, VAD, and overlap-add",
        "chunk_advance": "half of each actual inverse-STFT reconstructed output length",
        "partial_windows": "right-padded to the 3-second model window to preserve valid U-Net dimensions; reconstructed output is cropped to remaining source samples",
        "evaluation": {
            "recording": "Brova Barama, Akkarai Sisters, Saraga v1.5",
            "start_seconds": 0.0,
            "duration_seconds": SEGMENT_SECONDS,
            "sample_rate": sample_rate,
            "samples": target_samples,
            "channels": 1,
            "same_reference_for_all_estimates": True,
            "mixture_si_sdr_db": mixture_sisdr,
            "original_baseline_si_sdr_db": baseline_sisdr,
            "coverage_corrected_si_sdr_db": corrected_sisdr,
            "original_baseline_change_vs_mixture_db": baseline_sisdr - mixture_sisdr,
            "coverage_corrected_change_vs_mixture_db": corrected_sisdr - mixture_sisdr,
            "baseline_runner_recorded_si_sdr_db": baseline_metrics["vocal_reference"][
                "si_sdr_vocal_estimate_db"
            ],
        },
        "baseline_estimate": describe(baseline_estimate_mono),
        "coverage_corrected_estimate": describe(saved_output),
        "full_interval_covered_by_positive_overlap_weights": bool(
            np.all(weight_sum > 0)
        ),
        "overlap_weight_min": float(np.min(weight_sum)),
        "overlap_weight_max": float(np.max(weight_sum)),
        "chunks": chunk_trace,
        "checkpoint_prefix": str(checkpoint_path),
        "checkpoint_restore_assertion": "assert_existing_objects_matched passed after first model call",
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved estimate: {OUTPUT_WAV}")
    print(f"Saved metrics: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
