"""Evaluate coverage-corrected MTG inference on the local Saraga Shloka pair."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm

DIAGNOSTIC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DIAGNOSTIC_DIR.parents[2]
DIAGNOSTICS_ROOT = PROJECT_ROOT / "outputs" / "mtg_diagnostics"
MTG_SOURCE = PROJECT_ROOT / "src" / "mtg"
if str(MTG_SOURCE) not in sys.path:
    sys.path.insert(0, str(MTG_SOURCE))
if str(DIAGNOSTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS_ROOT))

from config import Config
from diagnose_mtg_baseline import si_sdr_diagnostic
from model import DiffWave
from utils.signal_processing import get_overlap_window

import run_coverage_corrected as coverage

TARGET_SR = coverage.TARGET_SR
SEGMENT_SECONDS = coverage.SEGMENT_SECONDS
SAMPLE_COUNT = int(TARGET_SR * SEGMENT_SECONDS)
DATA_DIR = PROJECT_ROOT / "data" / "saraga_mridangam_sample"
MIXTURE_PATH = DATA_DIR / "saraga_shloka_mixture.mp3"
REFERENCE_PATH = DATA_DIR / "saraga_shloka_vocal.mp3"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "mtg" / "saraga-8" / "saraga-8.ckpt-1"
)
OUTPUT_WAV = DIAGNOSTIC_DIR / "saraga_shloka_0_3s_coverage_corrected_vocal.wav"
OUTPUT_JSON = DIAGNOSTIC_DIR / "saraga_shloka_0_3s_coverage_corrected_metrics.json"


def metric_report(reference: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    return si_sdr_diagnostic(reference, estimate)


def signal_stats(audio: np.ndarray) -> dict[str, Any]:
    audio64 = np.asarray(audio, dtype=np.float64)
    nonzero = np.flatnonzero(audio64 != 0)
    trailing_zeros = int(audio64.size - nonzero[-1] - 1) if nonzero.size else int(audio64.size)
    return {
        "samples": int(audio64.size),
        "duration_seconds": float(audio64.size / TARGET_SR),
        "sample_rate": TARGET_SR,
        "channels": 1,
        "finite": bool(np.isfinite(audio64).all()),
        "peak": float(np.max(np.abs(audio64))),
        "rms": float(np.sqrt(np.mean(audio64**2))),
        "trailing_exact_zero_samples": trailing_zeros,
        "trailing_exact_zero_seconds": float(trailing_zeros / TARGET_SR),
    }


def main() -> None:
    if OUTPUT_WAV.exists() or OUTPUT_JSON.exists():
        raise FileExistsError(
            f"Refusing to overwrite evaluation outputs: {OUTPUT_WAV} or {OUTPUT_JSON}"
        )
    for path in (MIXTURE_PATH, REFERENCE_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required paired source audio is missing: {path}")
    if not Path(str(CHECKPOINT_PATH) + ".index").is_file():
        raise FileNotFoundError(f"Checkpoint index is missing: {CHECKPOINT_PATH}.index")
    if not Path(str(CHECKPOINT_PATH) + ".data-00000-of-00001").is_file():
        raise FileNotFoundError(f"Checkpoint data shard is missing for {CHECKPOINT_PATH}")

    mixture, sample_rate = coverage.load_mono(MIXTURE_PATH)
    reference, reference_sample_rate = coverage.load_mono(REFERENCE_PATH)
    mixture = mixture[:SAMPLE_COUNT]
    reference = reference[:SAMPLE_COUNT]
    if mixture.size != SAMPLE_COUNT or reference.size != SAMPLE_COUNT:
        raise ValueError("Paired source files do not both contain the complete 3-second interval.")
    if sample_rate != TARGET_SR or reference_sample_rate != TARGET_SR:
        raise ValueError("Mixture and reference must both be resampled to 22,050 Hz.")
    if not np.isfinite(mixture).all() or not np.isfinite(reference).all():
        raise ValueError("Mixture or vocal reference contains non-finite samples.")

    mixture_peak = float(np.max(mixture))
    if not np.isfinite(mixture_peak) or mixture_peak <= 0:
        raise ValueError("Mixture must have a positive, finite peak amplitude.")
    normalized_mixture = mixture / mixture_peak

    config = Config()
    model = DiffWave(config.model)
    restore_status = model.restore(str(CHECKPOINT_PATH))
    restore_status.expect_partial()

    output = np.zeros(SAMPLE_COUNT, dtype=np.float64)
    weight_sum = np.zeros(SAMPLE_COUNT, dtype=np.float64)
    chunk_samples = int(coverage.CHUNK_SECONDS * TARGET_SR)
    chunk_trace: list[dict[str, Any]] = []
    cursor = 0
    for chunk_index in tqdm(range(100), desc="Shloka coverage-corrected inference"):
        if cursor >= SAMPLE_COUNT:
            break

        end = min(cursor + chunk_samples, SAMPLE_COUNT)
        original_chunk = normalized_mixture[cursor:end]
        chunk = np.pad(original_chunk, (0, chunk_samples - original_chunk.size))
        prediction = coverage.predict_chunk(model, config, chunk)
        if chunk_index == 0:
            restore_status.assert_existing_objects_matched()
        if prediction.size == 0 or prediction.size // 2 == 0:
            raise RuntimeError(f"Model returned an unusable chunk at sample {cursor}.")
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Model returned non-finite samples at chunk {chunk_index}.")

        is_final = cursor + prediction.size >= SAMPLE_COUNT
        boundary = "start" if chunk_index == 0 else ("end" if is_final else None)
        window = get_overlap_window(prediction, boundary=boundary)
        write_length = min(prediction.size, SAMPLE_COUNT - cursor)
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
                "next_start_sample": int(cursor + step),
            }
        )
        cursor += step
        if is_final:
            break
    else:
        raise RuntimeError("Chunk loop exceeded its safety limit.")

    if not np.isfinite(weight_sum).all() or np.any(weight_sum <= 0):
        raise RuntimeError("Overlap-add weights are invalid or leave uncovered samples.")
    if not np.isfinite(output).all():
        raise RuntimeError("Overlap-add produced non-finite samples.")

    prediction_peak = float(np.max(np.abs(output)))
    if prediction_peak > 1e-6:
        output *= float(np.max(np.abs(mixture))) / prediction_peak
    output = output.astype(np.float32)

    sf.write(str(OUTPUT_WAV), output, TARGET_SR, subtype="PCM_16")
    saved_output, output_sr = sf.read(str(OUTPUT_WAV), dtype="float32")
    if output_sr != TARGET_SR or saved_output.shape != (SAMPLE_COUNT,):
        raise RuntimeError("Saved WAV does not match the requested mono 3-second format.")
    if not np.isfinite(saved_output).all():
        raise RuntimeError("Saved WAV contains non-finite samples.")

    mixture_metric = metric_report(reference, mixture)
    corrected_metric = metric_report(reference, saved_output)
    if mixture_metric["value_db"] is None or corrected_metric["value_db"] is None:
        change_db = None
        change_status = "undefined"
        change_reason = "Change requires finite SI-SDR values for both signals."
    else:
        change_db = float(corrected_metric["value_db"] - mixture_metric["value_db"])
        change_status = "finite"
        change_reason = None

    correlation = float(
        np.dot(
            mixture.astype(np.float64) - np.mean(mixture, dtype=np.float64),
            reference.astype(np.float64) - np.mean(reference, dtype=np.float64),
        )
        / (
            np.linalg.norm(mixture.astype(np.float64) - np.mean(mixture, dtype=np.float64))
            * np.linalg.norm(reference.astype(np.float64) - np.mean(reference, dtype=np.float64))
        )
    )
    report = {
        "recording": "Saraga Shloka sample",
        "reference_representation": "local vocal-reference MP3, downmixed to mono and resampled to 22,050 Hz",
        "start_seconds": 0.0,
        "duration_seconds": SEGMENT_SECONDS,
        "sample_rate": TARGET_SR,
        "samples": SAMPLE_COUNT,
        "source_mixture": str(MIXTURE_PATH),
        "source_vocal_reference": str(REFERENCE_PATH),
        "original_baseline": {
            "available": False,
            "si_sdr_db": None,
            "reason": "No original MTG baseline estimate exists for this Shloka excerpt.",
        },
        "same_reference_and_interval_for_comparison": True,
        "preprocessing": {
            "channel_conversion": "arithmetic mean across channels",
            "resampling": "scipy.signal.resample_poly to 22,050 Hz",
            "model_input_normalization": "mixture divided by its maximum signed sample, matching the verified Brova diagnostic",
            "mixture_reference_zero_lag_correlation": correlation,
        },
        "metrics": {
            "mixture_vs_vocal_reference": mixture_metric,
            "coverage_corrected_vs_vocal_reference": corrected_metric,
            "coverage_corrected_change_vs_mixture_db": change_db,
            "change_status": change_status,
            "change_reason": change_reason,
        },
        "mixture": signal_stats(mixture),
        "vocal_reference": signal_stats(reference),
        "coverage_corrected_estimate": signal_stats(saved_output),
        "full_interval_covered_by_positive_overlap_weights": bool(
            np.all(weight_sum > 0)
        ),
        "overlap_weights_finite": bool(np.isfinite(weight_sum).all()),
        "overlap_weight_min": float(np.min(weight_sum)),
        "overlap_weight_max": float(np.max(weight_sum)),
        "chunks": chunk_trace,
        "checkpoint_prefix": str(CHECKPOINT_PATH),
        "checkpoint_restore_assertion": "assert_existing_objects_matched passed after first model call",
        "model_settings": {
            "sample_rate": TARGET_SR,
            "segment_seconds": SEGMENT_SECONDS,
            "chunk_seconds": coverage.CHUNK_SECONDS,
            "clusters": coverage.CLUSTERS,
            "scheduler": coverage.SCHEDULER,
            "model_configuration": config.model.__dict__,
        },
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved estimate: {OUTPUT_WAV}")
    print(f"Saved metrics: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
