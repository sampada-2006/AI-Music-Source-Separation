"""Prepare chunked, aligned training and validation data for the MTG Carnatic separation model.

Adheres to the official MTG dataset specification:
- Resampled to 22,050 Hz mono.
- Normalized and clamped to [-1.0, 1.0].
- Exported as 16-bit PCM WAV.
- Chunked into aligned fixed-length segments (default 6.0 seconds).
- Naming convention: <recording_id>_<chunk_idx>_mixture.wav, <recording_id>_<chunk_idx>_vocals.wav.
- Strict recording-level splitting between train and validation to prevent data leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.signal
import soundfile as sf

TARGET_SR = 22_050
DEFAULT_CHUNK_SECONDS = 6.0
BITS_PER_SAMPLE = "PCM_16"


def load_and_resample(
    file_path: Path, target_sr: int = TARGET_SR
) -> Tuple[np.ndarray, int]:
    """Load an audio file, downmix to mono, and resample to target_sr using high-quality polyphase filtering."""
    audio, orig_sr = sf.read(str(file_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if orig_sr != target_sr:
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        audio = scipy.signal.resample_poly(audio, up, down).astype(np.float32)

    return audio, target_sr


def clamp_and_normalize(audio: np.ndarray, max_val: float = 0.99) -> np.ndarray:
    """Normalize peak if exceeding 1.0, then clamp to [-1.0, 1.0]."""
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / (peak + 1e-6) * max_val
    return np.clip(audio, -1.0, 1.0)


def process_recording(
    recording_name: str,
    mix_path: Path,
    vocal_path: Path,
    percussion_paths: List[Path],
    output_dir: Path,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    split_name: str = "train",
) -> Dict[str, Any]:
    """Process a single Saraga recording into aligned chunks and compute metadata."""
    out_split_dir = output_dir / split_name
    out_split_dir.mkdir(parents=True, exist_ok=True)

    mix, sr = load_and_resample(mix_path)
    voc, _ = load_and_resample(vocal_path)

    perc = None
    if percussion_paths:
        p_signals = [load_and_resample(p)[0] for p in percussion_paths]
        min_p_len = min(len(s) for s in p_signals)
        perc = sum(s[:min_p_len] for s in p_signals)

    common_len = min(len(mix), len(voc))
    if perc is not None:
        common_len = min(common_len, len(perc))

    mix = clamp_and_normalize(mix[:common_len])
    voc = clamp_and_normalize(voc[:common_len])
    if perc is not None:
        perc = clamp_and_normalize(perc[:common_len])

    chunk_samples = int(chunk_seconds * sr)
    num_chunks = common_len // chunk_samples

    chunks_info = []
    clean_id = recording_name.lower().replace(" ", "_")

    for idx in range(num_chunks):
        start = idx * chunk_samples
        end = start + chunk_samples

        chunk_mix = mix[start:end]
        chunk_voc = voc[start:end]

        mix_rms = float(np.sqrt(np.mean(np.square(chunk_mix))))
        voc_rms = float(np.sqrt(np.mean(np.square(chunk_voc))))

        if mix_rms < 1e-4:
            continue

        mix_fn = f"{clean_id}_{idx:03d}_mixture.wav"
        voc_fn = f"{clean_id}_{idx:03d}_vocals.wav"

        sf.write(str(out_split_dir / mix_fn), chunk_mix, sr, subtype=BITS_PER_SAMPLE)
        sf.write(str(out_split_dir / voc_fn), chunk_voc, sr, subtype=BITS_PER_SAMPLE)

        chunk_meta = {
            "index": idx,
            "mixture_file": mix_fn,
            "vocals_file": voc_fn,
            "mix_rms": mix_rms,
            "voc_rms": voc_rms,
            "duration_s": chunk_seconds,
        }

        if perc is not None:
            chunk_perc = perc[start:end]
            perc_fn = f"{clean_id}_{idx:03d}_percussion.wav"
            sf.write(str(out_split_dir / perc_fn), chunk_perc, sr, subtype=BITS_PER_SAMPLE)
            chunk_meta["percussion_file"] = perc_fn
            chunk_meta["perc_rms"] = float(np.sqrt(np.mean(np.square(chunk_perc))))

        chunks_info.append(chunk_meta)

    corr_mix_voc = float(np.corrcoef(mix, voc)[0, 1])
    bleed_diag = {
        "corr_mix_voc": corr_mix_voc,
        "mix_total_rms": float(np.sqrt(np.mean(np.square(mix)))),
        "voc_total_rms": float(np.sqrt(np.mean(np.square(voc)))),
    }
    if perc is not None:
        corr_voc_perc = float(np.corrcoef(voc, perc)[0, 1])
        bleed_diag["corr_voc_perc"] = corr_voc_perc
        bleed_diag["perc_total_rms"] = float(np.sqrt(np.mean(np.square(perc))))

    return {
        "recording_name": recording_name,
        "split": split_name,
        "sample_rate": sr,
        "total_source_duration_s": common_len / sr,
        "chunk_duration_s": chunk_seconds,
        "total_chunks_extracted": len(chunks_info),
        "total_prepared_duration_s": len(chunks_info) * chunk_seconds,
        "bleed_diagnostics": bleed_diag,
        "chunks": chunks_info,
    }


def prepare_dataset(
    project_root: Path,
    output_dir: Path,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
) -> Dict[str, Any]:
    """Run full preparation pipeline on all available Saraga recordings."""
    output_dir.mkdir(parents=True, exist_ok=True)

    recordings = [
        {
            "name": "brova_barama",
            "split": "train",
            "mix": project_root / "data" / "saraga_mridangam_brova_barama" / "saraga_brova_barama_mixture.mp3",
            "voc": project_root / "data" / "saraga_mridangam_brova_barama" / "saraga_brova_barama_vocal.mp3",
            "perc": [
                project_root / "data" / "saraga_mridangam_brova_barama" / "saraga_brova_barama_mridangam_left.mp3",
                project_root / "data" / "saraga_mridangam_brova_barama" / "saraga_brova_barama_mridangam_right.mp3",
            ],
        },
        {
            "name": "shloka_sri_ramachandra",
            "split": "val",
            "mix": project_root / "data" / "saraga_mridangam_sample" / "saraga_shloka_mixture.mp3",
            "voc": project_root / "data" / "saraga_mridangam_sample" / "saraga_shloka_vocal.mp3",
            "perc": [
                project_root / "data" / "saraga_mridangam_sample" / "saraga_shloka_mridangam_left.mp3",
                project_root / "data" / "saraga_mridangam_sample" / "saraga_shloka_mridangam_right.mp3",
            ],
        },
    ]

    summary = {
        "pipeline": "MTG Carnatic Source Separation Data Preparation",
        "sample_rate": TARGET_SR,
        "channels": 1,
        "format": BITS_PER_SAMPLE,
        "split_method": "Recording-level partition (Zero temporal leakage across splits)",
        "train_recording": "Brova Barama (Akkarai Sisters)",
        "val_recording": "Shloka Sri Ramachandra Shrita Parijata (Angarai V K Rajasimhan)",
        "recordings": [],
    }

    total_train_s = 0.0
    total_val_s = 0.0
    total_train_chunks = 0
    total_val_chunks = 0

    for rec in recordings:
        print(f"Processing recording: {rec['name']} (split: {rec['split']})...")
        res = process_recording(
            recording_name=rec["name"],
            mix_path=rec["mix"],
            vocal_path=rec["voc"],
            percussion_paths=rec["perc"],
            output_dir=output_dir,
            chunk_seconds=chunk_seconds,
            split_name=rec["split"],
        )
        summary["recordings"].append(res)
        if rec["split"] == "train":
            total_train_s += res["total_prepared_duration_s"]
            total_train_chunks += res["total_chunks_extracted"]
        else:
            total_val_s += res["total_prepared_duration_s"]
            total_val_chunks += res["total_chunks_extracted"]

    summary["total_train_chunks"] = total_train_chunks
    summary["total_train_duration_s"] = total_train_s
    summary["total_val_chunks"] = total_val_chunks
    summary["total_val_duration_s"] = total_val_s

    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset preparation complete! Summary saved to: {summary_path}")
    print(f"Train: {total_train_chunks} chunks ({total_train_s:.1f}s = {total_train_s/60:.2f} min)")
    print(f"Val:   {total_val_chunks} chunks ({total_val_s:.1f}s = {total_val_s/60:.2f} min)")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Saraga training data for MTG model.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"D:\AI-Music-Source-Separation\indian-instrument-separation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\AI-Music-Source-Separation\indian-instrument-separation\data_mtg_prepared"),
    )
    parser.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    args = parser.parse_args()

    prepare_dataset(args.project_root, args.output_dir, args.chunk_seconds)
