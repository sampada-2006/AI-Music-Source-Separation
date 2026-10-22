"""Run a bounded Saraga NMF follow-up on the complete Brova Barama triplet."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

import run_saraga_mridangam_demo as baseline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "saraga_mridangam_brova_barama"
OUTPUT_DIR = ROOT / "outputs" / "mridangam_brova_barama"
REPORT_PATH = ROOT / "SARAGA_MRIDANGAM_FOLLOWUP_REPORT.md"
RECORDING = "Brova Barama"
RECORDING_FOLDER = (
    "saraga1.5_carnatic/"
    "Akkarai Sisters at Arkay by Akkarai Sisters/"
    f"{RECORDING}/"
)

baseline.RECORDING = RECORDING
baseline.RECORDING_FOLDER = RECORDING_FOLDER
baseline.ARCHIVE_MEMBERS = {
    "mixture": RECORDING_FOLDER + RECORDING + ".mp3.mp3",
    "mridangam_left": RECORDING_FOLDER + RECORDING + ".multitrack-mridangam-left.mp3",
    "mridangam_right": RECORDING_FOLDER + RECORDING + ".multitrack-mridangam-right.mp3",
}
baseline.SOURCE_FILES = {
    "mixture": DATA_DIR / "saraga_brova_barama_mixture.mp3",
    "mridangam_left": DATA_DIR / "saraga_brova_barama_mridangam_left.mp3",
    "mridangam_right": DATA_DIR / "saraga_brova_barama_mridangam_right.mp3",
}
baseline.OUTPUTS = {
    "mixture": OUTPUT_DIR / "mridangam_mixture.wav",
    "reference": OUTPUT_DIR / "mridangam_reference.wav",
    "estimate": OUTPUT_DIR / "mridangam_separated.wav",
}
baseline.REPORT_PATH = REPORT_PATH


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def make_plot(audio_paths: dict[str, Path], destination: Path, sample_rate: int) -> None:
    labels = ("Mixture", "Reference", "Estimate")
    signals = []
    for label in labels:
        audio, rate = sf.read(audio_paths[label.lower()], dtype="float64")
        if rate != sample_rate or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError(f"Invalid plotted audio: {audio_paths[label]}")
        signals.append(audio)

    show_samples = min(len(signals[0]), 10 * sample_rate)
    times = np.arange(show_samples) / sample_rate
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    max_db = None
    spectra = []
    for audio in signals:
        frequencies, frame_times, zxx = signal.stft(
            audio[:show_samples],
            fs=sample_rate,
            nperseg=2048,
            noverlap=1024,
        )
        magnitude_db = 20 * np.log10(np.maximum(np.abs(zxx), 1e-12))
        spectra.append((frequencies, frame_times, magnitude_db))
        max_db = max(max_db if max_db is not None else -np.inf, float(magnitude_db.max()))

    for index, (label, audio) in enumerate(zip(labels, signals)):
        axes[0, index].plot(times, audio[:show_samples], linewidth=0.5)
        axes[0, index].set(title=f"{label} waveform (first 10 s)", xlabel="Time (s)")
        frequencies, frame_times, magnitude_db = spectra[index]
        image = axes[1, index].pcolormesh(
            frame_times,
            frequencies,
            magnitude_db,
            shading="auto",
            vmin=max_db - 80,
            vmax=max_db,
        )
        axes[1, index].set(
            title=f"{label} spectrogram",
            xlabel="Time (s)",
            ylabel="Frequency (Hz)",
            ylim=(0, 8000),
        )
    fig.colorbar(image, ax=axes[1, :], label="Magnitude (dB, shared scale)")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def make_report(
    source_stats: dict[str, dict[str, int | float]],
    compressed_bytes: int,
    range_bytes: int,
    scores: dict[str, float | int | str],
    audio_stats: dict[str, dict[str, float]],
    stereo_correlation: float,
    mono_rms_ratio: float,
) -> str:
    improvement = float(scores["estimate_si_sdr_db"]) - float(scores["mixture_si_sdr_db"])
    common_duration = min(float(item["duration"]) for item in source_stats.values())
    duration_spread = max(float(item["duration"]) for item in source_stats.values()) - common_duration
    return f"""# Saraga Mridangam Follow-up: Brova Barama

## Dataset and reference construction

- Saraga v1.5 Carnatic, **Brova Barama**, performed by Akkarai Sisters.
- Retrieved only the same-recording stereo mix and its `mridangam-left` and `mridangam-right` tracks from the Zenodo archive. Selected compressed audio totals {compressed_bytes:,} bytes; HTTP range traffic, including ZIP metadata, was {range_bytes:,} bytes. The 14.38 GB archive was not downloaded.
- Saraga documentation defines the left track as the mridangam bass-drum part and the right track as the treble-drum part. The reference here is their aligned sum for this mono baseline, not a pair of microphones. All three decoded files are 44.1 kHz; durations: mixture {source_stats['mixture']['duration']:.3f}s, bass {source_stats['mridangam_left']['duration']:.3f}s, treble {source_stats['mridangam_right']['duration']:.3f}s.
- The mix is {duration_spread * 1000:.1f} ms shorter than the longer stems; processing keeps the common initial {int(min(item['frames'] for item in source_stats.values())):,} samples and trims only the excess at the end. The generated held-out WAVs reproduce the corresponding source crops within floating-point write precision. A 100 ms-step RMS-envelope check found no gross offset; it cannot rule out a small sample-level delay.
- The mixture's stereo-channel correlation over the first 30 seconds is {stereo_correlation:.3f}; mono downmix retains {mono_rms_ratio:.1%} of stereo RMS. There is no evidence of strong cancellation from this downmix.
- The prior Shloka source tracks were decoded and reproduced exactly in their generated WAVs, but had a reference RMS {audio_stats['previous_reference']['rms']:.6f} versus mixture RMS {audio_stats['previous_mixture']['rms']:.6f}. The previously measured full-recording bass/treble RMS levels were about 0.00120/0.00276, versus 0.07469 for the mono mix. This supports unusually low source level in those files; it does not establish why the source was quiet or explain its perceived sound quality. Saraga's track descriptions also show the earlier report's “microphones” wording was inaccurate.

| Source track | Channels | Duration | Mono RMS | Peak |
|---|---:|---:|---:|---:|
| Stereo mix | {source_stats['mixture']['channels']} | {source_stats['mixture']['duration']:.3f}s | {source_stats['mixture']['rms']:.6f} | {source_stats['mixture']['peak']:.6f} |
| Mridangam bass | {source_stats['mridangam_left']['channels']} | {source_stats['mridangam_left']['duration']:.3f}s | {source_stats['mridangam_left']['rms']:.6f} | {source_stats['mridangam_left']['peak']:.6f} |
| Mridangam treble | {source_stats['mridangam_right']['channels']} | {source_stats['mridangam_right']['duration']:.3f}s | {source_stats['mridangam_right']['rms']:.6f} | {source_stats['mridangam_right']['peak']:.6f} |

## Method and evaluation

- A lightweight supervised NMF baseline reuses the existing implementation: four non-negative mridangam templates and four accompaniment-residual templates are learned from the first 70% of this recording. Non-negative activations are estimated on the final 30%; a soft magnitude mask is applied to the mixture STFT and inverse-transformed using mixture phase.
- This is a temporal holdout from one recording, not a test of generalization to new performances. The target reference is the aligned bass-plus-treble sum.
- Sample rate: {scores['sample_rate']} Hz; test duration: {scores['test_duration_seconds']:.3f}s ({scores['test_samples']:,} samples).
- SI-SDR vs reference: mixture {scores['mixture_si_sdr_db']:.2f} dB; estimate {scores['estimate_si_sdr_db']:.2f} dB; change {improvement:+.2f} dB.
- Relative magnitude reconstruction error: {scores['magnitude_relative_error']:.4f}; fit convergence warnings: {scores['fit_warnings']}.
- Output RMS: mixture {audio_stats['mixture']['rms']:.6f}, reference {audio_stats['reference']['rms']:.6f}, estimate {audio_stats['estimate']['rms']:.6f}. Output peaks: mixture {audio_stats['mixture']['peak']:.6f}, reference {audio_stats['reference']['peak']:.6f}, estimate {audio_stats['estimate']['peak']:.6f}.
- The reference peak exceeds 1.0 in floating-point WAV; the FLOAT encoding preserves it without integer PCM clipping. Normalize before converting to integer PCM.

SI-SDR is scale-invariant; it does not measure loudness matching or guarantee perceptually clean isolation. This NMF run **worsened** SI-SDR by {abs(improvement):.2f} dB, so it is not a successful separation result. I cannot independently audition the audio here, so no sound-quality claim is made. The comparison plot uses a shared spectrogram scale and is diagnostic, not proof of separation.

## Files

- Mixture: [WAV](outputs/mridangam_brova_barama/mridangam_mixture.wav)
- Reference: [WAV](outputs/mridangam_brova_barama/mridangam_reference.wav)
- Estimate: [WAV](outputs/mridangam_brova_barama/mridangam_separated.wav)
- Plot: [waveform and spectrogram comparison](outputs/mridangam_brova_barama/quality_plot.png)
- Downloaded source tracks: `data/saraga_mridangam_brova_barama/`

## Reproduction

From the project root, using the existing environment:

```powershell
D:\\Miniconda3\\envs\\ml\\python.exe src\\run_saraga_mridangam_brova_demo.py
```

The script refuses to overwrite existing paths and fetches only the three named ZIP members through bounded HTTP ranges.

## Limitations and next step

- No mridangam-specific pretrained model is established for this project; this is a basic NMF demonstration, not a production system.
- The audio is lossy MP3. Reusing mixture phase can cause artifacts. NMF templates can mix instruments and changing activity patterns.
- The test segment comes from the same performance as template training; generalization, clean separation, and BTP-level performance are not demonstrated.
- Saraga's repository identifies CC BY-NC 4.0, while Zenodo identifies CC BY-NC-SA 4.0. This project conservatively follows CC BY-NC-SA 4.0: attribute the dataset, use non-commercially, and share adaptations alike. Verify terms with the maintainers before any other use.
- Attribution: Bozkurt, B., Srinivasamurthy, A., Gulati, S., and Serra, X., *Saraga: research datasets of Indian Art Music*, version 1.5, Zenodo record 4301737, https://doi.org/10.5281/zenodo.4301737.
- Next useful step: listen to the aligned mixture, reference, and estimate here before deciding whether to pursue a different recording or method. The current numerical result does not justify further model tuning as if separation had succeeded.

Official references: [Saraga repository](https://github.com/MTG/saraga), [access instructions](https://mtg.github.io/saraga/access.html), [Zenodo record](https://zenodo.org/records/4301737), [repository license](https://github.com/MTG/saraga/blob/master/LICENSE.md).
"""


def main() -> None:
    baseline.ensure_destinations_are_new()
    previous_paths = {
        "mixture": ROOT / "outputs" / "mridangam_mixture.wav",
        "reference": ROOT / "outputs" / "mridangam_reference.wav",
    }
    previous_stats = {}
    for label, path in previous_paths.items():
        previous, _ = sf.read(path, dtype="float64")
        previous_stats[f"previous_{label}"] = {"rms": rms(previous)}

    with tempfile.TemporaryDirectory(prefix=".saraga-brova-stage-", dir=ROOT) as temp:
        stage_dir = Path(temp)
        staged_sources, compressed_bytes, range_bytes = baseline.stage_archive_members(stage_dir)
        source_stats = {}
        for name, path in staged_sources.items():
            audio, sample_rate = baseline.read_audio(path, name)
            info = sf.info(path)
            source_stats[name] = {
                "sample_rate": sample_rate,
                "channels": info.channels,
                "frames": info.frames,
                "duration": info.duration,
                "rms": rms(audio),
                "peak": float(np.max(np.abs(audio))),
            }
            print(
                f"{name}: {sample_rate} Hz, {info.channels} channel(s), "
                f"{info.frames:,} frames, {info.duration:.3f}s"
            )
        mixture_excerpt, excerpt_rate = sf.read(
            staged_sources["mixture"],
            frames=30 * baseline.SAMPLE_RATE_EXPECTED,
            dtype="float64",
            always_2d=True,
        )
        if excerpt_rate != baseline.SAMPLE_RATE_EXPECTED or mixture_excerpt.shape[1] != 2:
            raise ValueError("Expected a stereo 44.1 kHz mix for the channel diagnostic.")
        stereo_correlation = float(np.corrcoef(mixture_excerpt.T)[0, 1])
        mono_rms_ratio = rms(mixture_excerpt.mean(axis=1)) / rms(mixture_excerpt)

        staged_wavs, scores = baseline.run_demo(staged_sources, stage_dir)
        audio_stats = dict(previous_stats)
        for name, path in staged_wavs.items():
            audio, sample_rate = sf.read(path, dtype="float64")
            if audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError(f"Invalid generated audio: {path}")
            info = sf.info(path)
            if (
                sample_rate != int(scores["sample_rate"])
                or info.frames != int(scores["test_samples"])
                or info.channels != 1
            ):
                raise ValueError(f"Generated WAV metadata failed validation: {path}")
            audio_stats[name] = {
                "rms": rms(audio),
                "peak": float(np.max(np.abs(audio))),
            }
        plot_path = stage_dir / "quality_plot.png"
        make_plot(staged_wavs, plot_path, int(scores["sample_rate"]))
        report_path = stage_dir / REPORT_PATH.name
        report_path.write_text(
            make_report(
                source_stats,
                compressed_bytes,
                range_bytes,
                scores,
                audio_stats,
                stereo_correlation,
                mono_rms_ratio,
            ),
            encoding="utf-8",
        )

        files_to_commit = {
            **{
                baseline.SOURCE_FILES[name]: source_path
                for name, source_path in staged_sources.items()
            },
            **{
                baseline.OUTPUTS[name]: wav_path
                for name, wav_path in staged_wavs.items()
            },
            OUTPUT_DIR / "quality_plot.png": plot_path,
            REPORT_PATH: report_path,
        }
        baseline.commit_files(files_to_commit)

        change = float(scores["estimate_si_sdr_db"]) - float(scores["mixture_si_sdr_db"])
        print(
            f"SI-SDR: mixture={scores['mixture_si_sdr_db']:.2f} dB, "
            f"estimate={scores['estimate_si_sdr_db']:.2f} dB, change={change:+.2f} dB"
        )
        print(f"Saved validated outputs and report; source data: {DATA_DIR}")


if __name__ == "__main__":
    main()
