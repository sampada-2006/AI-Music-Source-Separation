"""Run a small, no-training HPSS percussion emphasis diagnostic."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import median_filter


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "outputs" / "mridangam_brova_barama"
OUTPUT_DIR = ROOT / "outputs" / "percussion_hpss_demo"
REPORT_PATH = ROOT / "PERCUSSION_HARMONIC_REDUCTION_REPORT.md"
SAMPLE_RATE = 44_100
SEGMENT_SECONDS = 10
N_PER_SEGMENT = 2048
N_OVERLAP = 1536
HARMONIC_MEDIAN_FRAMES = 31
PERCUSSIVE_MEDIAN_BINS = 31
MASK_POWER = 2
EPSILON = 1e-12


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    reference_energy = float(reference @ reference)
    if reference_energy <= 0 or not np.isfinite(reference_energy):
        raise ValueError("Reference must have finite, non-zero energy.")
    projection = float(estimate @ reference) / reference_energy * reference
    residual = estimate - projection
    numerator = float(projection @ projection)
    denominator = float(residual @ residual)
    if denominator <= 0:
        return float("inf")
    return 10.0 * np.log10(max(numerator, EPSILON) / denominator)


def read_mono(path: Path, label: str) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} audio is missing: {path}")
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape[0] == 0:
        raise ValueError(
            f"Unexpected {label} audio: {sample_rate} Hz, shape {audio.shape}."
        )
    if not np.isfinite(audio).all():
        raise ValueError(f"{label} audio contains NaN or infinite values.")
    mono = audio.mean(axis=1)
    if mono.ndim != 1 or not np.isfinite(mono).all():
        raise ValueError(f"Could not safely convert {label} to mono.")
    return mono, sample_rate


def inverse_stft(
    stft: np.ndarray,
    sample_rate: int,
    expected_samples: int,
    label: str,
) -> np.ndarray:
    _, audio = signal.istft(
        stft,
        fs=sample_rate,
        window="hann",
        nperseg=N_PER_SEGMENT,
        noverlap=N_OVERLAP,
        nfft=N_PER_SEGMENT,
        input_onesided=True,
        boundary=True,
        scaling="spectrum",
    )
    length_difference = len(audio) - expected_samples
    hop_length = N_PER_SEGMENT - N_OVERLAP
    if 0 <= length_difference < hop_length:
        audio = audio[:expected_samples]
    elif length_difference != 0:
        raise ValueError(
            f"{label} inverse-STFT length is {len(audio)}; "
            f"expected {expected_samples} (difference {length_difference})."
        )
    if audio.shape != (expected_samples,) or not np.isfinite(audio).all():
        raise ValueError(f"{label} inverse-STFT output is invalid.")
    return audio


def hpss(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, mixture_stft = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=N_PER_SEGMENT,
        noverlap=N_OVERLAP,
        nfft=N_PER_SEGMENT,
        boundary="zeros",
        padded=True,
        scaling="spectrum",
    )
    magnitude = np.abs(mixture_stft)
    harmonic_magnitude = median_filter(
        magnitude, size=(1, HARMONIC_MEDIAN_FRAMES), mode="reflect"
    )
    percussive_magnitude = median_filter(
        magnitude, size=(PERCUSSIVE_MEDIAN_BINS, 1), mode="reflect"
    )
    harmonic_power = harmonic_magnitude**MASK_POWER
    percussive_power = percussive_magnitude**MASK_POWER
    denominator = harmonic_power + percussive_power + EPSILON
    harmonic_mask = harmonic_power / denominator
    percussive_mask = percussive_power / denominator
    harmonic_stft = harmonic_mask * mixture_stft
    percussive_stft = percussive_mask * mixture_stft
    if (
        harmonic_stft.shape != mixture_stft.shape
        or percussive_stft.shape != mixture_stft.shape
        or not np.isfinite(harmonic_stft).all()
        or not np.isfinite(percussive_stft).all()
    ):
        raise ValueError("HPSS masks produced invalid STFT dimensions or values.")
    harmonic = inverse_stft(
        harmonic_stft, sample_rate, len(audio), "Harmonic"
    )
    percussive = inverse_stft(
        percussive_stft, sample_rate, len(audio), "Percussive"
    )
    residual = mixture_stft - harmonic_stft - percussive_stft
    residual_fraction = float(
        np.linalg.norm(residual) / max(np.linalg.norm(mixture_stft), EPSILON)
    )
    return harmonic, percussive, np.asarray(
        [frequencies[-1], times[-1], residual_fraction], dtype=np.float64
    )


def write_float_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Refusing to write invalid audio: {path}")
    sf.write(path, audio, sample_rate, format="WAV", subtype="FLOAT")
    check, check_rate = sf.read(path, dtype="float64", always_2d=True)
    if (
        check_rate != sample_rate
        or check.shape != (len(audio), 1)
        or not np.isfinite(check).all()
    ):
        raise IOError(f"Generated WAV did not validate: {path}")


def make_plot(
    mixture: np.ndarray,
    reference: np.ndarray,
    harmonic: np.ndarray,
    percussive: np.ndarray,
    path: Path,
    sample_rate: int,
) -> None:
    labels = (
        ("Mixture", mixture),
        ("Reference (bass + treble tracks)", reference),
        ("HPSS harmonic", harmonic),
        ("HPSS percussive", percussive),
    )
    figure, axes = plt.subplots(4, 2, figsize=(13, 12), constrained_layout=True)
    shared_max_db = -np.inf
    spectra = []
    for _, audio in labels:
        frequencies, times, stft = signal.stft(
            audio,
            fs=sample_rate,
            nperseg=N_PER_SEGMENT,
            noverlap=N_OVERLAP,
        )
        magnitude_db = 20 * np.log10(np.maximum(np.abs(stft), EPSILON))
        spectra.append((frequencies, times, magnitude_db))
        shared_max_db = max(shared_max_db, float(magnitude_db.max()))
    image = None
    time_axis = np.arange(len(mixture)) / sample_rate
    for row, ((label, audio), (frequencies, times, magnitude_db)) in enumerate(
        zip(labels, spectra)
    ):
        axes[row, 0].plot(time_axis, audio, linewidth=0.35)
        axes[row, 0].set(title=f"{label} waveform", xlabel="Time (s)", ylabel="Amplitude")
        image = axes[row, 1].pcolormesh(
            times,
            frequencies,
            magnitude_db,
            shading="auto",
            vmin=shared_max_db - 80,
            vmax=shared_max_db,
        )
        axes[row, 1].set(
            title=f"{label} spectrogram",
            xlabel="Time (s)",
            ylabel="Frequency (Hz)",
            ylim=(0, 8000),
        )
    if image is not None:
        figure.colorbar(image, ax=axes[:, 1], label="Magnitude (dB; shared scale)")
    figure.savefig(path, dpi=130)
    plt.close(figure)


def make_report(
    results: dict[str, float | int | str],
    source_info: dict[str, dict[str, int | float]],
) -> str:
    return f"""# Bounded Percussion/Harmonic Reduction Trial

## Summary and decision

This is a 10-second, no-training HPSS signal-processing trial, not a pretrained instrument separator. It separates transient-like spectral energy from sustained harmonic-like energy. Its percussive output had SI-SDR **{float(results['percussive_si_sdr_db']):.2f} dB** against the available mridangam bass-plus-treble reference, compared with **{float(results['mixture_si_sdr_db']):.2f} dB** for the mixture (change **{float(results['si_sdr_change_db']):+.2f} dB**). This metric alone cannot prove sound quality. No vocal-only reference is available, and I cannot audition audio here; therefore vocal reduction and perceptual percussion preservation are **not verified**.

## Dataset tracks and reference caveat

Saraga describes `mridangam-left` and `mridangam-right` as the mridangam bass-drum and treble-drum tracks. That identifies their intended part, but the documentation does not claim each file is leakage-free or a studio-clean isolated stem. The reference used here is their sample-aligned sum; it is a useful part-track reference, not guaranteed ground truth.

The older `SARAGA_MRIDANGAM_REPORT.md` called these “microphones.” That wording is not supported by Saraga's track documentation; the files are identified as bass- and treble-drum parts. The existing report was left unchanged.

For the full *Brova Barama* sources, the left/bass track RMS was {float(source_info['bass_rms']):.6f} and the right/treble track RMS was {float(source_info['treble_rms']):.6f}; their ratio is about {float(source_info['treble_to_bass_db']):.1f} dB. The mono mix RMS was {float(source_info['mix_rms']):.6f}. This imbalance means the summed reference is dominated by the treble part; it does not demonstrate that the bass file is absent or that either track is clean.

The evaluation excerpt is the first {SEGMENT_SECONDS} seconds of the existing held-out segment from **Brova Barama**, Akkarai Sisters, Saraga v1.5. It is drawn from the same performance as the reference.

## Pretrained model feasibility (checked; none downloaded)

| Candidate | Targets and domain | Requirements / size | License and decision |
|---|---|---|---|
| Demucs v4 `htdemucs` / `htdemucs_ft` | Four broad stems: vocals, drums, bass, other; trained on MUSDB-HQ plus 800 additional songs. “Drums” is not mridangam-specific and the training domain is not Carnatic. | Official code requires Python >=3.10, PyTorch >=2.1 and additional packages. A HEAD-only check of official checkpoint URLs returned {int(results['demucs_checkpoint_bytes']):,} bytes (`htdemucs`) and {int(results['demucs_ft_checkpoint_bytes']):,} bytes (`htdemucs_ft`) per model. The current environment has no PyTorch, torchaudio, CUDA, or Demucs. | Repository code is MIT. The reviewed repo does not clearly establish a separate license for the pretrained weights, so no weights were downloaded. Not selected as an instrument-specific solution. |
| MTG Carnatic cold-diffusion model | Specifically targets Carnatic singing voice, so it could be relevant for vocal reduction; it does not produce mridangam as a dedicated target. | TensorFlow 2 project; its requirements pin old NumPy/librosa/Numba versions. The public README links a Google Drive feature-extraction U-Net checkpoint but does not publish a verifiable download size in the documentation. This Python 3.11 environment has NumPy {results['numpy_version']} and no TensorFlow; compatibility and model setup are substantial. | Repository code is MIT; the linked weights' license is not clearly stated in the README. No weights downloaded. Not run. |
| Spleeter 2/4/5-stem models | Vocal/accompaniment or broad drums/bass/other categories, not Indian instrument classes. | TensorFlow-based; the current environment does not have TensorFlow. Public project docs did not establish a reliable size for a suitable checkpoint during this read-only check. | Code is MIT; no model assets were downloaded. Not selected. |

Demucs official sources: [repository and model description](https://github.com/adefossez/demucs), [requirements](https://github.com/adefossez/demucs/blob/main/pyproject.toml), [MIT code license](https://github.com/adefossez/demucs/blob/main/LICENSE), [published model files](https://github.com/adefossez/demucs/blob/main/demucs/remote/files.txt).

Carnatic model sources: [official repository](https://github.com/MTG/carnatic-separation-ismir23), [requirements](https://github.com/MTG/carnatic-separation-ismir23/blob/main/requirements.txt). Spleeter sources: [official repository](https://github.com/deezer/spleeter).

## HPSS method and measurements

The existing mono mixture and its matching bass-plus-treble reference were read from the same held-out crop and truncated to {SEGMENT_SECONDS} seconds ({int(results['samples']):,} samples at {SAMPLE_RATE} Hz). No source files were changed.

The experiment used SciPy STFT with a {N_PER_SEGMENT}-sample Hann window, {N_OVERLAP} overlap, and a {N_PER_SEGMENT - N_OVERLAP}-sample hop. Median filters estimated harmonic structure across {HARMONIC_MEDIAN_FRAMES} frames and percussive structure across {PERCUSSIVE_MEDIAN_BINS} frequency bins. Squared soft masks were applied to the mixture complex STFT, then inverse-STFT reconstructed two estimates. No model weights, packages, or downloads were used.

| Signal | Peak | RMS | SI-SDR vs reference |
|---|---:|---:|---:|
| Mixture | {float(results['mixture_peak']):.6f} | {float(results['mixture_rms']):.6f} | {float(results['mixture_si_sdr_db']):.2f} dB |
| Reference | {float(results['reference_peak']):.6f} | {float(results['reference_rms']):.6f} | — |
| HPSS percussive | {float(results['percussive_peak']):.6f} | {float(results['percussive_rms']):.6f} | {float(results['percussive_si_sdr_db']):.2f} dB |
| HPSS harmonic | {float(results['harmonic_peak']):.6f} | {float(results['harmonic_rms']):.6f} | {float(results['harmonic_si_sdr_db']):.2f} dB |

The estimated harmonic branch has {float(results['harmonic_energy_attenuation_db']):.2f} dB less STFT energy than the input, by construction of the HPSS masks. This is **not a vocal-reduction metric**: sustained melodic instruments and mridangam resonances can also be harmonic, while vocals are not represented by an isolated reference. Magnitude reconstruction residual fraction: {float(results['stft_residual_fraction']):.3e}. All generated signals were finite, aligned, and have exactly {int(results['samples']):,} samples. Float WAVs may exceed amplitude 1 without integer-PCM clipping; normalize before converting to integer PCM.

## Outputs and reproduction

- Mixture excerpt: [WAV](outputs/percussion_hpss_demo/mixture_10s.wav)
- Matching reference excerpt: [WAV](outputs/percussion_hpss_demo/mridangam_reference_10s.wav)
- HPSS percussive branch: [WAV](outputs/percussion_hpss_demo/percussive_hpss_10s.wav)
- HPSS harmonic branch: [WAV](outputs/percussion_hpss_demo/harmonic_hpss_10s.wav)
- Waveform/spectrogram comparison: [PNG](outputs/percussion_hpss_demo/comparison.png)

Run from the project root:

```powershell
D:\\Miniconda3\\envs\\ml\\python.exe src\\run_percussion_hpss_demo.py
```

The script refuses to overwrite its report or output directory.

## Limitations

- HPSS is a generic transient/harmonic filter. Carnatic vocal, melodic instruments, and mridangam all have mixed harmonic and percussive characteristics. A percussive mask may remove useful mridangam tone or retain voice transients.
- The percussive output must not be called a mridangam stem. No clean vocal stem exists here, the part tracks may contain bleed, and no listening check was available. Consequently the requested audible vocal reduction is unverified.
- The existing supervised NMF result remains unsuccessful (SI-SDR worsened by 5.10 dB); this trial does not change that conclusion or establish successful BTP separation.
- Saraga v1.5 audio is licensed CC BY-NC 4.0 by its repository license; the Zenodo record has separately been reported as CC BY-NC-SA 4.0. Conservatively follow the more restrictive terms pending clarification, attribute Saraga, use non-commercially, and share adaptations alike.
- Saraga attribution: Bozkurt, B., Srinivasamurthy, A., Gulati, S., and Serra, X., *Saraga: research datasets of Indian Art Music*, version 1.5, Zenodo record 4301737, https://doi.org/10.5281/zenodo.4301737. Official references: [Saraga](https://mtg.github.io/saraga/), [Zenodo](https://zenodo.org/records/4301737), [license](https://github.com/MTG/saraga/blob/master/LICENSE.md).

## Conclusion and most promising next option

This is a reproducible diagnostic, not a verified improvement. The most promising pretrained candidate for reducing Carnatic vocals is the MTG Carnatic voice-separation model, but its old TensorFlow stack and checkpoint-license/size uncertainty must be resolved before using it; it does not promise a clean mridangam stem. No further experiment was started.
"""


def main() -> None:
    if OUTPUT_DIR.exists() or REPORT_PATH.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing HPSS demo output or report."
        )

    mixture_path = INPUT_DIR / "mridangam_mixture.wav"
    reference_path = INPUT_DIR / "mridangam_reference.wav"
    mixture_full, sample_rate = read_mono(mixture_path, "mixture")
    reference_full, reference_rate = read_mono(reference_path, "mridangam reference")
    if sample_rate != reference_rate or len(mixture_full) != len(reference_full):
        raise ValueError(
            "Mixture and mridangam reference must have identical sample rate and length."
        )
    samples = SEGMENT_SECONDS * sample_rate
    if len(mixture_full) < samples:
        raise ValueError(f"Input must contain at least {SEGMENT_SECONDS} seconds.")
    mixture = mixture_full[:samples]
    reference = reference_full[:samples]

    harmonic, percussive, diagnostics = hpss(mixture, sample_rate)
    frequencies_last, times_last, residual_fraction = diagnostics
    if frequencies_last <= 0 or times_last <= 0:
        raise ValueError("STFT did not produce valid frequency/time axes.")
    mixture_si_sdr = si_sdr(reference, mixture)
    percussive_si_sdr = si_sdr(reference, percussive)
    harmonic_si_sdr = si_sdr(reference, harmonic)

    _, _, mixture_stft = signal.stft(
        mixture,
        fs=sample_rate,
        window="hann",
        nperseg=N_PER_SEGMENT,
        noverlap=N_OVERLAP,
        nfft=N_PER_SEGMENT,
        boundary="zeros",
        padded=True,
        scaling="spectrum",
    )
    _, _, harmonic_stft = signal.stft(
        harmonic,
        fs=sample_rate,
        window="hann",
        nperseg=N_PER_SEGMENT,
        noverlap=N_OVERLAP,
        nfft=N_PER_SEGMENT,
        boundary="zeros",
        padded=True,
        scaling="spectrum",
    )
    mixture_spectral_energy = float(np.sum(np.abs(mixture_stft) ** 2))
    harmonic_spectral_energy = float(np.sum(np.abs(harmonic_stft) ** 2))
    harmonic_energy_attenuation_db = 10 * np.log10(
        max(harmonic_spectral_energy, EPSILON)
        / max(mixture_spectral_energy, EPSILON)
    )

    def file_stats(audio: np.ndarray) -> tuple[float, float]:
        return float(np.max(np.abs(audio))), rms(audio)

    mixture_peak, mixture_rms = file_stats(mixture)
    reference_peak, reference_rms = file_stats(reference)
    percussive_peak, percussive_rms = file_stats(percussive)
    harmonic_peak, harmonic_rms = file_stats(harmonic)

    source_dir = ROOT / "data" / "saraga_mridangam_brova_barama"
    source_names = {
        "mixture": "saraga_brova_barama_mixture.mp3",
        "bass": "saraga_brova_barama_mridangam_left.mp3",
        "treble": "saraga_brova_barama_mridangam_right.mp3",
    }
    source_levels = {}
    for name, filename in source_names.items():
        audio, rate = read_mono(source_dir / filename, f"source {name}")
        if rate != sample_rate:
            raise ValueError(f"Source {name} sample rate differs from the output.")
        source_levels[name] = rms(audio)
    source_info = {
        "mix_rms": source_levels["mixture"],
        "bass_rms": source_levels["bass"],
        "treble_rms": source_levels["treble"],
        "treble_to_bass_db": 20
        * np.log10(max(source_levels["treble"], EPSILON) / max(source_levels["bass"], EPSILON)),
    }
    results: dict[str, float | int | str] = {
        "samples": samples,
        "mixture_peak": mixture_peak,
        "reference_peak": reference_peak,
        "percussive_peak": percussive_peak,
        "harmonic_peak": harmonic_peak,
        "mixture_rms": mixture_rms,
        "reference_rms": reference_rms,
        "percussive_rms": percussive_rms,
        "harmonic_rms": harmonic_rms,
        "mixture_si_sdr_db": mixture_si_sdr,
        "percussive_si_sdr_db": percussive_si_sdr,
        "harmonic_si_sdr_db": harmonic_si_sdr,
        "si_sdr_change_db": percussive_si_sdr - mixture_si_sdr,
        "harmonic_energy_attenuation_db": harmonic_energy_attenuation_db,
        "stft_residual_fraction": residual_fraction,
        "demucs_checkpoint_bytes": 84_141_911,
        "demucs_ft_checkpoint_bytes": 84_141_271,
        "numpy_version": np.__version__,
    }

    with tempfile.TemporaryDirectory(prefix=".percussion-hpss-stage-", dir=ROOT) as temp:
        stage_root = Path(temp)
        stage_output = stage_root / OUTPUT_DIR.name
        stage_output.mkdir()
        write_float_wav(stage_output / "mixture_10s.wav", mixture, sample_rate)
        write_float_wav(
            stage_output / "mridangam_reference_10s.wav", reference, sample_rate
        )
        write_float_wav(
            stage_output / "percussive_hpss_10s.wav", percussive, sample_rate
        )
        write_float_wav(
            stage_output / "harmonic_hpss_10s.wav", harmonic, sample_rate
        )
        make_plot(
            mixture,
            reference,
            harmonic,
            percussive,
            stage_output / "comparison.png",
            sample_rate,
        )
        staged_report = stage_root / REPORT_PATH.name
        staged_report.write_text(
            make_report(results, source_info), encoding="utf-8"
        )
        if OUTPUT_DIR.exists() or REPORT_PATH.exists():
            raise FileExistsError(
                "An HPSS destination appeared during execution; refusing to overwrite."
            )
        os.rename(stage_output, OUTPUT_DIR)
        os.rename(staged_report, REPORT_PATH)

    print(
        f"Mixture SI-SDR={mixture_si_sdr:.2f} dB; "
        f"HPSS percussive SI-SDR={percussive_si_sdr:.2f} dB; "
        f"change={percussive_si_sdr - mixture_si_sdr:+.2f} dB"
    )
    print(
        f"HPSS harmonic-branch energy change={harmonic_energy_attenuation_db:.2f} dB; "
        "this is not a vocal-isolation score."
    )
    print(f"Created: {OUTPUT_DIR} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
