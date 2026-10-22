"""Fetch one small Saraga mridangam multitrack and run a controlled NMF demo."""

from __future__ import annotations

import io
import os
import tempfile
import warnings
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
from scipy import signal
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_URL = (
    "https://zenodo.org/api/records/4301737/files/"
    "saraga1.5_carnatic.zip/content"
)
ARCHIVE_SIZE = 14_376_627_433
RECORDING = "Shloka Sri Ramachandra Shrita Parijata"
RECORDING_FOLDER = (
    "saraga1.5_carnatic/"
    "A V K Rajasimhan at Arkay by Angarai V K Rajasimhan/"
    f"{RECORDING}/"
)
ARCHIVE_MEMBERS = {
    "mixture": RECORDING_FOLDER + RECORDING + ".mp3.mp3",
    "mridangam_left": RECORDING_FOLDER + RECORDING + ".multitrack-mridangam-left.mp3",
    "mridangam_right": RECORDING_FOLDER + RECORDING + ".multitrack-mridangam-right.mp3",
}
SAMPLE_DIR = PROJECT_ROOT / "data" / "saraga_mridangam_sample"
OUTPUTS = {
    "mixture": PROJECT_ROOT / "outputs" / "mridangam_mixture.wav",
    "reference": PROJECT_ROOT / "outputs" / "mridangam_reference.wav",
    "estimate": PROJECT_ROOT / "outputs" / "mridangam_separated.wav",
}
SOURCE_FILES = {
    "mixture": SAMPLE_DIR / "saraga_shloka_mixture.mp3",
    "mridangam_left": SAMPLE_DIR / "saraga_shloka_mridangam_left.mp3",
    "mridangam_right": SAMPLE_DIR / "saraga_shloka_mridangam_right.mp3",
}
REPORT_PATH = PROJECT_ROOT / "SARAGA_MRIDANGAM_REPORT.md"

BLOCK_SIZE = 1024 * 1024
MAX_RANGE_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 10 * 1024 * 1024
SAMPLE_RATE_EXPECTED = 44_100
N_PER_SEGMENT = 2048
N_OVERLAP = 1024
TEMPLATES_PER_GROUP = 4
NMF_MAX_ITERATIONS = 100
ACTIVATION_ITERATIONS = 120
RANDOM_SEED = 0


class RemoteRangeFile:
    """Small seekable file facade that fetches bounded HTTP byte ranges."""

    def __init__(self, url: str, file_size: int) -> None:
        self.url = url
        self.file_size = file_size
        self.position = 0
        self.bytes_received = 0
        self._cache_start = -1
        self._cache = b""
        self._closed = False

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        return self._closed

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_position = offset
        elif whence == io.SEEK_CUR:
            new_position = self.position + offset
        elif whence == io.SEEK_END:
            new_position = self.file_size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if new_position < 0:
            raise ValueError("Cannot seek before the start of the archive.")
        self.position = new_position
        return self.position

    def _load_cache(self, start: int) -> None:
        end = min(start + BLOCK_SIZE, self.file_size) - 1
        expected_length = end - start + 1
        if self.bytes_received + expected_length > MAX_RANGE_BYTES:
            raise RuntimeError("Selective download exceeded its 32 MiB safety limit.")

        request = Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "Saraga-mridangam-subset-demo/1.0",
            },
        )
        try:
            response = urlopen(request, timeout=90)
        except (HTTPError, URLError) as error:
            raise RuntimeError(f"Could not retrieve a Saraga archive byte range: {error}") from error

        with response:
            if response.status != 206:
                raise RuntimeError(
                    f"Zenodo did not honor the byte range (HTTP {response.status}); "
                    "the archive body was not read."
                )
            content_range = response.headers.get("Content-Range", "")
            if content_range != f"bytes {start}-{end}/{self.file_size}":
                raise RuntimeError(f"Unexpected Zenodo Content-Range: {content_range!r}")
            data = response.read(expected_length + 1)
        if len(data) != expected_length:
            raise RuntimeError(
                f"Short range read: expected {expected_length} bytes, received {len(data)}."
            )
        self.bytes_received += len(data)
        self._cache_start = start
        self._cache = data

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on a closed archive.")
        if size is None or size < 0:
            size = self.file_size - self.position
        size = min(size, self.file_size - self.position)
        if size <= 0:
            return b""

        result = bytearray()
        remaining = size
        while remaining:
            cache_end = self._cache_start + len(self._cache)
            if not (self._cache_start <= self.position < cache_end):
                self._load_cache(self.position)
                cache_end = self._cache_start + len(self._cache)
            count = min(remaining, cache_end - self.position)
            start = self.position - self._cache_start
            result.extend(self._cache[start : start + count])
            self.position += count
            remaining -= count
        return bytes(result)

    def close(self) -> None:
        self._closed = True


def ensure_destinations_are_new() -> None:
    destinations = [*SOURCE_FILES.values(), *OUTPUTS.values(), REPORT_PATH]
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing project files:\n  " + "\n  ".join(existing)
        )


def stage_archive_members(stage_dir: Path) -> tuple[dict[str, Path], int, int]:
    remote_archive = RemoteRangeFile(ARCHIVE_URL, ARCHIVE_SIZE)
    staged_sources: dict[str, Path] = {}
    with zipfile.ZipFile(remote_archive) as archive:
        selected_infos = {}
        for name, member_path in ARCHIVE_MEMBERS.items():
            try:
                info = archive.getinfo(member_path)
            except KeyError as error:
                raise FileNotFoundError(
                    f"Expected Saraga archive member is missing: {member_path}"
                ) from error
            if info.file_size <= 0 or info.file_size > MAX_MEMBER_BYTES:
                raise ValueError(
                    f"Unexpected size for {member_path}: {info.file_size} bytes."
                )
            selected_infos[name] = info

        total_compressed_size = sum(info.compress_size for info in selected_infos.values())
        if total_compressed_size > MAX_RANGE_BYTES:
            raise ValueError(
                f"Selected audio is unexpectedly large: {total_compressed_size} bytes."
            )

        for name, info in selected_infos.items():
            staged_path = stage_dir / SOURCE_FILES[name].name
            written = 0
            with archive.open(info) as source, staged_path.open("xb") as destination:
                while True:
                    chunk = source.read(256 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > info.file_size:
                        raise IOError(f"Archive member exceeded its declared size: {info.filename}")
                    destination.write(chunk)
            if written != info.file_size:
                raise IOError(
                    f"Incomplete archive member {info.filename}: "
                    f"expected {info.file_size}, got {written}."
                )
            if staged_path.stat().st_size != info.file_size:
                raise IOError(f"Extracted file size does not match: {staged_path}")
            staged_sources[name] = staged_path

    return staged_sources, total_compressed_size, remote_archive.bytes_received


def read_audio(path: Path, label: str) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    except (RuntimeError, OSError) as error:
        raise RuntimeError(f"Could not decode {label} audio at {path}: {error}") from error
    if audio.shape[0] == 0 or audio.shape[1] not in (1, 2):
        raise ValueError(f"Unexpected {label} dimensions: {audio.shape}.")
    if not np.isfinite(audio).all():
        raise ValueError(f"{label} contains NaN or infinite values.")
    return audio.mean(axis=1), sample_rate


def make_stft(audio: np.ndarray, sample_rate: int):
    return signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=N_PER_SEGMENT,
        noverlap=N_OVERLAP,
        nfft=N_PER_SEGMENT,
        return_onesided=True,
        boundary="zeros",
        padded=True,
        scaling="spectrum",
    )


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    reference_energy = float(np.dot(reference, reference))
    if reference_energy <= 0 or not np.isfinite(reference_energy):
        raise ValueError("Reference audio must have finite, non-zero energy.")
    scale = float(np.dot(estimate, reference) / reference_energy)
    projected = scale * reference
    residual = estimate - projected
    noise_energy = float(np.dot(residual, residual))
    if noise_energy <= 0:
        return float("inf")
    signal_energy = float(np.dot(projected, projected))
    return 10.0 * np.log10(max(signal_energy, 1e-30) / noise_energy)


def write_float_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Refusing to save invalid waveform data to {path}.")
    sf.write(path, audio, sample_rate, format="WAV", subtype="FLOAT")
    info = sf.info(path)
    if (
        info.samplerate != sample_rate
        or info.channels != 1
        or info.frames != len(audio)
        or info.subtype != "FLOAT"
    ):
        raise IOError(f"Saved WAV metadata did not validate: {path}")
    check, check_rate = sf.read(path, dtype="float64", always_2d=True)
    if check_rate != sample_rate or not np.isfinite(check).all():
        raise IOError(f"Saved WAV cannot be decoded or contains invalid values: {path}")


def run_demo(
    staged_sources: dict[str, Path], stage_dir: Path
) -> tuple[dict[str, Path], dict[str, float | int | str]]:
    mixture, mixture_rate = read_audio(staged_sources["mixture"], "mixture")
    left, left_rate = read_audio(staged_sources["mridangam_left"], "mridangam-left")
    right, right_rate = read_audio(staged_sources["mridangam_right"], "mridangam-right")
    if not (mixture_rate == left_rate == right_rate):
        raise ValueError(
            f"Track sample rates differ: mix={mixture_rate}, left={left_rate}, right={right_rate}."
        )
    if mixture_rate != SAMPLE_RATE_EXPECTED:
        raise ValueError(
            f"Expected {SAMPLE_RATE_EXPECTED} Hz Saraga tracks, received {mixture_rate} Hz."
        )

    common_samples = min(len(mixture), len(left), len(right))
    if common_samples < mixture_rate * 8:
        raise ValueError("The aligned recording is too short for the train/test split.")
    duration_spread = (max(len(mixture), len(left), len(right)) - common_samples) / mixture_rate
    if duration_spread > 0.1:
        raise ValueError(
            f"Track durations differ by {duration_spread:.3f} seconds; refusing unsafe alignment."
        )

    mixture = mixture[:common_samples]
    reference = (left[:common_samples] + right[:common_samples]).astype(np.float64)
    split_sample = int(common_samples * 0.7)
    if split_sample < N_PER_SEGMENT or common_samples - split_sample < N_PER_SEGMENT:
        raise ValueError("Recording is too short to create the requested held-out segment.")

    train_mixture = mixture[:split_sample]
    train_reference = reference[:split_sample]
    train_accompaniment = train_mixture - train_reference
    test_mixture = mixture[split_sample:]
    test_reference = reference[split_sample:]
    sample_rate = mixture_rate

    train_frequencies, _, reference_train_stft = make_stft(train_reference, sample_rate)
    accompaniment_frequencies, _, accompaniment_train_stft = make_stft(
        train_accompaniment, sample_rate
    )
    test_frequencies, test_times, mixture_test_stft = make_stft(test_mixture, sample_rate)
    reference_frequencies, reference_times, _ = make_stft(
        test_reference, sample_rate
    )
    if not (
        np.array_equal(train_frequencies, accompaniment_frequencies)
        and np.array_equal(train_frequencies, test_frequencies)
        and np.array_equal(test_frequencies, reference_frequencies)
    ):
        raise ValueError("STFT frequency axes do not match.")
    if not np.array_equal(test_times, reference_times):
        raise ValueError("Mixture and reference STFT frame axes do not match.")

    reference_train_magnitude = np.abs(reference_train_stft).astype(np.float32)
    accompaniment_train_magnitude = np.abs(accompaniment_train_stft).astype(np.float32)
    mixture_test_magnitude = np.abs(mixture_test_stft).astype(np.float32)
    if not all(
        np.isfinite(magnitude).all() and np.all(magnitude >= 0)
        for magnitude in (
            reference_train_magnitude,
            accompaniment_train_magnitude,
            mixture_test_magnitude,
        )
    ):
        raise ValueError("A training or mixture magnitude spectrogram is invalid.")

    templates = []
    fit_warnings: dict[str, bool] = {}
    nmf_settings = {
        "n_components": TEMPLATES_PER_GROUP,
        "init": "nndsvda",
        "solver": "cd",
        "beta_loss": "frobenius",
        "max_iter": NMF_MAX_ITERATIONS,
        "tol": 1e-3,
        "random_state": RANDOM_SEED,
    }
    for name, magnitude in (
        ("mridangam", reference_train_magnitude),
        ("accompaniment residual", accompaniment_train_magnitude),
    ):
        model = NMF(**nmf_settings)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            model.fit_transform(magnitude.T)
        patterns = model.components_.astype(np.float32)
        norms = patterns.sum(axis=1, keepdims=True)
        if patterns.shape != (TEMPLATES_PER_GROUP, len(train_frequencies)):
            raise ValueError(f"Unexpected {name} template shape: {patterns.shape}.")
        if not np.isfinite(patterns).all() or np.any(norms <= 0):
            raise ValueError(f"Invalid {name} templates.")
        templates.append(patterns.T / norms.T)
        fit_warnings[name] = any(
            issubclass(item.category, ConvergenceWarning) for item in caught_warnings
        )
        print(
            f"{name.capitalize()} template fit: {model.n_iter_} iterations; "
            f"convergence warning={fit_warnings[name]}"
        )

    dictionary = np.concatenate(templates, axis=1)
    if dictionary.shape != (len(test_frequencies), 2 * TEMPLATES_PER_GROUP):
        raise ValueError(f"Unexpected dictionary shape: {dictionary.shape}.")
    if not np.isfinite(dictionary).all() or np.any(dictionary < 0):
        raise ValueError("The learned dictionary is not finite and non-negative.")

    rng = np.random.default_rng(RANDOM_SEED)
    activations = rng.random(
        (dictionary.shape[1], mixture_test_magnitude.shape[1]), dtype=np.float32
    ) + np.float32(1e-4)
    numerator = dictionary.T @ mixture_test_magnitude
    gram_matrix = dictionary.T @ dictionary
    epsilon = np.float32(1e-8)
    initial_error = float(
        np.linalg.norm(mixture_test_magnitude - dictionary @ activations)
        / max(np.linalg.norm(mixture_test_magnitude), 1e-12)
    )
    for _ in range(ACTIVATION_ITERATIONS):
        activations *= numerator / (gram_matrix @ activations + epsilon)
    if not np.isfinite(activations).all() or np.any(activations < 0):
        raise ValueError("Mixture activations are invalid.")

    mridangam_magnitude = dictionary[:, :TEMPLATES_PER_GROUP] @ activations[
        :TEMPLATES_PER_GROUP
    ]
    accompaniment_magnitude = dictionary[:, TEMPLATES_PER_GROUP:] @ activations[
        TEMPLATES_PER_GROUP:
    ]
    total_magnitude = mridangam_magnitude + accompaniment_magnitude
    mixture_relative_error = float(
        np.linalg.norm(mixture_test_magnitude - total_magnitude)
        / max(np.linalg.norm(mixture_test_magnitude), 1e-12)
    )
    mask = np.zeros_like(total_magnitude)
    modeled_bins = total_magnitude > epsilon
    np.divide(
        mridangam_magnitude,
        total_magnitude,
        out=mask,
        where=modeled_bins,
    )
    mask = np.clip(mask, 0.0, 1.0)
    estimated_stft = mask * mixture_test_stft
    _, estimate = signal.istft(
        estimated_stft,
        fs=sample_rate,
        window="hann",
        nperseg=N_PER_SEGMENT,
        noverlap=N_OVERLAP,
        nfft=N_PER_SEGMENT,
        input_onesided=True,
        boundary=True,
        scaling="spectrum",
    )
    extra_samples = len(estimate) - len(test_mixture)
    if extra_samples < 0 or extra_samples >= N_PER_SEGMENT - N_OVERLAP:
        raise ValueError(
            f"Unexpected estimate length {len(estimate)} for {len(test_mixture)} input samples."
        )
    estimate = estimate[: len(test_mixture)]
    if estimate.shape != test_reference.shape or not np.isfinite(estimate).all():
        raise ValueError("Estimated and reference waveforms have invalid dimensions or values.")

    scores = {
        "mixture_si_sdr_db": si_sdr(test_reference, test_mixture),
        "estimate_si_sdr_db": si_sdr(test_reference, estimate),
        "magnitude_relative_error": mixture_relative_error,
        "initial_random_activation_error": initial_error,
        "sample_rate": sample_rate,
        "train_duration_seconds": split_sample / sample_rate,
        "test_duration_seconds": len(test_mixture) / sample_rate,
        "test_samples": len(test_mixture),
        "fit_warnings": str(fit_warnings),
    }
    staged_wavs = {
        "mixture": stage_dir / OUTPUTS["mixture"].name,
        "reference": stage_dir / OUTPUTS["reference"].name,
        "estimate": stage_dir / OUTPUTS["estimate"].name,
    }
    write_float_wav(staged_wavs["mixture"], test_mixture, sample_rate)
    write_float_wav(staged_wavs["reference"], test_reference, sample_rate)
    write_float_wav(staged_wavs["estimate"], estimate, sample_rate)
    return staged_wavs, scores


def make_report(
    compressed_bytes: int,
    range_bytes: int,
    scores: dict[str, float | int | str],
) -> str:
    return f"""# Saraga Mridangam Separation Practice Experiment

## Data and access

- Recording: **{RECORDING}**, A V K Rajasimhan at Arkay by Angarai V K Rajasimhan.
- Source: Saraga v1.5 Carnatic, Zenodo record [4301737](https://zenodo.org/records/4301737), DOI [10.5281/zenodo.4301737](https://doi.org/10.5281/zenodo.4301737).
- Tracks: recording mix plus its `mridangam-left` and `mridangam-right` multitracks, all from the same recording.
- The 14,376,627,433-byte Carnatic ZIP was **not downloaded**. Zenodo HTTP range requests were used to inspect its ZIP directory and fetch only these three members (about {compressed_bytes:,} compressed audio bytes; {range_bytes:,} total ranged bytes including archive metadata).
- The Saraga repository license states CC BY-NC 4.0, while Zenodo record metadata states CC BY-NC-SA 4.0. This project follows the more restrictive **CC BY-NC-SA 4.0** terms: attribution, non-commercial use, and share-alike. Confirm with the maintainers for uses outside this scope.
- Attribution: Bozkurt, B., Srinivasamurthy, A., Gulati, S., and Serra, X., *Saraga: research datasets of Indian Art Music*, version 1.5, Zenodo 4301737. Recording and multitrack labels are retained in the downloaded filenames above.

## Method and commands

- No mridangam-specific pretrained model was found. The available official Carnatic separation model targets **singing voice**, not mridangam; a general Demucs drum stem is not instrument-specific and was not used.
- A small supervised NMF baseline was run with four mridangam templates and four accompaniment-residual templates. Templates were fit on the first 70% of the same recording; activations and estimates were evaluated on the final 30%. The estimated magnitude was soft-masked onto the mixture STFT using mixture phase.
- Interpreter: `D:\\Miniconda3\\envs\\ml\\python.exe` (NumPy, SciPy, scikit-learn, soundfile already available; no packages installed).
- Re-run from the project root: `D:\\Miniconda3\\envs\\ml\\python.exe src\\run_saraga_mridangam_demo.py`

## Outputs and evaluation

- Held-out mixture: `outputs/mridangam_mixture.wav`
- Held-out reference (left + right mridangam tracks): `outputs/mridangam_reference.wav`
- Held-out estimate: `outputs/mridangam_separated.wav`
- Sample rate: {scores['sample_rate']} Hz; held-out duration: {scores['test_duration_seconds']:.2f} s ({scores['test_samples']:,} samples).
- SI-SDR against the aligned reference: input mixture {scores['mixture_si_sdr_db']:.2f} dB; NMF estimate {scores['estimate_si_sdr_db']:.2f} dB.
- Fixed-dictionary relative magnitude error: {scores['magnitude_relative_error']:.4f}; initial random-activation error: {scores['initial_random_activation_error']:.4f}.
- NMF convergence-warning status: {scores['fit_warnings']}.

## Limitations

This is a practice experiment on one Saraga recording, with a within-recording temporal split; it does not show generalization to other concerts or real-world recordings. The source tracks are lossy MP3, and summed left/right microphones are a practical reference construction. Soft masking reuses mixture phase and can create artifacts. NMF errors and SI-SDR do not prove clean or production-quality separation. This is **not evidence that the BTP has achieved general Indian-instrument source separation**.

## Official references

- [Saraga repository and documentation](https://github.com/MTG/saraga)
- [Saraga access instructions](https://mtg.github.io/saraga/access.html)
- [Saraga Zenodo record](https://zenodo.org/records/4301737)
- [Saraga repository license](https://github.com/MTG/saraga/blob/master/LICENSE.md)
- [Carnatic voice separation research code](https://github.com/MTG/carnatic-separation-ismir23) (voice target only; not used)
"""


def commit_files(staged: dict[Path, Path]) -> None:
    placed: list[Path] = []
    created_directories: list[Path] = []
    try:
        for destination in staged:
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite: {destination}")
        for directory in {path.parent for path in staged}:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=False)
                created_directories.append(directory)
        for destination, staged_path in staged.items():
            os.rename(staged_path, destination)
            placed.append(destination)
    except Exception:
        for destination in reversed(placed):
            if destination.is_file():
                destination.unlink()
        for directory in reversed(created_directories):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        raise


def main() -> None:
    ensure_destinations_are_new()
    with tempfile.TemporaryDirectory(
        prefix=".saraga-mridangam-stage-", dir=PROJECT_ROOT
    ) as temporary_directory:
        stage_dir = Path(temporary_directory)
        staged_sources, compressed_bytes, range_bytes = stage_archive_members(stage_dir)

        source_stats = {}
        for name, staged_path in staged_sources.items():
            audio, sample_rate = read_audio(staged_path, name)
            info = sf.info(staged_path)
            source_stats[name] = {
                "bytes": staged_path.stat().st_size,
                "sample_rate": sample_rate,
                "channels": info.channels,
                "frames": info.frames,
                "duration": len(audio) / sample_rate,
            }
            print(
                f"{name}: {info.samplerate} Hz, {info.channels} channel(s), "
                f"{info.frames:,} frames, {info.duration:.3f} s, "
                f"{staged_path.stat().st_size:,} bytes"
            )

        staged_wavs, scores = run_demo(staged_sources, stage_dir)
        print(
            f"Train/test split: {scores['train_duration_seconds']:.2f} / "
            f"{scores['test_duration_seconds']:.2f} seconds"
        )
        print(
            f"SI-SDR: mixture={scores['mixture_si_sdr_db']:.2f} dB, "
            f"estimate={scores['estimate_si_sdr_db']:.2f} dB"
        )
        print(
            f"Magnitude relative error={scores['magnitude_relative_error']:.4f}; "
            f"initial activation error={scores['initial_random_activation_error']:.4f}"
        )

        report_path = stage_dir / REPORT_PATH.name
        report_path.write_text(
            make_report(compressed_bytes, range_bytes, scores),
            encoding="utf-8",
        )
        staged_files = {
            **{
                SOURCE_FILES[name]: staged_path
                for name, staged_path in staged_sources.items()
            },
            **{
                OUTPUTS[name]: staged_path for name, staged_path in staged_wavs.items()
            },
            REPORT_PATH: report_path,
        }
        commit_files(staged_files)

    print(f"Selective audio range bytes received: {range_bytes:,}")
    for name, path in SOURCE_FILES.items():
        print(f"Source {name}: {path} ({path.stat().st_size:,} bytes)")
    for name, path in OUTPUTS.items():
        info = sf.info(path)
        print(
            f"Output {name}: {path} ({path.stat().st_size:,} bytes), "
            f"{info.samplerate} Hz, {info.channels} channel(s), "
            f"{info.frames:,} frames"
        )
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
