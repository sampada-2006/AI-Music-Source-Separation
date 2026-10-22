# Indian Instrument Separation

A music source-separation project exploring methods to isolate Indian musical instruments, with an initial focus on separating **mridangam and percussion from mixed Carnatic music recordings**.

## Project Overview

Indian classical music recordings often contain multiple instruments and vocals mixed together. Separating an individual instrument can help with music analysis, practice, and audio processing.

This project explores signal-processing and machine-learning approaches for instrument separation and evaluates their performance using available reference recordings.

## Current Progress

The following approaches have been explored:

- **Supervised Non-negative Matrix Factorization (NMF):** A baseline approach using instrument and accompaniment templates with spectrogram-based masking.
- **Harmonic-Percussive Source Separation (HPSS):** A signal-processing baseline that separates harmonic and percussive components of audio.
- **Evaluation:** Signal-to-distortion ratio (SI-SDR) is used to compare estimated outputs against available reference recordings.

Initial experiments have produced mixed results. Performance varies across recordings, and reliable separation and perceptual quality still need further evaluation.

These experiments are baselines, not a claim that clean instrument separation has been achieved.

## Project Structure

```text
indian-instrument-separation/
├── src/
│   ├── run_saraga_mridangam_demo.py
│   ├── run_saraga_mridangam_brova_demo.py
│   └── run_percussion_hpss_demo.py
├── outputs/
│   └── mridangam_quality_check.md
├── SARAGA_MRIDANGAM_REPORT.md
├── SARAGA_MRIDANGAM_FOLLOWUP_REPORT.md
├── PERCUSSION_HARMONIC_REDUCTION_REPORT.md
├── .gitignore
└── README.md
```

The structure above lists the main experiment scripts and reports. Other local files, datasets, notebooks, and generated outputs may be kept outside the public repository.

## Dataset

Initial experiments use selected Saraga Carnatic music recordings and their available instrument-reference tracks.

Reference tracks may contain leakage or other musical components, so they should not automatically be treated as perfectly isolated instrument stems.

Dataset licensing and redistribution permissions must be checked before sharing recordings or derived audio publicly.

## Technologies and Methods

- Python
- NumPy and SciPy
- scikit-learn
- SoundFile
- Matplotlib
- Short-Time Fourier Transform (STFT)
- Non-negative Matrix Factorization (NMF)
- Harmonic-Percussive Source Separation (HPSS)
- SI-SDR-based evaluation

## Setup

There is no `requirements.txt` in this project. The MTG voice-separation baseline was verified with Python 3.11.15 and the following package versions:

**Windows PowerShell**

```powershell
python -m venv .venv_mtg
.\.venv_mtg\Scripts\Activate.ps1
python -m pip install tensorflow==2.15.0 keras==2.15.0 numpy==1.26.4 scipy==1.17.1 soundfile==0.14.0 librosa==0.11.0 scikit-learn==1.9.1 matplotlib==3.11.2 tqdm==4.70.1
```

The model checkpoint is distributed separately and is not installed by this command.

## Running the Experiments

Run the scripts from the project root after ensuring the required local audio files are available in the expected locations.

```bash
python src/run_saraga_mridangam_demo.py
python src/run_saraga_mridangam_brova_demo.py
python src/run_percussion_hpss_demo.py
```

### MTG Carnatic singing-voice baseline

The official MTG `saraga-8` cold-diffusion model estimates the singing voice. The accompaniment output is the residual (`mixture - vocal estimate`); this is **not** an mridangam-specific separator. No training is performed. The checkpoint files must already be present in `models/mtg/saraga-8/`, including `saraga-8.ckpt-1.index` and `saraga-8.ckpt-1.data-00000-of-00001`.

From the project root, run a short Saraga segment with its vocal reference:

```powershell
python .\src\mtg\run_mtg_baseline.py `
  --mixture .\data\saraga_mridangam_brova_barama\saraga_brova_barama_mixture.mp3 `
  --vocal-ref .\data\saraga_mridangam_brova_barama\saraga_brova_barama_vocal.mp3 `
  --output-dir .\outputs\mtg_baseline `
  --duration 3 `
  --clusters 5 `
  --scheduler 4
```

Inputs are downmixed to mono and resampled to 22,050 Hz before applying the official peak normalization and separation pipeline. The output directory contains the mixture, vocal estimate, accompaniment residual, optional reference, diagnostic plot, and metrics JSON. SI-SDR compares the estimate with the available Saraga vocal track; reference bleed and alignment limit how literally to interpret it. Mridangam references are not included unless explicitly passed to `--mridangam-left` and `--mridangam-right`.

A verified 3-second run produced vocal SI-SDR of **-3.62 dB** versus **-1.68 dB** for the mixture, a change of **-1.94 dB**. This short result did not improve SI-SDR and is not a general performance claim.
