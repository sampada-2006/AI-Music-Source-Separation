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

Use a Python environment and install the dependencies listed in `requirements.txt`.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Experiments

Run the scripts from the project root after ensuring the required local audio files are available in the expected locations.

```bash
python src/run_saraga_mridangam_demo.py
python src/run_saraga_mridangam_brova_demo.py
python src/run_percussion_hpss_demo.py
```
