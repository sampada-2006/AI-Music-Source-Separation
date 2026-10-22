# Bounded Percussion/Harmonic Reduction Trial

## Summary and decision

This is a 10-second, no-training HPSS signal-processing trial, not a pretrained instrument separator. It separates transient-like spectral energy from sustained harmonic-like energy. Its percussive output had SI-SDR **-22.89 dB** against the available mridangam bass-plus-treble reference, compared with **-24.20 dB** for the mixture (change **+1.31 dB**). This metric alone cannot prove sound quality. No vocal-only reference is available, and I cannot audition audio here; therefore vocal reduction and perceptual percussion preservation are **not verified**.

## Dataset tracks and reference caveat

Saraga describes `mridangam-left` and `mridangam-right` as the mridangam bass-drum and treble-drum tracks. That identifies their intended part, but the documentation does not claim each file is leakage-free or a studio-clean isolated stem. The reference used here is their sample-aligned sum; it is a useful part-track reference, not guaranteed ground truth.

The older `SARAGA_MRIDANGAM_REPORT.md` called these “microphones.” That wording is not supported by Saraga's track documentation; the files are identified as bass- and treble-drum parts. The existing report was left unchanged.

For the full *Brova Barama* sources, the left/bass track RMS was 0.002428 and the right/treble track RMS was 0.091338; their ratio is about 31.5 dB. The mono mix RMS was 0.172315. This imbalance means the summed reference is dominated by the treble part; it does not demonstrate that the bass file is absent or that either track is clean.

The evaluation excerpt is the first 10 seconds of the existing held-out segment from **Brova Barama**, Akkarai Sisters, Saraga v1.5. It is drawn from the same performance as the reference.

## Pretrained model feasibility (checked; none downloaded)

| Candidate | Targets and domain | Requirements / size | License and decision |
|---|---|---|---|
| Demucs v4 `htdemucs` / `htdemucs_ft` | Four broad stems: vocals, drums, bass, other; trained on MUSDB-HQ plus 800 additional songs. “Drums” is not mridangam-specific and the training domain is not Carnatic. | Official code requires Python >=3.10, PyTorch >=2.1 and additional packages. A HEAD-only check of official checkpoint URLs returned 84,141,911 bytes (`htdemucs`) and 84,141,271 bytes (`htdemucs_ft`) per model. The current environment has no PyTorch, torchaudio, CUDA, or Demucs. | Repository code is MIT. The reviewed repo does not clearly establish a separate license for the pretrained weights, so no weights were downloaded. Not selected as an instrument-specific solution. |
| MTG Carnatic cold-diffusion model | Specifically targets Carnatic singing voice, so it could be relevant for vocal reduction; it does not produce mridangam as a dedicated target. | TensorFlow 2 project; its requirements pin old NumPy/librosa/Numba versions. The public README links a Google Drive feature-extraction U-Net checkpoint but does not publish a verifiable download size in the documentation. This Python 3.11 environment has NumPy 2.4.6 and no TensorFlow; compatibility and model setup are substantial. | Repository code is MIT; the linked weights' license is not clearly stated in the README. No weights downloaded. Not run. |
| Spleeter 2/4/5-stem models | Vocal/accompaniment or broad drums/bass/other categories, not Indian instrument classes. | TensorFlow-based; the current environment does not have TensorFlow. Public project docs did not establish a reliable size for a suitable checkpoint during this read-only check. | Code is MIT; no model assets were downloaded. Not selected. |

Demucs official sources: [repository and model description](https://github.com/adefossez/demucs), [requirements](https://github.com/adefossez/demucs/blob/main/pyproject.toml), [MIT code license](https://github.com/adefossez/demucs/blob/main/LICENSE), [published model files](https://github.com/adefossez/demucs/blob/main/demucs/remote/files.txt).

Carnatic model sources: [official repository](https://github.com/MTG/carnatic-separation-ismir23), [requirements](https://github.com/MTG/carnatic-separation-ismir23/blob/main/requirements.txt). Spleeter sources: [official repository](https://github.com/deezer/spleeter).

## HPSS method and measurements

The existing mono mixture and its matching bass-plus-treble reference were read from the same held-out crop and truncated to 10 seconds (441,000 samples at 44100 Hz). No source files were changed.

The experiment used SciPy STFT with a 2048-sample Hann window, 1536 overlap, and a 512-sample hop. Median filters estimated harmonic structure across 31 frames and percussive structure across 31 frequency bins. Squared soft masks were applied to the mixture complex STFT, then inverse-STFT reconstructed two estimates. No model weights, packages, or downloads were used.

| Signal | Peak | RMS | SI-SDR vs reference |
|---|---:|---:|---:|
| Mixture | 0.976805 | 0.201033 | -24.20 dB |
| Reference | 1.001726 | 0.095330 | — |
| HPSS percussive | 0.589045 | 0.076495 | -22.89 dB |
| HPSS harmonic | 0.785693 | 0.156038 | -27.07 dB |

The estimated harmonic branch has -2.20 dB less STFT energy than the input, by construction of the HPSS masks. This is **not a vocal-reduction metric**: sustained melodic instruments and mridangam resonances can also be harmonic, while vocals are not represented by an isolated reference. Magnitude reconstruction residual fraction: 6.692e-05. All generated signals were finite, aligned, and have exactly 441,000 samples. Float WAVs may exceed amplitude 1 without integer-PCM clipping; normalize before converting to integer PCM.

## Outputs and reproduction

- Mixture excerpt: [WAV](outputs/percussion_hpss_demo/mixture_10s.wav)
- Matching reference excerpt: [WAV](outputs/percussion_hpss_demo/mridangam_reference_10s.wav)
- HPSS percussive branch: [WAV](outputs/percussion_hpss_demo/percussive_hpss_10s.wav)
- HPSS harmonic branch: [WAV](outputs/percussion_hpss_demo/harmonic_hpss_10s.wav)
- Waveform/spectrogram comparison: [PNG](outputs/percussion_hpss_demo/comparison.png)

Run from the project root:

```powershell
D:\Miniconda3\envs\ml\python.exe src\run_percussion_hpss_demo.py
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
