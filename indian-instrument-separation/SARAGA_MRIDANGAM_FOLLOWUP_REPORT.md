# Saraga Mridangam Follow-up: Brova Barama

## Dataset and reference construction

- Saraga v1.5 Carnatic, **Brova Barama**, performed by Akkarai Sisters.
- Retrieved only the same-recording stereo mix and its `mridangam-left` and `mridangam-right` tracks from the Zenodo archive. Selected compressed audio totals 17,525,683 bytes; HTTP range traffic, including ZIP metadata, was 20,009,821 bytes. The 14.38 GB archive was not downloaded.
- Saraga documentation defines the left track as the mridangam bass-drum part and the right track as the treble-drum part. The reference here is their aligned sum for this mono baseline, not a pair of microphones. All three decoded files are 44.1 kHz; durations: mixture 263.564s, bass 263.582s, treble 263.582s.
- The mix is 18.4 ms shorter than the stems. The experiment uses the common initial 11,623,151 samples and trims only the excess at the end. Generated held-out WAVs match their source crops within floating-point write precision. A 100 ms-step RMS-envelope check found no gross offset; it cannot rule out a smaller sample-level delay.
- The stereo mix's channel correlation over its first 30 seconds is 0.940; mono downmix retains 98.4% of its stereo RMS. There is no evidence of strong cancellation from downmixing.
- The prior Shloka source tracks were decoded and reproduced exactly in their generated WAVs, but had a held-out reference RMS 0.002497 versus mixture RMS 0.054987. Their full-recording bass and treble RMS levels were about 0.00120 and 0.00276, versus 0.07469 for the mono mix. This supports unusually low source levels in those files; it does not establish why they were quiet or explain their perceived sound quality. Saraga's track descriptions also show the earlier report's “microphones” wording was inaccurate.

| Source track | Channels | Duration | Mono RMS | Peak |
|---|---:|---:|---:|---:|
| Stereo mix | 2 | 263.564s | 0.172315 | 0.987455 |
| Mridangam bass | 1 | 263.582s | 0.002428 | 0.059412 |
| Mridangam treble | 1 | 263.582s | 0.091338 | 1.010197 |

## Method and evaluation

- A lightweight supervised NMF baseline reuses the existing implementation: four non-negative mridangam templates and four accompaniment-residual templates are learned from the first 70% of this recording. Non-negative activations are estimated on the final 30%; a soft magnitude mask is applied to the mixture STFT and inverse-transformed using mixture phase.
- This is a temporal holdout from one recording, not a test of generalization to new performances. The target reference is the aligned bass-plus-treble sum.
- Sample rate: 44100 Hz; test duration: 79.069s (3,486,946 samples).
- SI-SDR vs reference: mixture -26.65 dB; estimate -31.75 dB; change -5.10 dB.
- Relative magnitude reconstruction error: 0.5480; fit convergence warnings: {'mridangam': False, 'accompaniment residual': False}.
- Output RMS: mixture 0.172643, reference 0.093660, estimate 0.086567. Output peaks: mixture 0.980999, reference 1.010403, estimate 0.765494.
- The summed reference is dominated by its treble track because the bass track is much quieter. Its peak exceeds 1.0 in floating-point WAV; FLOAT encoding preserves this value, but it can clip if converted to integer PCM without normalization.

SI-SDR is scale-invariant; it does not measure loudness matching or guarantee perceptually clean isolation. This run worsened SI-SDR by 5.10 dB, so it is not a successful separation result. I cannot independently audition the audio here, so no sound-quality claim is made. The comparison plot uses a shared spectrogram scale and is diagnostic, not proof of separation.

## Files

- Mixture: [WAV](outputs/mridangam_brova_barama/mridangam_mixture.wav)
- Reference: [WAV](outputs/mridangam_brova_barama/mridangam_reference.wav)
- Estimate: [WAV](outputs/mridangam_brova_barama/mridangam_separated.wav)
- Plot: [waveform and spectrogram comparison](outputs/mridangam_brova_barama/quality_plot.png)
- Downloaded source tracks: `data/saraga_mridangam_brova_barama/`

## Reproduction

From the project root, using the existing environment:

```powershell
D:\Miniconda3\envs\ml\python.exe src\run_saraga_mridangam_brova_demo.py
```

The script refuses to overwrite existing paths and fetches only the three named ZIP members through bounded HTTP ranges. The command above was run once to create these files; it will refuse to overwrite them on a second run. Use a clean project copy or new destination paths to reproduce without deleting these results.

## Limitations and next step

- this is a basic NMF demonstration, not a production system.
- The audio is lossy MP3. Reusing mixture phase can cause artifacts. NMF templates can mix instruments and changing activity patterns.
- The test segment comes from the same performance as template training; generalization, clean separation, and BTP-level performance are not demonstrated.
- Saraga's repository identifies CC BY-NC 4.0, while Zenodo identifies CC BY-NC-SA 4.0. This project conservatively follows CC BY-NC-SA 4.0: attribute the dataset, use non-commercially, and share adaptations alike. Verify terms with the maintainers before any other use.
- Attribution: Bozkurt, B., Srinivasamurthy, A., Gulati, S., and Serra, X., *Saraga: research datasets of Indian Art Music*, version 1.5, Zenodo record 4301737, https://doi.org/10.5281/zenodo.4301737.
- Next useful step: listen to the aligned mixture, reference, and estimate before choosing whether to pursue a different recording or method. The numerical result does not justify further tuning as though separation had succeeded.

Official references: [Saraga repository](https://github.com/MTG/saraga), [access instructions](https://mtg.github.io/saraga/access.html), [Zenodo record](https://zenodo.org/records/4301737), [repository license](https://github.com/MTG/saraga/blob/master/LICENSE.md).
