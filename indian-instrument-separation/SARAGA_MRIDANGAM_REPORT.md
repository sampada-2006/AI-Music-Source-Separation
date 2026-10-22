# Saraga Mridangam Separation Practice Experiment

## Data and access

- Recording: **Shloka Sri Ramachandra Shrita Parijata**, A V K Rajasimhan at Arkay by Angarai V K Rajasimhan.
- Source: Saraga v1.5 Carnatic, Zenodo record [4301737](https://zenodo.org/records/4301737), DOI [10.5281/zenodo.4301737](https://doi.org/10.5281/zenodo.4301737).
- Tracks: recording mix plus its `mridangam-left` and `mridangam-right` multitracks, all from the same recording.
- The 14,376,627,433-byte Carnatic ZIP was **not downloaded**. Zenodo HTTP range requests were used to inspect its ZIP directory and fetch only these three members (about 1,576,679 compressed audio bytes; 4,281,181 total ranged bytes including archive metadata).
- The Saraga repository license states CC BY-NC 4.0, while Zenodo record metadata states CC BY-NC-SA 4.0. This project follows the more restrictive **CC BY-NC-SA 4.0** terms: attribution, non-commercial use, and share-alike. Confirm with the maintainers for uses outside this scope.
- Attribution: Bozkurt, B., Srinivasamurthy, A., Gulati, S., and Serra, X., *Saraga: research datasets of Indian Art Music*, version 1.5, Zenodo 4301737. Recording and multitrack labels are retained in the downloaded filenames above.

## Method and commands

- No mridangam-specific pretrained model was found. The available official Carnatic separation model targets **singing voice**, not mridangam; a general Demucs drum stem is not instrument-specific and was not used.
- A small supervised NMF baseline was run with four mridangam templates and four accompaniment-residual templates. Templates were fit on the first 70% of the same recording; activations and estimates were evaluated on the final 30%. The estimated magnitude was soft-masked onto the mixture STFT using mixture phase.
- Interpreter: `D:\Miniconda3\envs\ml\python.exe` (NumPy, SciPy, scikit-learn, soundfile already available; no packages installed).
- Re-run from the project root: `D:\Miniconda3\envs\ml\python.exe src\run_saraga_mridangam_demo.py`

## Outputs and evaluation

- Held-out mixture: `outputs/mridangam_mixture.wav`
- Held-out reference (left + right mridangam tracks): `outputs/mridangam_reference.wav`
- Held-out estimate: `outputs/mridangam_separated.wav`
- Sample rate: 44100 Hz; held-out duration: 8.31 s (366,253 samples).
- SI-SDR against the aligned reference: input mixture -8.75 dB; NMF estimate -7.09 dB.
- Fixed-dictionary relative magnitude error: 0.5298; initial random-activation error: 10.9840.
- NMF convergence-warning status: {'mridangam': False, 'accompaniment residual': False}.

## Limitations

This is a practice experiment on one Saraga recording, with a within-recording temporal split; it does not show generalization to other concerts or real-world recordings. The source tracks are lossy MP3, and summed left/right microphones are a practical reference construction. Soft masking reuses mixture phase and can create artifacts. NMF errors and SI-SDR do not prove clean or production-quality separation. This is **not evidence that the BTP has achieved general Indian-instrument source separation**.

## Official references

- [Saraga repository and documentation](https://github.com/MTG/saraga)
- [Saraga access instructions](https://mtg.github.io/saraga/access.html)
- [Saraga Zenodo record](https://zenodo.org/records/4301737)
- [Saraga repository license](https://github.com/MTG/saraga/blob/master/LICENSE.md)
- [Carnatic voice separation research code](https://github.com/MTG/carnatic-separation-ismir23) (voice target only; not used)
