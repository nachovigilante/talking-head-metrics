# PBAS and an evaluation suite for speech-driven 3D facial animation

Code of the undergraduate thesis *Evaluación de métricas para animaciones faciales
3D guiadas por habla y su relación con la sincronía prosódica* (Ignacio Vigilante,
advised by Emmanuel Iarussi and Pablo Brusco, Departamento de Computación, FCEN,
Universidad de Buenos Aires, 2026).

The main contribution is PBAS (Prosodic Beat Alignment Score), a metric of
whether the head movement of a talking head is synchronized with the speech it
accompanies. It scores head-motion beats, detected from the SO(3) derivatives of
the head rotation, against prosodic events of the audio (pitch accents, onset
strength, RMS energy and a combined prominence). To apply it alongside the
metrics already used in the field, the repository also implements LVE, FDD, MOD
and a version of BAS for head pose, the evaluation protocol of the thesis on
TalkingHead-1KH, and the statistical analysis of four models (CodeTalker,
MultiTalk, DiffPoseTalk and ARTalk).

## Contents

| Path | |
|---|---|
| `metrics/` | The metrics, usable from Python or the command line ([metrics/README.md](metrics/README.md)) |
| `utils/`, `models/` | FLAME loading and decoding, SO(3) derivatives, statistics, shared protocol code |
| `scripts/evaluation/` | The main evaluation and its statistical tests |
| `scripts/ablations/` | The ablations of Section 5.3: kernel width, denoised audio and lag correction |
| `scripts/checks/` | The checks of Section 4.4: English subset and canonical orientation |
| `scripts/figures/` | The figures of the thesis made by this code; `inputs/` holds the word timings and the author's listening judgments of the prosodic-event figures |
| `scripts/dataset/` | Construction of the canonical clip list and language identification |
| `data/` | `manifest_canonical.csv`, the 1609 clips of the evaluation, and `language_ids_final.csv`, the language of each source video; the rest of the data goes here (see [Data](#data)) |
| `results/` | The results reported in the thesis; the figures are written to `results/figures/` |

## Installation

```bash
uv sync                      # Python 3.12; add --group transcription or --group denoise if needed
```

The FLAME model files are not redistributable; download them as described in
[models/data/README.md](models/data/README.md).

## Using the metrics

```bash
uv run python -m metrics pbas prediction.npz speech.wav
uv run python -m metrics lve prediction.npz ground_truth.npz
```

```python
from metrics import PBASCalculator

pbas = PBASCalculator(device="cpu")
scores = pbas.calculate_pbas_from_files("prediction.npz", "speech.wav", motion_format="ensemble")
```

See [metrics/README.md](metrics/README.md) for the input formats and for what each metric measures.

## Data

Of the data, only the ground-truth reconstructions are distributed (see below).
The scripts expect this layout under `data/` (or under the directory set in the
`DATA_DIR` environment variable):

```
TH1KH/wav/            speech of each clip, 16 kHz
TH1KH/wav_denoised/   the same audio after scripts/ablations/denoise_canonical.py (only for the noise ablation)
TH1KH/npz/            FLAME parameters of each clip reconstructed with SMIRK (the ground truth)
DiffPoseTalk/         DiffPoseTalk inferences (.npz)
ARTalk/               ARTalk inferences ((T, 106) tensors saved with torch.save)
MultiTalk/            MultiTalk inferences (vertices, (T, 15069) .npy)
CodeTalker/           CodeTalker inferences (vertices, (T, 15069) .npy)
```

- **Clips.** TalkingHead-1KH (Wang et al., CVPR 2021) provides talking-head clips
  from YouTube. `data/manifest_canonical.csv` lists the 1609 clips of the evaluation,
  one per source video; Section 4.4.2 of the thesis describes how they were
  selected. The audio and video filtering behind the selection is not part of
  this repository; the manifest fixes its result.
- **Audio.** Not distributed. The scripts of TalkingHead-1KH download the videos,
  split them and crop each clip; the crop step keeps only the video. Clip names
  follow TalkingHead-1KH (`<video>_<index>_S<first frame>_E<last frame>_L_T_R_B`),
  and the audio of each clip covers frames S to E of the split video, as 16 kHz
  mono WAV. For 95% of the clips its duration matches that stretch at 24, 25 or
  30 fps; audio cut this way may differ slightly from ours on the rest.
- **Ground truth.** FLAME parameters reconstructed from each clip with SMIRK
  (Retsinas et al., CVPR 2024) at 25 fps: shape, 50 expression coefficients,
  global rotation, jaw and eyelids. The evaluation uses the expression, the jaw
  and the global rotation, with a neutral shape for every sequence. The
  reconstructions of the 1609 clips are published as `th1kh_smirk_canonical.zip`
  in the releases of this repository, without the shape coefficients, which
  describe each person's face and are not used. From the repository root,
  `unzip th1kh_smirk_canonical.zip -d data` puts them in `data/TH1KH/npz/`.
- **Model inferences.** Not distributed. Each model was run on the audio of every
  clip with the checkpoint released by its authors and the settings of its demo,
  without any training or fine-tuning:

| Model | Code | Checkpoint and conditioning | Output |
|---|---|---|---|
| CodeTalker | [Doubiiu/CodeTalker](https://github.com/Doubiiu/CodeTalker) | `vocaset_stage1` and `vocaset_stage2`; speaker condition `FaceTalk_170725_00137_TA` | vertices, no head pose |
| MultiTalk | [kaist-ami/MultiTalk](https://github.com/kaist-ami/MultiTalk) | `stage1` and `stage2`; language condition `English` for every clip | vertices, no head pose |
| DiffPoseTalk | [DiffPoseTalk/DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk) | `head-SA-hubert-WM` at 110k iterations; shape and style `TH217`; audio and style guidance 1.15 and 3.0, dynamic threshold 0.99; one sample per clip | FLAME expression and pose |
| ARTalk | [xg-chu/ARTalk](https://github.com/xg-chu/ARTalk) | `ARTalk_wav2vec.pt`; style `natural_0` | FLAME expression and pose |

Decoding ARTalk's output needs ARTalk's own FLAME implementation: set
`ARTALK_REPO` to a clone of its repository.

## Reproducing the results

With the data in place, run from the repository root:

| Thesis | Command | Output |
|---|---|---|
| Tables 5.4 to 5.7, Sections 5.2.2 and 5.2.3 | `python -m scripts.evaluation.run_metrics` then `python -m scripts.evaluation.run_stats` | `results/metrics_results.csv`, `results/stats/` |
| Tables 5.1 and 5.2 | `python -m scripts.evaluation.run_canonical_metrics` then `python -m scripts.evaluation.run_canonical_stats` | `results/metrics_results_canonical.csv`, `results/metrics_canonical_stats.csv` |
| Table 5.8, Section 5.3.1 | `python -m scripts.ablations.run_sigma_sweep` then `python -m scripts.ablations.analyze_sigma_sweep` | `results/sigma_sweep_results.csv` |
| Table 5.9, Section 5.3.2 | `python -m scripts.ablations.denoise_canonical`, `python -m scripts.ablations.run_pbas_denoised`, `python -m scripts.ablations.compare_denoise_ablation` | `results/metrics_results_denoised.csv` |
| Section 5.3.3 | `python -m scripts.ablations.run_pbas_lag` | `results/pbas_lag_results.csv` |
| Section 4.4.2 (English subset) | `python -m scripts.checks.check_english_subset` | printed |
| Section 4.4.3 (canonical orientation) | `python -m scripts.checks.check_canonical_orientation` | printed |
| Figures 4.1 and 4.2 | `python -m scripts.figures.make_lve_mod_figures` | `results/figures/` |
| Figures 4.3 and 4.4 | `python -m scripts.figures.make_prosodic_events_figure --denoised 7-QzoS-dW-c_0011_S74_E269_L471_T0_R1111_B592 LylMvKFdwJU_0003_S769_E911_L471_T33_R775_B337` | `results/figures/` |
| Figure 5.1 | `python -m scripts.figures.make_xcorr_figure` | `results/figures/` |
| Figure 5.2 | `python -m scripts.figures.make_prosodic_events_figure --compare LylMvKFdwJU_0003_S769_E911_L471_T33_R775_B337` | `results/figures/` |

Prefix each command with `uv run`. `run_metrics` and `run_canonical_metrics` need
`ARTALK_REPO`. The sigma sweep is not included in `results/` because of its size;
`run_sigma_sweep` regenerates it. Table 5.3 compares with values reported in the
ARTalk paper and is not computed here.

## License

The code is released under the MIT license, except `models/flame.py` and
`models/lbs.py`, which keep the license of the Max Planck Institute for
Intelligent Systems (see [models/data/README.md](models/data/README.md)). The
reconstructions in the releases are distributed under CC BY 4.0; they derive from
TalkingHead-1KH videos published on YouTube under CC BY 3.0 by their authors. If
you appear in one of the videos and want its data removed, open an issue.
