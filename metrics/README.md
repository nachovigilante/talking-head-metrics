# Metrics

Metrics for speech-driven 3D facial animation on FLAME meshes (5023 vertices).

| Metric | What it compares | Reference | Details |
|---|---|---|---|
| LVE | Lip shape: maximal lip-vertex error per frame, averaged over frames | Ground-truth mesh | [lve/](lve/README.md) |
| FDD | Upper-face motion: difference in per-vertex temporal variability | Ground-truth mesh | [fdd/](fdd/README.md) |
| MOD | Mouth opening: mean difference in vertical lip extent | Ground-truth mesh | [mod/](mod/README.md) |
| BAS | Timing of head-motion beats | Ground-truth head motion | [bas/](bas/README.md) |
| PBAS | Timing of head-motion beats | Prosodic events of the speech | [pbas/](pbas/README.md) |

LVE, FDD and MOD come from MeshTalk, CodeTalker and DiffPoseTalk. BAS comes
from AI Choreographer, here with head-motion beats detected in SO(3). PBAS keeps
the BAS kernel and replaces the reference: the beats are scored against
prosodic events of the speech instead of against the ground-truth motion, so it
measures whether the head moves with the speech rather than whether it repeats
the recorded movement.

The FLAME model files are not included; see [models/data/README.md](../models/data/README.md).

## Python

```python
from metrics import LVECalculator, PBASCalculator

lve = LVECalculator(device="cpu")
value = lve.calculate_sequence_lve(pred_vertices, gt_vertices)   # (T, 5023, 3) tensors

pbas = PBASCalculator(device="cpu")
scores = pbas.calculate_pbas_from_files("prediction.npz", "speech.wav", motion_format="ensemble")
# {'omega': {'pitch_accent': ..., 'onset': ..., 'energy': ..., 'prominence': ...}, 'alpha': {...}, 'jerk': {...}}
```

## Command line

```bash
python -m metrics lve prediction.npz ground_truth.npz
python -m metrics bas prediction.npz ground_truth.npz
python -m metrics pbas prediction.npz speech.wav
python -m metrics mod --pairs pairs.txt --output mod.json   # one "prediction reference" pair per line
```

`--pred-format` and `--ref-format` select how FLAME parameter files are read
(`ensemble` for DiffPoseTalk outputs, `th1kh` for the TalkingHead-1KH
reconstructions, `artalk` for ARTalk exported as npz); `.npy` files are read as
vertices.

## Comparing models with and without head pose

The command line and the calculators use the meshes as given, with their global
head rotation. Models that do not predict pose (CodeTalker, MultiTalk) output
meshes in a fixed orientation, so comparing them with a rotated ground truth
mixes rigid head motion into the expression error. The evaluation in the thesis
removes the global rotation from every mesh before computing LVE, FDD and MOD;
that protocol is in `scripts/evaluation/run_canonical_metrics.py`.
