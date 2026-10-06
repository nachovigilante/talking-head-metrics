# LVE (Lip Vertex Error)

LVE was introduced in MeshTalk (Richard et al., ICCV 2021), which defines "the
lip error of a single frame to be the maximal ℓ2 error of all lip vertices" and
reports "the average over all frames". The authors use the maximum rather than
the mean because the upper lip and the mouth corners move much less than the
lower lip, and averaging over all lip vertices tends to hide inaccurate lip
shapes.

```
LVE = (1/T) * sum_t max_{v in S_L} || x_{t,v} - x̂_{t,v} ||_2
```

`S_L` is the `lips` mask of `FLAME_masks.pkl` (254 vertices). Values are in the
units of the mesh; FLAME meshes are in meters.

## Usage

```python
from metrics.lve import LVECalculator

calc = LVECalculator(device="cpu")
lve = calc.calculate_sequence_lve(pred_vertices, gt_vertices)       # (T, 5023, 3) tensors
lve = calc.calculate_lve_from_files("pred.npz", "gt.npz", format1="ensemble", format2="th1kh")
stats = calc.calculate_detailed_lve(pred_vertices, gt_vertices)    # per-frame values and summary
```

## Squared or unsquared distance

The implementation follows the paper and uses the Euclidean distance. In our
reading of the evaluation script that CodeTalker publishes (`main/cal_metric.py`),
which later work builds on, the per-vertex error is the squared distance. If that
reading is correct, LVE values reported with that script are not on the same
scale as values computed from the paper's definition, so absolute numbers should
only be compared when they come from the same implementation.
