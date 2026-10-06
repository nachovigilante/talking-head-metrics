# MOD (Mouth Opening Difference)

MOD was introduced in DiffPoseTalk (Sun et al., SIGGRAPH Asia 2024), which
describes it as a metric "which measures the average difference in the size of
the mouth opening between the prediction and ground truth". As far as we know,
there is no published formula or reference implementation, and the paper does
not say which vertices define the opening, so the metric cannot be reproduced
exactly from the paper. This is our reconstruction:

```
open(M_t) = max_{v in S_M} y_{t,v} - min_{v in S_M} y_{t,v}
MOD = (1/T) * sum_t | open(M_t) - open(M̂_t) |
```

`y` is the vertical axis of the FLAME template and `S_M` is the `lips` mask of
`FLAME_masks.pkl` (254 vertices, the same set as LVE); the file has no separate
mouth mask. `open` is the vertical extent of the lips, which grows as the mouth
opens.

Because the vertex set and the definition of the opening are our choices, MOD
values from this implementation should not be compared with the numbers in the
DiffPoseTalk paper. Comparisons between models evaluated with this same
implementation are meaningful.

`open` is measured along the vertical axis of the mesh, so it is affected by
head pitch; the thesis computes MOD on meshes without global head rotation
(`scripts/evaluation/run_canonical_metrics.py`).

## Usage

```python
from metrics.mod import MODCalculator

calc = MODCalculator(device="cpu")
mod = calc.calculate_mod(pred_vertices, gt_vertices)
stats = calc.calculate_detailed_mod(pred_vertices, gt_vertices)   # per-frame openings
```
