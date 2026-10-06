# FDD (Upper-face Dynamics Deviation)

FDD was introduced in CodeTalker (Xing et al., CVPR 2023). The upper face is only
loosely tied to the speech and depends on the speaker's style, so instead of a
per-vertex error FDD compares how much each upper-face vertex varies over time
in the ground truth and in the prediction:

```
FDD = (1/|S_U|) * sum_{v in S_U} ( dyn(M_v) - dyn(M̂_v) )
```

CodeTalker describes `dyn` as "the standard deviation of the element-wise L2 norm
along the temporal axis". Here, for each vertex, `dyn` is the standard deviation
over time of the L2 distance between the vertex and its mean position over the
sequence:

```
d_t = || M_{t,v} - mean_t(M_{t,v}) ||_2
dyn(M_v) = std_t(d_t)          (with Bessel's correction)
```

`S_U` is the union of the `forehead` and `eye_region` masks of `FLAME_masks.pkl`
(827 vertices). FDD is signed: it is positive when the ground truth moves more
than the prediction and negative when the prediction moves more. When ranking
models, |FDD| measures how far a prediction is from the ground-truth dynamics.

## Usage

```python
from metrics.fdd import FDDCalculator

calc = FDDCalculator(device="cpu")
fdd = calc.calculate_fdd(pred_vertices, gt_vertices)    # order matters: prediction first
stats = calc.calculate_detailed_fdd(pred_vertices, gt_vertices)
```
