# BAS (Beat Alignment Score)

## Definition

BAS was introduced in AI Choreographer (Li et al., ICCV 2021) to measure whether
generated dance follows the music. Each motion beat is scored with a Gaussian
kernel on its distance to the nearest reference beat, and the score is the mean
over motion beats:

```
BAS = (1/n) * sum_{i=1}^{n} exp( -min_j |t_i - t_j|^2 / (2 sigma^2) )
```

`t_i` are the motion beats and `t_j` the reference beats, in frames. The score is
in [0, 1]. It is directional: it iterates over the motion beats, so a motion beat
far from every reference beat lowers the score, while reference beats without a
motion beat nearby do not. We use sigma = 3 frames at 25 fps (about 120 ms).

## Use in talking heads

SadTalker (Zhang et al., CVPR 2023) applies BAS to head motion against the audio
("we compute Beat Align Score as in Bailando"). DiffPoseTalk (Sun et al.,
SIGGRAPH Asia 2024) applies it "with a minor modification": the reference is the
head-motion beats of the ground truth, so the score measures how similar the
timing of the predicted motion is to the recorded one. `BASCalculator` computes
this second version; PBAS (`metrics/pbas`) keeps the kernel and uses the speech
as the reference.

## Head-motion beats

`SO3BeatDetector` derives the beats from the head rotation:

1. The axis-angle rotations are converted to rotation matrices and, optionally,
   smoothed (Savitzky-Golay, window 9, order 3, on sign-continuous quaternions).
2. The body-frame angular velocity is `omega_t = Log(R_t^T R_{t+1})`; the angular
   acceleration and jerk are its first and second finite differences. Working
   in SO(3) avoids the singularities and the wrap-around jumps of Euler angles,
   which the second and third derivatives amplify.
3. Beats are the peaks of the magnitude of each derivative above a percentile
   threshold (75 for omega and alpha, 80 for jerk), at least 10 frames apart.
4. Each finite difference lies half a frame after the previous one, so the beats
   are placed at their actual times: sample i of omega, alpha and jerk is at
   frame i + 0.5, i + 1 and i + 1.5.

AI Choreographer defines dance beats as the local minima of the kinetic
velocity, the moments when the dancer stops on a pose. For co-speech head
movement we take the maxima: the gesture literature anchors the timing of a
movement to its stroke, the most effortful phase, where the speed peaks. The
three derivatives are kept as three variants of the metric.

## Usage

```python
from metrics.bas import BASCalculator

calc = BASCalculator(device="cpu", sigma=3.0)
scores = calc.calculate_bas_from_files(
    "ground_truth.npz", "prediction.npz",
    reference_format="th1kh", prediction_format="ensemble",
)
# {'omega': ..., 'alpha': ..., 'jerk': ..., 'direction_change': ...}
```

`direction_change` beats (sudden changes in the axis of rotation) are also
reported; the thesis uses the omega, alpha and jerk beats.

## References

- Li, R., Yang, S., Ross, D. A., & Kanazawa, A. (2021). AI Choreographer: Music Conditioned 3D Dance Generation with AIST++. ICCV.
- Zhang, W., et al. (2023). SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation. CVPR.
- Sun, Z., et al. (2024). DiffPoseTalk: Speech-Driven Stylistic 3D Facial Animation and Head Pose Generation via Diffusion Models. ACM TOG (SIGGRAPH Asia).
