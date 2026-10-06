# PBAS (Prosodic Beat Alignment Score)

PBAS measures whether head movements are synchronized with the speech they
accompany. It keeps the kernel of BAS and changes the reference: instead of the
head-motion beats of the ground truth, the motion beats are scored against
prosodic events detected in the audio.

```
PBAS = (1/n) * sum_{i=1}^{n} exp( -min_j |t_i^m - t_j^p|^2 / (2 sigma^2) )
```

`t_i^m` are the head-motion beats (see [bas/](../bas/README.md)) and `t_j^p` the
prosodic events, in video frames; sigma = 3 frames at 25 fps (about 120 ms). A
motion-only score can be high for a model that reproduces the recorded movement
without any relation to the audio; PBAS needs only the motion and the speech, so
it can also score the ground truth itself, which gives a human reference level.

With three motion beat types (omega, alpha, jerk) and four prosodic event types,
PBAS has 12 variants.

## Prosodic events

`ProsodicExtractor` works on 16 kHz audio and puts every signal on the video
frame grid (frame k at time k / 25 s):

| Event | Signal | Detection |
|---|---|---|
| `pitch_accent` | F0 from Praat autocorrelation (parselmouth), 75-600 Hz, linearly interpolated across unvoiced stretches | peaks at least 200 ms apart with prominence of at least 0.5 times the F0 standard deviation |
| `onset` | librosa onset strength, median aggregation, fmax 8 kHz | peaks above the 70th percentile, at least 200 ms apart |
| `energy` | librosa RMS, 80 ms window, 40 ms hop | peaks above the 70th percentile, at least 200 ms apart |
| `prominence` | 0.4 F0 + 0.3 Praat intensity + 0.3 RMS, each scaled to [0, 1] | peaks with prominence of at least 0.1, at least 200 ms apart |

Praat computes F0 and intensity on its own frame grid, offset by a few tens of
milliseconds; both are interpolated onto the video grid, and pitch accents are
placed at the time Praat reports for the peak.

## Usage

```python
from metrics.pbas import PBASCalculator

calc = PBASCalculator(device="cpu", sigma=3.0)
scores = calc.calculate_pbas_from_files("prediction.npz", "speech.wav", motion_format="ensemble")
# scores["alpha"]["onset"] is PBAS for angular-acceleration beats against onset peaks
```

From precomputed beats:

```python
motion = calc.beat_detector.detect_all_beats_so3(head_rotation)    # (T, 3) axis-angle
prosody = calc.prosodic_extractor.extract_all("speech.wav")
score = calc.calculate_pbas(motion["omega"]["beats"], prosody["pitch_accent"]["beats"])
```

## Lag-corrected variant

`calculate_pbas_lag_corrected` first estimates the per-clip lag between the
continuous motion magnitude and the continuous prosodic signal (the lag within
±480 ms that maximizes the absolute normalized cross-correlation, see
`optimal_lag`), shifts the motion beats by that lag and then scores them. It
separates the relative timing of the beats from a global offset between motion
and speech. On the human reference it does not improve the separation between
true and shuffled pairs (Section 5.3.3 of the thesis), so the main evaluation
uses the plain score; the variant is useful to characterize a model whose motion
leads or lags the speech.

## Validation

`scripts/evaluation/run_metrics.py` scores real pairs (motion and its own
speech) and shuffled pairs (motion and another clip's speech) for the ground
truth and two models, and `scripts/evaluation/run_stats.py` tests the
difference. Chapter 5 of the thesis reports the results.
