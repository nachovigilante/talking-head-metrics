"""PBAS at 25 values of sigma (ablation of Section 5.3.1).

Motion beats and prosodic events are detected once per clip; the kernel width
only enters the final score, so PBAS is computed for every sigma from the same
beats. Real and shuffled pairs use the derangement of
scripts/evaluation/run_metrics.py. The script also estimates, for each
ground-truth clip, the dominant period of the head motion.

Output: results/sigma_sweep_results.csv, one row per (video_id, model, pair_type, sigma),
with the 12 pbas_<motion>_<prosodic> columns and motion_period_s.
"""
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from metrics.bas.so3_beat_detector import SO3BeatDetector
from metrics.pbas.pbas_calculator import MOTION_TYPES, PROSODIC_TYPES
from metrics.pbas.prosodic_extractor import ProsodicExtractor
from utils.protocol_utils import (
    ARTALK_DIR, DPT_DIR, GT_NPZ_DIR, MANIFEST_PATH, RESULTS_DIR, WAV_DIR,
    derangement, load_rot_artalk, load_rot_dpt, load_rot_gt,
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_CSV = RESULTS_DIR / "sigma_sweep_results.csv"
FPS = 25
SIGMAS = np.unique(np.round(np.geomspace(0.3, 25.0, 25), 3))  # frames


def pbas_all_sigmas(motion_beats, ref_beats, sigmas):
    """PBAS for several kernel widths at once (array of len(sigmas))."""
    motion_beats = np.asarray(motion_beats, dtype=float)
    ref_beats = np.asarray(ref_beats, dtype=float)
    if len(motion_beats) == 0 or len(ref_beats) == 0:
        return np.zeros(len(sigmas))
    min_dists = np.abs(motion_beats[:, None] - ref_beats[None, :]).min(axis=1)
    return np.exp(-(min_dists[:, None] ** 2) / (2 * sigmas[None, :] ** 2)).mean(axis=0)


def estimate_motion_period_s(rot, fps=FPS):
    """
    Dominant period, in seconds, of the head angular speed in the 0.5-10 Hz band.

    The speed is the norm of the frame-to-frame difference of the axis-angle
    rotation, which for the small head rotations of the dataset is close to the
    SO(3) angular speed.
    """
    if rot.shape[0] < 8:
        return np.nan
    speed = np.linalg.norm(np.diff(rot, axis=0), axis=1) * fps
    speed = speed - speed.mean()
    spectrum = np.abs(np.fft.rfft(speed))
    freqs = np.fft.rfftfreq(len(speed), d=1.0 / fps)
    band = (freqs >= 0.5) & (freqs <= 10.0)
    if not band.any() or spectrum[band].max() < 1e-9:
        return np.nan
    return float(1.0 / freqs[band][np.argmax(spectrum[band])])


def sweep_rows(video_id, model, pair_type, motion_beats, prosodic, period):
    rows = {s: {"video_id": video_id, "model": model, "pair_type": pair_type,
                "sigma": float(s), "motion_period_s": period} for s in SIGMAS}
    for bt in MOTION_TYPES:
        for pt in PROSODIC_TYPES:
            values = pbas_all_sigmas(motion_beats[bt]["beats"], prosodic[pt]["beats"], SIGMAS)
            for s, v in zip(SIGMAS, values):
                rows[s][f"pbas_{bt}_{pt}"] = float(v)
    return list(rows.values())


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    detect = SO3BeatDetector().detect_all_beats_so3
    extractor = ProsodicExtractor()
    print(f"sigmas ({len(SIGMAS)}): {SIGMAS.tolist()}")

    beats_cache, prosody_cache, period_cache, results = {}, {}, {}, []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Real pairs"):
        try:
            gt_rot = load_rot_gt(GT_NPZ_DIR / row.gt_npz, DEVICE)
            beats = {
                "gt": detect(gt_rot),
                "dpt": detect(load_rot_dpt(DPT_DIR / row.dpt_inference, DEVICE)),
                "artalk": detect(load_rot_artalk(ARTALK_DIR / row.artalk_inference, DEVICE)),
            }
        except Exception as e:
            print(f"skip {row.video_id}: {e}")
            continue
        try:
            prosodic = extractor.extract_all(str(WAV_DIR / row.wav))
        except Exception:
            prosodic = None

        beats_cache[row.video_id] = beats
        prosody_cache[row.video_id] = prosodic
        period_cache[row.video_id] = estimate_motion_period_s(gt_rot)
        if prosodic is None:
            continue
        for model in ("gt", "dpt", "artalk"):
            results += sweep_rows(row.video_id, model, "real", beats[model], prosodic,
                                  period_cache[row.video_id])

    video_ids = [v for v in manifest.video_id if v in beats_cache]
    partner_of = derangement(video_ids)
    for v in tqdm(video_ids, desc="Shuffled pairs"):
        partner_prosody = prosody_cache.get(partner_of[v])
        if partner_prosody is None:
            continue
        for model in ("gt", "dpt", "artalk"):
            results += sweep_rows(v, model, "shuffled", beats_cache[v][model], partner_prosody,
                                  period_cache[v])

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} {df.shape}")


if __name__ == "__main__":
    main()
