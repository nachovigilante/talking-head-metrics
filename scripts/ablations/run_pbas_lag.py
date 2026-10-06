"""Lag-corrected PBAS on the canonical set (ablation of Section 5.3.3).

For each clip and each of the 12 motion x prosodic combinations, estimate the
per-clip lag between the continuous motion magnitude and the continuous prosodic
signal (cross-correlation within ±480 ms), shift the motion beats by that lag and
score PBAS again. Runs on the human reference, DiffPoseTalk and ARTalk, in real
and shuffled pairs (same derangement as scripts/evaluation/run_metrics.py).

Output: results/pbas_lag_results.csv with one row per (video_id, model, pair_type) and,
for each combination, the standard PBAS, the lag-corrected PBAS and the lag in
frames (positive: motion follows speech). A summary is printed to stdout.
"""
import warnings

import pandas as pd
import torch
from tqdm import tqdm

from metrics.bas.so3_beat_detector import SO3BeatDetector
from metrics.pbas.pbas_calculator import PBASCalculator, MOTION_TYPES, PROSODIC_TYPES
from utils.protocol_utils import (
    ARTALK_DIR, DPT_DIR, GT_NPZ_DIR, MANIFEST_PATH, RESULTS_DIR, WAV_DIR,
    derangement, load_rot_artalk, load_rot_dpt, load_rot_gt,
)
from utils.stats_utils import rank_biserial

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_CSV = RESULTS_DIR / "pbas_lag_results.csv"

FPS = 25
MAX_LAG_FRAMES = 12  # ±480 ms
MODELS = ("gt", "dpt", "artalk")
COMBOS = [(bt, pt) for bt in MOTION_TYPES for pt in PROSODIC_TYPES]

beat_detector = SO3BeatDetector()
pbas_calc = PBASCalculator(device=DEVICE)


def score_row(motion_beats, prosodic):
    row = {}
    for bt, pt in COMBOS:
        mb = motion_beats[bt]["beats"]
        sb = prosodic[pt]["beats"]
        corrected, lag = pbas_calc.calculate_pbas_lag_corrected(
            mb, sb, motion_beats[bt]["signal"], prosodic[pt]["signal"], MAX_LAG_FRAMES
        )
        row[f"pbas_{bt}_{pt}"] = pbas_calc.calculate_pbas(mb, sb)
        row[f"pbaslag_{bt}_{pt}"] = corrected
        row[f"lag_{bt}_{pt}"] = lag
    return row


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"manifest: {len(manifest)} clips")

    beats_cache, prosody_cache, rows = {}, {}, []
    for r in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Real pairs"):
        try:
            rots = {
                "gt": load_rot_gt(GT_NPZ_DIR / r.gt_npz, DEVICE),
                "dpt": load_rot_dpt(DPT_DIR / r.dpt_inference, DEVICE),
                "artalk": load_rot_artalk(ARTALK_DIR / r.artalk_inference, DEVICE),
            }
            beats = {m: beat_detector.detect_all_beats_so3(rot) for m, rot in rots.items()}
            prosodic = pbas_calc.prosodic_extractor.extract_all(str(WAV_DIR / r.wav))
        except Exception as e:
            print(f"  skip {r.video_id}: {e}")
            continue
        beats_cache[r.video_id] = beats
        prosody_cache[r.video_id] = prosodic
        for m in MODELS:
            rows.append({"video_id": r.video_id, "partner_video_id": r.video_id,
                         "model": m, "pair_type": "real", **score_row(beats[m], prosodic)})

    video_ids = [v for v in manifest.video_id if v in beats_cache]
    partner_of = derangement(video_ids)
    for v in tqdm(video_ids, desc="Shuffled pairs"):
        p = partner_of[v]
        for m in MODELS:
            rows.append({"video_id": v, "partner_video_id": p, "model": m,
                         "pair_type": "shuffled", **score_row(beats_cache[v][m], prosody_cache[p])})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)")
    summarize(df)


def summarize(df):
    for m in MODELS:
        real = df[(df.model == m) & (df.pair_type == "real")].set_index("video_id")
        shuf = df[(df.model == m) & (df.pair_type == "shuffled")].set_index("video_id")
        common = real.index.intersection(shuf.index)
        real, shuf = real.loc[common], shuf.loc[common]
        print(f"\n=== {m} (n = {len(common)}) ===")
        print(f"{'combination':26s}{'PBAS':>8s}{'corr.':>8s}{'change':>9s}{'mean lag':>10s}"
              f"{'r std':>8s}{'r corr':>8s}")
        for bt, pt in COMBOS:
            std, cor = real[f"pbas_{bt}_{pt}"], real[f"pbaslag_{bt}_{pt}"]
            change = (cor.mean() - std.mean()) / std.mean() * 100
            lag_ms = real[f"lag_{bt}_{pt}"].mean() * 1000 / FPS
            r_std = rank_biserial(std.values, shuf[f"pbas_{bt}_{pt}"].values)
            r_cor = rank_biserial(cor.values, shuf[f"pbaslag_{bt}_{pt}"].values)
            print(f"{bt + ' x ' + pt:26s}{std.mean():8.3f}{cor.mean():8.3f}{change:+8.1f}%"
                  f"{lag_ms:+8.0f} ms{r_std:8.3f}{r_cor:8.3f}")


if __name__ == "__main__":
    main()
