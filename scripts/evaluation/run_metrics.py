"""Main evaluation on the canonical set: LVE, FDD, MOD, BAS and PBAS, real and shuffled pairs.

For each clip of data/manifest_canonical.csv:
  - DiffPoseTalk and ARTalk: LVE, FDD and MOD against the ground truth (meshes
    with their global head rotation), BAS against the ground-truth head motion,
    and PBAS against the clip's own speech;
  - ground truth: PBAS against its own speech, the human reference.

Shuffled pairs pair each clip's motion with another clip's ground-truth motion
(BAS) and speech (PBAS), with the fixed derangement of utils.protocol_utils.

Output: results/metrics_results.csv, one row per (video_id, model, pair_type).
The canonical-orientation version of LVE, FDD and MOD, with the four models, is
run_canonical_metrics.py.
"""
import warnings

import pandas as pd
import torch
from tqdm import tqdm

from metrics import BASCalculator, FDDCalculator, LVECalculator, MODCalculator, PBASCalculator
from utils.flame_utils import get_flame_model
from utils.protocol_utils import (
    ARTALK_DIR, DPT_DIR, GT_NPZ_DIR, MANIFEST_PATH, RESULTS_DIR, WAV_DIR,
    bas_scores, decode_artalk, decode_flame_npz, derangement, load_artalk_flame,
    load_rot_artalk, load_rot_dpt, load_rot_gt, pbas_scores,
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = RESULTS_DIR / "metrics_results.csv"
NAN = float("nan")


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    flame = get_flame_model(device=DEVICE)
    flame_artalk = load_artalk_flame(DEVICE)
    lve_calc = LVECalculator(device=DEVICE)
    fdd_calc = FDDCalculator(device=DEVICE)
    mod_calc = MODCalculator(device=DEVICE)
    bas_calc = BASCalculator(device=DEVICE)
    pbas_calc = PBASCalculator(device=DEVICE)
    detect = bas_calc.beat_detector.detect_all_beats_so3

    results, beats_cache, prosody_cache = [], {}, {}
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Real pairs"):
        try:
            gt_v = decode_flame_npz(flame, GT_NPZ_DIR / row.gt_npz, DEVICE, "th1kh")
            dpt_v = decode_flame_npz(flame, DPT_DIR / row.dpt_inference, DEVICE, "ensemble")
            art_v = decode_artalk(flame_artalk, ARTALK_DIR / row.artalk_inference, DEVICE)
        except Exception as e:
            print(f"skip {row.video_id}: {e}")
            continue

        beats = {
            "gt": detect(load_rot_gt(GT_NPZ_DIR / row.gt_npz, DEVICE)),
            "dpt": detect(load_rot_dpt(DPT_DIR / row.dpt_inference, DEVICE)),
            "artalk": detect(load_rot_artalk(ARTALK_DIR / row.artalk_inference, DEVICE)),
        }
        try:
            prosodic = pbas_calc.prosodic_extractor.extract_all(str(WAV_DIR / row.wav))
        except Exception as e:
            print(f"no prosody for {row.video_id}: {e}")
            prosodic = None
        beats_cache[row.video_id] = beats
        prosody_cache[row.video_id] = prosodic

        for model, pred_v in (("dpt", dpt_v), ("artalk", art_v)):
            T = min(pred_v.shape[0], gt_v.shape[0])
            with torch.no_grad():
                lve = lve_calc.calculate_sequence_lve(pred_v[:T], gt_v[:T])
                fdd = fdd_calc.calculate_fdd(pred_v[:T], gt_v[:T])
                mod = mod_calc.calculate_mod(pred_v[:T], gt_v[:T])
            results.append({
                "video_id": row.video_id, "partner_video_id": row.video_id,
                "clip_stem": row.clip_stem, "model": model, "pair_type": "real", "frames": T,
                "lve": lve, "fdd": fdd, "mod": mod,
                **bas_scores(beats[model], beats["gt"]),
                **pbas_scores(beats[model], prosodic),
            })
        # For the ground truth only PBAS is meaningful on real pairs (BAS against itself is 1).
        results.append({
            "video_id": row.video_id, "partner_video_id": row.video_id,
            "clip_stem": row.clip_stem, "model": "gt", "pair_type": "real", "frames": gt_v.shape[0],
            "lve": NAN, "fdd": NAN, "mod": NAN, "bas_omega": NAN, "bas_alpha": NAN, "bas_jerk": NAN,
            **pbas_scores(beats["gt"], prosodic),
        })

    video_ids = [v for v in manifest.video_id if v in beats_cache]
    partner_of = derangement(video_ids)
    clip_stem = dict(zip(manifest.video_id, manifest.clip_stem))
    for v in tqdm(video_ids, desc="Shuffled pairs"):
        p = partner_of[v]
        for model in ("gt", "dpt", "artalk"):
            motion = beats_cache[v][model]
            results.append({
                "video_id": v, "partner_video_id": p, "clip_stem": clip_stem[v],
                "model": model, "pair_type": "shuffled", "frames": -1,
                "lve": NAN, "fdd": NAN, "mod": NAN,
                # For the ground truth this is the agreement between two different speakers.
                **bas_scores(motion, beats_cache[p]["gt"]),
                **pbas_scores(motion, prosody_cache.get(p)),
            })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"wrote {RESULTS_CSV} ({len(df)} rows)")
    cols = [c for c in df.columns if c.startswith(("bas_", "pbas_"))]
    print(df.groupby(["model", "pair_type"])[cols].mean().round(3).T.to_string())


if __name__ == "__main__":
    main()
