"""BAS and PBAS on denoised audio (ablation of Section 5.3.2).

Same real and shuffled pairs as scripts/evaluation/run_metrics.py, with the
prosodic events extracted from the output of denoise_canonical.py. LVE, FDD and
MOD do not depend on the audio and are left empty; BAS does not either and is
kept as a check (it must equal the values in metrics_results.csv).

Output: results/metrics_results_denoised.csv, with the columns of metrics_results.csv.
"""
import warnings

import pandas as pd
import torch
from tqdm import tqdm

from metrics.bas.so3_beat_detector import SO3BeatDetector
from metrics.pbas.prosodic_extractor import ProsodicExtractor
from utils.protocol_utils import (
    ARTALK_DIR, DPT_DIR, GT_NPZ_DIR, MANIFEST_PATH, RESULTS_DIR, WAV_DENOISED_DIR,
    bas_scores, derangement, load_rot_artalk, load_rot_dpt, load_rot_gt, pbas_scores,
)

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = RESULTS_DIR / "metrics_results_denoised.csv"
NAN = float("nan")


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    detect = SO3BeatDetector().detect_all_beats_so3
    extractor = ProsodicExtractor()

    rows, beats_cache, prosody_cache = [], {}, {}
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Real pairs"):
        try:
            beats = {
                "gt": detect(load_rot_gt(GT_NPZ_DIR / row.gt_npz, DEVICE)),
                "dpt": detect(load_rot_dpt(DPT_DIR / row.dpt_inference, DEVICE)),
                "artalk": detect(load_rot_artalk(ARTALK_DIR / row.artalk_inference, DEVICE)),
            }
        except Exception as e:
            print(f"skip {row.video_id}: {e}")
            continue
        try:
            prosodic = extractor.extract_all(str(WAV_DENOISED_DIR / row.wav))
        except Exception as e:
            print(f"no prosody for {row.video_id}: {e}")
            prosodic = None
        beats_cache[row.video_id] = beats
        prosody_cache[row.video_id] = prosodic

        for model in ("dpt", "artalk"):
            rows.append({
                "video_id": row.video_id, "partner_video_id": row.video_id,
                "clip_stem": row.clip_stem, "model": model, "pair_type": "real",
                "lve": NAN, "fdd": NAN, "mod": NAN,
                **bas_scores(beats[model], beats["gt"]),
                **pbas_scores(beats[model], prosodic),
            })
        rows.append({
            "video_id": row.video_id, "partner_video_id": row.video_id,
            "clip_stem": row.clip_stem, "model": "gt", "pair_type": "real",
            "lve": NAN, "fdd": NAN, "mod": NAN, "bas_omega": NAN, "bas_alpha": NAN, "bas_jerk": NAN,
            **pbas_scores(beats["gt"], prosodic),
        })

    video_ids = [v for v in manifest.video_id if v in beats_cache]
    partner_of = derangement(video_ids)
    clip_stem = dict(zip(manifest.video_id, manifest.clip_stem))
    for v in tqdm(video_ids, desc="Shuffled pairs"):
        p = partner_of[v]
        for model in ("gt", "dpt", "artalk"):
            rows.append({
                "video_id": v, "partner_video_id": p, "clip_stem": clip_stem[v],
                "model": model, "pair_type": "shuffled",
                "lve": NAN, "fdd": NAN, "mod": NAN,
                **bas_scores(beats_cache[v][model], beats_cache[p]["gt"]),
                **pbas_scores(beats_cache[v][model], prosody_cache.get(p)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"wrote {RESULTS_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
