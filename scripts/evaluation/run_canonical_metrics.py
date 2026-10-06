"""LVE, FDD and MOD for the four models in canonical orientation.

MultiTalk and CodeTalker do not predict head pose, so every mesh is compared
without global head rotation: the ground truth, DiffPoseTalk and ARTalk are
decoded with the global rotation set to zero (jaw and expression kept), and
MultiTalk and CodeTalker vertices are used as predicted. The canonical ground
truth is the reference for the four models.

Output: results/metrics_results_canonical.csv, one row per (video_id, model).
The values with global rotation, used for DiffPoseTalk vs ARTalk, are in
metrics_results.csv (run_metrics.py).
"""
import pandas as pd
import torch
from tqdm import tqdm

from metrics import FDDCalculator, LVECalculator, MODCalculator
from utils.flame_utils import get_flame_model
from utils.protocol_utils import (
    ARTALK_DIR, CODETALKER_DIR, DPT_DIR, GT_NPZ_DIR, MANIFEST_PATH, MULTITALK_DIR, RESULTS_DIR,
    decode_artalk, decode_flame_npz, load_artalk_flame, load_vertices,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = RESULTS_DIR / "metrics_results_canonical.csv"


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    flame = get_flame_model(device=DEVICE)
    flame_artalk = load_artalk_flame(DEVICE)
    lve_calc = LVECalculator(device=DEVICE)
    fdd_calc = FDDCalculator(device=DEVICE)
    mod_calc = MODCalculator(device=DEVICE)

    results, errors = [], []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Canonical"):
        try:
            gt_v = decode_flame_npz(flame, GT_NPZ_DIR / row.gt_npz, DEVICE, "th1kh", canonical=True)
            model_verts = {
                "dpt": decode_flame_npz(flame, DPT_DIR / row.dpt_inference, DEVICE, "ensemble", canonical=True),
                "artalk": decode_artalk(flame_artalk, ARTALK_DIR / row.artalk_inference, DEVICE, canonical=True),
                "multitalk": load_vertices(MULTITALK_DIR / f"{row.clip_stem}.npy", DEVICE),
                "codetalker": load_vertices(CODETALKER_DIR / f"{row.clip_stem}.npy", DEVICE),
            }
        except Exception as e:
            errors.append((row.video_id, str(e)))
            continue

        for model, pred_v in model_verts.items():
            T = min(pred_v.shape[0], gt_v.shape[0])
            if T < 2:
                errors.append((row.video_id, f"{model}: fewer than 2 shared frames"))
                continue
            with torch.no_grad():
                results.append({
                    "video_id": row.video_id,
                    "clip_stem": row.clip_stem,
                    "model": model,
                    "frames": T,
                    "lve": lve_calc.calculate_sequence_lve(pred_v[:T], gt_v[:T]),
                    "fdd": fdd_calc.calculate_fdd(pred_v[:T], gt_v[:T]),
                    "mod": mod_calc.calculate_mod(pred_v[:T], gt_v[:T]),
                })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"wrote {RESULTS_CSV} ({len(df)} rows, {len(errors)} errors)")
    for video_id, error in errors[:10]:
        print(f"  {video_id}: {error}")
    if len(df):
        print(df.groupby("model")[["lve", "fdd", "mod"]].mean().round(5))


if __name__ == "__main__":
    main()
