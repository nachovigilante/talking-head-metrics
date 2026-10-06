"""How much does removing the global head rotation shrink the GT-vs-model distance?

Backs the claim in Section 4.4.3: for the models that do not predict pose
(MultiTalk, CodeTalker), compare the mean per-vertex distance to the GT mesh
decoded with its global rotation (as in metrics_results.csv) and to the GT mesh in
canonical orientation (global rotation set to zero, as in
metrics_results_canonical.csv). Distances are averaged over all 5023 vertices and
all shared frames of each clip, and reported in millimetres.
"""
import pandas as pd
import torch
from tqdm import tqdm

from utils.flame_utils import get_flame_model
from utils.protocol_utils import (
    CODETALKER_DIR, GT_NPZ_DIR, MANIFEST_PATH, MULTITALK_DIR, decode_flame_npz, load_vertices,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIRS = {"multitalk": MULTITALK_DIR, "codetalker": CODETALKER_DIR}


def mean_vertex_distance(a, b):
    T = min(a.shape[0], b.shape[0])
    return torch.norm(a[:T] - b[:T], dim=2).mean().item() * 1000


def main():
    flame = get_flame_model(device=DEVICE)
    manifest = pd.read_csv(MANIFEST_PATH)
    rows = []
    for r in tqdm(manifest.itertuples(index=False), total=len(manifest)):
        gt_rot = decode_flame_npz(flame, GT_NPZ_DIR / r.gt_npz, DEVICE, "th1kh")
        gt_can = decode_flame_npz(flame, GT_NPZ_DIR / r.gt_npz, DEVICE, "th1kh", canonical=True)
        for model, directory in MODEL_DIRS.items():
            verts = load_vertices(directory / f"{r.clip_stem}.npy", DEVICE)
            rows.append({
                "video_id": r.video_id, "model": model,
                "dist_with_rotation_mm": mean_vertex_distance(gt_rot, verts),
                "dist_canonical_mm": mean_vertex_distance(gt_can, verts),
            })

    df = pd.DataFrame(rows)
    df["ratio"] = df.dist_with_rotation_mm / df.dist_canonical_mm
    for model, g in df.groupby("model"):
        print(f"{model:10s} n={len(g)}  with rotation: mean {g.dist_with_rotation_mm.mean():.2f} mm"
              f" (median {g.dist_with_rotation_mm.median():.2f})  canonical: mean"
              f" {g.dist_canonical_mm.mean():.2f} mm (median {g.dist_canonical_mm.median():.2f})"
              f"  ratio of means {g.dist_with_rotation_mm.mean() / g.dist_canonical_mm.mean():.1f}"
              f"  median per-clip ratio {g.ratio.median():.1f}")


if __name__ == "__main__":
    main()
