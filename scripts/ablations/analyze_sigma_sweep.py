"""Analysis of sigma_sweep_results.csv (Table 5.8 and Section 5.3.1).

For each model and sigma:
  - mean PBAS on real and shuffled pairs (averaged over the 12 combinations);
  - r_avg: rank-biserial r between the per-clip averages of the 12 combinations,
    real vs shuffled. Its maximum defines sigma_opt (Table 5.8);
  - r_mean12: the mean of the 12 per-combination r values.
Then, for the ground truth, the dominant period of the head motion and whether
sigma_opt follows it, at the population level and clip by clip.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from metrics.pbas.pbas_calculator import MOTION_TYPES, PROSODIC_TYPES
from utils.protocol_utils import RESULTS_DIR
from utils.stats_utils import rank_biserial

FPS = 25
PBAS_COLS = [f"pbas_{bt}_{pt}" for bt in MOTION_TYPES for pt in PROSODIC_TYPES]


def sigma_curve(df):
    """One row per sigma with the mean scores and both discrimination measures."""
    rows = []
    for sigma, g in df.groupby("sigma"):
        real = g[g.pair_type == "real"].set_index("video_id")[PBAS_COLS]
        shuf = g[g.pair_type == "shuffled"].set_index("video_id")[PBAS_COLS]
        common = real.index.intersection(shuf.index)
        real, shuf = real.loc[common], shuf.loc[common]
        rows.append({
            "sigma": sigma,
            "sigma_ms": sigma / FPS * 1000,
            "real": real.mean(axis=1).mean(),
            "shuffled": shuf.mean(axis=1).mean(),
            "r_avg": rank_biserial(real.mean(axis=1).values, shuf.mean(axis=1).values),
            "r_mean12": np.mean([rank_biserial(real[c].values, shuf[c].values) for c in PBAS_COLS]),
        })
    return pd.DataFrame(rows).sort_values("sigma").reset_index(drop=True)


def main():
    df = pd.read_csv(RESULTS_DIR / "sigma_sweep_results.csv")
    print(f"{len(df)} rows, {df.video_id.nunique()} clips, {df.sigma.nunique()} sigmas")

    curves = {}
    for model in ("gt", "dpt", "artalk"):
        curve = sigma_curve(df[df.model == model])
        curves[model] = curve
        best = curve.loc[curve.r_avg.idxmax()]
        print(f"\n--- {model} ---")
        print(curve.round(4).to_string(index=False))
        print(f"sigma_opt = {best.sigma:.2f} frames ({best.sigma_ms:.0f} ms), r_avg = {best.r_avg:.4f}")

    gt = df[df.model == "gt"]
    period = gt.drop_duplicates("video_id")["motion_period_s"].dropna()
    print(f"\nMotion period (dominant frequency of the head angular speed, 0.5-10 Hz), {len(period)} clips:")
    print(f"  median {period.median():.3f} s, interquartile range {period.quantile(0.25):.3f}-{period.quantile(0.75):.3f} s")

    sigma_opt_s = curves["gt"].loc[curves["gt"].r_avg.idxmax(), "sigma"] / FPS
    for k in (2, 4, 6, 8):
        print(f"  sigma_opt / (T/{k}) = {sigma_opt_s / (period.median() / k):.2f}")

    # Clip by clip: the sigma with the largest gap between real and shuffled pairs.
    real = gt[gt.pair_type == "real"].assign(real=lambda d: d[PBAS_COLS].mean(axis=1))
    shuf = gt[gt.pair_type == "shuffled"].assign(shuffled=lambda d: d[PBAS_COLS].mean(axis=1))
    merged = real[["video_id", "sigma", "real", "motion_period_s"]].merge(
        shuf[["video_id", "sigma", "shuffled"]], on=["video_id", "sigma"]
    )
    merged["gap"] = merged.real - merged.shuffled
    per_clip = merged.loc[merged.groupby("video_id")["gap"].idxmax()].dropna(subset=["motion_period_s"])
    rho, p = spearmanr(per_clip.sigma / FPS, per_clip.motion_period_s)
    print(f"\nPer clip ({len(per_clip)} clips): median sigma_opt {per_clip.sigma.median():.2f} frames; "
          f"Spearman between sigma_opt and the motion period rho = {rho:+.3f} (p = {p:.3g})")


if __name__ == "__main__":
    main()
