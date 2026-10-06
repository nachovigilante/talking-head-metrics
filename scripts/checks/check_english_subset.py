"""Robustness check: repeat the main analyses on the English-only videos.

Reads the per-video language labels (data/language_ids_final.csv, produced by
scripts/dataset/detect_language.py) and keeps the videos labelled English with an
unambiguous label. On that subset it recomputes:

- the mean rank-biserial r between real and shuffled PBAS, for each motion source;
- the model-vs-human PBAS comparison (median r, and in how many of the 12
  combinations the model scores above the human reference);
- the ordering of the four models in canonical LVE, |FDD| and MOD.
"""
import numpy as np
import pandas as pd

from metrics.pbas.pbas_calculator import MOTION_TYPES, PROSODIC_TYPES
from utils.protocol_utils import LANGUAGE_IDS_PATH, RESULTS_DIR
from utils.stats_utils import rank_biserial

PBAS_COLS = [f"pbas_{bt}_{pt}" for bt in MOTION_TYPES for pt in PROSODIC_TYPES]


def rhythm_summary(df):
    lines = []
    for m in ("gt", "dpt", "artalk"):
        real = df[(df.model == m) & (df.pair_type == "real")].set_index("video_id")
        shuf = df[(df.model == m) & (df.pair_type == "shuffled")].set_index("video_id")
        common = real.index.intersection(shuf.index)
        r = np.mean([rank_biserial(real.loc[common, c], shuf.loc[common, c]) for c in PBAS_COLS])
        lines.append(f"  {m}: real vs shuffled, mean r = {r:.3f}")
    real = df[df.pair_type == "real"]
    gt = real[real.model == "gt"].set_index("video_id")
    for m in ("dpt", "artalk"):
        model = real[real.model == m].set_index("video_id")
        common = gt.index.intersection(model.index)
        rs = [rank_biserial(model.loc[common, c], gt.loc[common, c]) for c in PBAS_COLS]
        lines.append(f"  {m} vs gt: median r = {np.median(rs):.3f}, model above gt in "
                     f"{sum(r > 0 for r in rs)}/12 combinations")
    return lines


def expression_ordering(df):
    df = df.assign(fdd=df.fdd.abs())
    means = df.groupby("model")[["lve", "fdd", "mod"]].mean()
    return [f"  {m.upper()}: " + " < ".join(means[m].sort_values().index) for m in ("lve", "fdd", "mod")]


def main():
    lang = pd.read_csv(LANGUAGE_IDS_PATH)
    english = set(lang[(lang.language == "en") & (lang.is_ambiguous == 0)].video_id)
    rhythm = pd.read_csv(RESULTS_DIR / "metrics_results.csv")
    canonical = pd.read_csv(RESULTS_DIR / "metrics_results_canonical.csv")

    for name, keep in (("all videos", lambda d: d), ("English only", lambda d: d[d.video_id.isin(english)])):
        n = keep(canonical).video_id.nunique()
        print(f"=== {name} (n = {n}) ===")
        print("\n".join(rhythm_summary(keep(rhythm))))
        print("\n".join(expression_ordering(keep(canonical))))


if __name__ == "__main__":
    main()
