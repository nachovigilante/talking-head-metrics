"""Raw vs denoised audio (Table 5.9 and Section 5.3.2).

Reads metrics_results.csv and metrics_results_denoised.csv and prints, for both
versions of the audio:
  - the change of the mean PBAS of the human reference on real pairs, per prosodic event;
  - real vs shuffled: mean rank-biserial r over the 12 PBAS combinations, per model;
  - model vs human reference: median r over the 12 combinations and the number of
    significant ones (Wilcoxon, Benjamini-Hochberg within the model);
  - DiffPoseTalk vs ARTalk: significant PBAS combinations (correction over the 3
    BAS and 12 PBAS tests) and the largest |r|.
BAS does not depend on the audio; the last check confirms it is identical.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from metrics.pbas.pbas_calculator import MOTION_TYPES, PROSODIC_TYPES
from utils.protocol_utils import RESULTS_DIR
from utils.stats_utils import rank_biserial

PBAS_COLS = [f"pbas_{bt}_{pt}" for bt in MOTION_TYPES for pt in PROSODIC_TYPES]
BAS_COLS = [f"bas_{bt}" for bt in MOTION_TYPES]


def wilcoxon_p(x, y):
    d = np.asarray(x) - np.asarray(y)
    d = d[~np.isnan(d) & (d != 0)]
    if len(d) < 3:
        return np.nan
    return float(wilcoxon(d, alternative="two-sided").pvalue)


def fdr(pvals):
    pvals = [1.0 if np.isnan(p) else p for p in pvals]
    return multipletests(pvals, method="fdr_bh")[1]


def rows(df, model, pair_type):
    return df[(df.model == model) & (df.pair_type == pair_type)].set_index("video_id")


def summarize(df):
    out = {}
    for model in ("gt", "dpt", "artalk"):
        real, shuf = rows(df, model, "real"), rows(df, model, "shuffled")
        common = real.index.intersection(shuf.index)
        out[f"real vs shuffled, {model}: mean r"] = np.mean(
            [rank_biserial(real.loc[common, c], shuf.loc[common, c]) for c in PBAS_COLS])

    gt = rows(df, "gt", "real")
    for model in ("dpt", "artalk"):
        pred = rows(df, model, "real")
        common = pred.index.intersection(gt.index)
        r = [rank_biserial(pred.loc[common, c], gt.loc[common, c]) for c in PBAS_COLS]
        p = fdr([wilcoxon_p(pred.loc[common, c], gt.loc[common, c]) for c in PBAS_COLS])
        out[f"{model} vs human: median r"] = np.median(r)
        out[f"{model} vs human: significant"] = int((p < 0.05).sum())

    dpt, art = rows(df, "dpt", "real"), rows(df, "artalk", "real")
    common = dpt.index.intersection(art.index)
    cols = BAS_COLS + PBAS_COLS
    r = [rank_biserial(dpt.loc[common, c], art.loc[common, c]) for c in cols]
    p = fdr([wilcoxon_p(dpt.loc[common, c], art.loc[common, c]) for c in cols])
    out["dpt vs artalk: significant PBAS"] = int((p[len(BAS_COLS):] < 0.05).sum())
    out["dpt vs artalk: max |r| PBAS"] = max(abs(x) for x in r[len(BAS_COLS):])
    return out


def main():
    raw = pd.read_csv(RESULTS_DIR / "metrics_results.csv")
    den = pd.read_csv(RESULTS_DIR / "metrics_results_denoised.csv")

    print("Mean PBAS of the human reference on real pairs, denoised vs raw:")
    raw_gt, den_gt = rows(raw, "gt", "real"), rows(den, "gt", "real")
    for pt in PROSODIC_TYPES:
        change = np.mean([
            (den_gt[f"pbas_{bt}_{pt}"].mean() - raw_gt[f"pbas_{bt}_{pt}"].mean())
            / raw_gt[f"pbas_{bt}_{pt}"].mean() * 100
            for bt in MOTION_TYPES
        ])
        print(f"  {pt:13s} {change:+.1f}%")

    print(f"\n{'':40s}{'raw':>8s}{'denoised':>10s}")
    s_raw, s_den = summarize(raw), summarize(den)
    for key in s_raw:
        fmt = "{:>8d}{:>10d}" if isinstance(s_raw[key], int) else "{:>8.3f}{:>10.3f}"
        print(f"{key:40s}" + fmt.format(s_raw[key], s_den[key]))

    max_diff = max(
        np.nanmax(np.abs(rows(raw, m, "real")[BAS_COLS].values - rows(den, m, "real").loc[
            rows(raw, m, "real").index, BAS_COLS].values))
        for m in ("dpt", "artalk")
    )
    print(f"\nBAS, raw vs denoised: max |difference| = {max_diff:.1e}")


if __name__ == "__main__":
    main()
