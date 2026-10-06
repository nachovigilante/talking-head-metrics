"""Pairwise comparison of the four models on the canonical LVE, FDD and MOD.

Reads metrics_results_canonical.csv (run_canonical_metrics.py) and, for each
metric, compares every pair of models on the clips where all four have a value:
paired Wilcoxon signed-rank test, Benjamini-Hochberg correction within the
metric, and rank-biserial r. FDD is signed and ideally zero, so the comparison
uses |FDD|; for LVE and MOD lower is better.

Writes results/metrics_canonical_stats.csv and prints means, orderings and the
significant pairs.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from utils.protocol_utils import RESULTS_DIR
from utils.stats_utils import rank_biserial

RESULTS_CSV = RESULTS_DIR / "metrics_results_canonical.csv"
OUT_CSV = RESULTS_DIR / "metrics_canonical_stats.csv"

MODELS = ["dpt", "artalk", "multitalk", "codetalker"]
# metric -> (column, transform, lower is better)
METRICS = {
    "lve": ("lve", lambda x: x, True),
    "fdd": ("fdd", lambda x: np.abs(x), True),
    "mod": ("mod", lambda x: x, True),
}


def main():
    df = pd.read_csv(RESULTS_CSV)
    print(f"rows: {len(df)} | models: {sorted(df.model.unique())}")

    rows = []
    for metric, (col, tf, lower_better) in METRICS.items():
        wide = df.pivot_table(index="clip_stem", columns="model", values=col)
        wide = wide.dropna(subset=MODELS)
        n = len(wide)
        for m1, m2 in combinations(MODELS, 2):
            a = tf(wide[m1].values)
            b = tf(wide[m2].values)
            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                stat, p = np.nan, 1.0
            r = rank_biserial(a, b)
            rows.append({
                "metric": metric, "model_a": m1, "model_b": m2,
                "n": n,
                "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                "median_a": float(np.median(a)), "median_b": float(np.median(b)),
                "wilcoxon_stat": float(stat), "p_raw": float(p),
                "rank_biserial_r": r,
            })

    res = pd.DataFrame(rows)
    res["p_fdr"] = np.nan
    res["significant"] = False
    for metric in METRICS:
        mask = res.metric == metric
        pvals = res.loc[mask, "p_raw"].values
        rej, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        res.loc[mask, "p_fdr"] = p_adj
        res.loc[mask, "significant"] = rej

    res.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}\n")

    print("=== Mean and median per model ===")
    summ = df.groupby("model")[["lve", "fdd", "mod"]].agg(["mean", "median"])
    print(summ.round(5))

    print("\n=== Ordering per metric (lower is better; FDD uses |FDD|) ===")
    for metric, (col, tf, _) in METRICS.items():
        wide = df.pivot_table(index="clip_stem", columns="model", values=col).dropna(subset=MODELS)
        means = {m: float(np.mean(tf(wide[m].values))) for m in MODELS}
        order = sorted(means, key=means.get)
        print(f"  {metric.upper():4s}: " + "  <  ".join(f"{m}={means[m]:.5f}" for m in order))

    print("\n=== Significant pairs (FDR < 0.05) ===")
    sig = res[res.significant].sort_values(["metric", "p_fdr"])
    for _, r in sig.iterrows():
        better = r.model_a if r.mean_a < r.mean_b else r.model_b
        print(f"  {r.metric.upper():4s} {r.model_a:10s} vs {r.model_b:10s} | "
              f"p_fdr={r.p_fdr:.2e} | r={r.rank_biserial_r:+.3f} | better: {better}")


if __name__ == "__main__":
    main()
