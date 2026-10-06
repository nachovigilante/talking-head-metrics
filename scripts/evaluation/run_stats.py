"""Paired Wilcoxon signed-rank tests on metrics_results.csv (output of run_metrics.py).

Three families of comparisons, each with its own FDR (Benjamini-Hochberg) and
Bonferroni correction:

  1. Real vs shuffled pairs, per model: does BAS/PBAS separate the true pairing
     from a random one? (12 PBAS tests for the ground truth, 3 BAS + 12 PBAS for
     each model.)
  2. Model vs human reference on the same audio, for the 12 PBAS combinations.
  3. DiffPoseTalk vs ARTalk on the same clip, for LVE, FDD, MOD, BAS and PBAS.

Effect sizes: matched-pairs rank-biserial r (utils.stats_utils) and Cohen's d of
the paired differences. With about 1600 pairs most p-values are tiny, so the
effect sizes are the informative numbers.

Writes results/stats/wilcoxon_test{1,2,3}_*.csv and
results/stats/summary_for_latex.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from metrics.pbas.pbas_calculator import MOTION_TYPES, PROSODIC_TYPES
from utils.protocol_utils import RESULTS_DIR
from utils.stats_utils import rank_biserial

RESULTS_CSV = RESULTS_DIR / "metrics_results.csv"
OUT_DIR = RESULTS_DIR / "stats"

VERTEX_METRICS = ("lve", "fdd", "mod")
BAS_METRICS = tuple(f"bas_{bt}" for bt in MOTION_TYPES)
PBAS_METRICS = tuple(f"pbas_{bt}_{pt}" for bt in MOTION_TYPES for pt in PROSODIC_TYPES)
RHYTHM_METRICS = BAS_METRICS + PBAS_METRICS


def wilcoxon_paired(x, y, label):
    """Wilcoxon signed-rank test on paired (x, y), dropping pairs with a NaN."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    out = {"label": label, "n": int(len(x))}
    diffs = x - y
    n_pos, n_neg = int((diffs > 0).sum()), int((diffs < 0).sum())
    if len(x) < 5 or (diffs != 0).sum() < 5:
        return {**out, "W": np.nan, "p": np.nan, "r_rb": np.nan, "d": np.nan,
                "mean_diff": float(diffs.mean()) if len(x) else np.nan,
                "n_pos": n_pos, "n_neg": n_neg}
    res = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    sd = float(np.std(diffs, ddof=1))
    return {
        **out, "W": float(res.statistic), "p": float(res.pvalue),
        "r_rb": rank_biserial(x, y), "d": float(np.mean(diffs) / sd) if sd > 0 else np.nan,
        "mean_diff": float(diffs.mean()), "n_pos": n_pos, "n_neg": n_neg,
    }


def add_corrections(df):
    """Bonferroni and Benjamini-Hochberg corrected p-values within the family."""
    pvals = df["p"].to_numpy()
    valid = ~np.isnan(pvals)
    p_bonf, p_fdr = np.full_like(pvals, np.nan), np.full_like(pvals, np.nan)
    if valid.any():
        p_bonf[valid] = multipletests(pvals[valid], alpha=0.05, method="bonferroni")[1]
        p_fdr[valid] = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")[1]
    df = df.copy()
    df["p_bonferroni"] = p_bonf
    df["p_fdr_bh"] = p_fdr
    df["sig_at_05_fdr"] = (df["p_fdr_bh"] < 0.05).astype(int)
    return df


def pivot(df, model, pair_type, columns):
    sub = df[(df.model == model) & (df.pair_type == pair_type)]
    return sub.set_index("video_id")[list(columns)]


def test_real_vs_shuffled(df):
    rows = []
    for model in ("gt", "dpt", "artalk"):
        cols = PBAS_METRICS if model == "gt" else RHYTHM_METRICS
        real, shuf = pivot(df, model, "real", cols), pivot(df, model, "shuffled", cols)
        common = real.index.intersection(shuf.index)
        for col in cols:
            rows.append({"model": model, "metric": col,
                         **wilcoxon_paired(real.loc[common, col], shuf.loc[common, col], f"{model}/{col}")})
    return add_corrections(pd.DataFrame(rows))


def test_model_vs_gt(df):
    rows = []
    gt = pivot(df, "gt", "real", PBAS_METRICS)
    for model in ("dpt", "artalk"):
        pred = pivot(df, model, "real", PBAS_METRICS)
        common = pred.index.intersection(gt.index)
        for col in PBAS_METRICS:
            rows.append({"model": model, "metric": col,
                         **wilcoxon_paired(pred.loc[common, col], gt.loc[common, col], f"{model}/{col}_vs_gt")})
    return add_corrections(pd.DataFrame(rows))


def test_dpt_vs_artalk(df):
    cols = VERTEX_METRICS + RHYTHM_METRICS
    dpt, art = pivot(df, "dpt", "real", cols), pivot(df, "artalk", "real", cols)
    common = dpt.index.intersection(art.index)
    rows = [{"metric": col, **wilcoxon_paired(dpt.loc[common, col], art.loc[common, col], f"dpt_vs_artalk/{col}")}
            for col in cols]
    return add_corrections(pd.DataFrame(rows))


def summary_table(test1, test2, test3):
    rows = []
    for metric in VERTEX_METRICS + RHYTHM_METRICS:
        row = {"metric": metric}
        for model in ("gt", "dpt", "artalk"):
            r = test1[(test1.model == model) & (test1.metric == metric)]
            row[f"r_rb_real_vs_shuf_{model}"] = r.r_rb.iloc[0] if len(r) else np.nan
            row[f"p_fdr_real_vs_shuf_{model}"] = r.p_fdr_bh.iloc[0] if len(r) else np.nan
        if metric in PBAS_METRICS:
            for model in ("dpt", "artalk"):
                r = test2[(test2.model == model) & (test2.metric == metric)]
                row[f"r_rb_{model}_vs_gt"] = r.r_rb.iloc[0]
                row[f"p_fdr_{model}_vs_gt"] = r.p_fdr_bh.iloc[0]
        r = test3[test3.metric == metric]
        row["r_rb_dpt_vs_artalk"] = r.r_rb.iloc[0]
        row["p_fdr_dpt_vs_artalk"] = r.p_fdr_bh.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(RESULTS_CSV)
    OUT_DIR.mkdir(exist_ok=True)

    test1 = test_real_vs_shuffled(df)
    test2 = test_model_vs_gt(df)
    test3 = test_dpt_vs_artalk(df)
    test1.to_csv(OUT_DIR / "wilcoxon_test1_real_vs_shuffled.csv", index=False)
    test2.to_csv(OUT_DIR / "wilcoxon_test2_model_vs_gt.csv", index=False)
    test3.to_csv(OUT_DIR / "wilcoxon_test3_dpt_vs_artalk.csv", index=False)
    summary_table(test1, test2, test3).to_csv(OUT_DIR / "summary_for_latex.csv", index=False)
    print(f"wrote {OUT_DIR}/")

    print("\n1. Real vs shuffled (r > 0: real pairs score higher)")
    for model in ("gt", "dpt", "artalk"):
        sub = test1[(test1.model == model) & test1.metric.str.startswith("pbas")]
        print(f"   {model:7s} PBAS: mean r = {sub.r_rb.mean():.3f}, significant {sub.sig_at_05_fdr.sum()}/12")

    print("\n2. Model vs human reference (r > 0: model above the human)")
    for model in ("dpt", "artalk"):
        sub = test2[test2.model == model]
        print(f"   {model:7s} median r = {sub.r_rb.median():.3f}, max r = {sub.r_rb.max():.3f}, "
              f"significant {sub.sig_at_05_fdr.sum()}/12")

    print("\n3. DiffPoseTalk vs ARTalk (r on dpt - artalk)")
    for _, row in test3.iterrows():
        lower_is_better = row.metric in VERTEX_METRICS
        if not row.sig_at_05_fdr:
            winner = "no significant difference"
        elif (row.r_rb > 0) != lower_is_better:
            winner = "DiffPoseTalk better"
        else:
            winner = "ARTalk better"
        print(f"   {row.metric:25s} r = {row.r_rb:+.3f}  p_fdr = {row.p_fdr_bh:.2e}  {winner}")


if __name__ == "__main__":
    main()
