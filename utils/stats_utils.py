import numpy as np
from scipy.stats import rankdata


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation of the differences x - y.

    r = (T+ - T-) / (T+ + T-), where T+ and T- are the sums of the ranks of |x - y|
    over the positive and the negative differences. Pairs with a NaN or a zero
    difference are dropped, as in the Wilcoxon signed-rank test.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[~np.isnan(d) & (d != 0)]
    if len(d) == 0:
        return np.nan
    ranks = rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())
