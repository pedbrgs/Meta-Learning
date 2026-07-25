"""Stochastic stability of PSTD-selected feature subsets across independent runs.

Uses the 30 independent runs per dataset already stored for the decomposition
experiments (same data, different random seeds), so it isolates the effect of the
algorithm's stochastic components. Each run's selection is reconstructed in the original
feature space as kept[feature_idxs][context_vector].

Stability is quantified with the measure of Nogueira, Sechidis and Brown (2018), which
corrects for chance, admits subsets of varying size, and has a known sampling variance.
Given the binary selection matrix Z (M runs x p features) with per-feature selection
frequency p_f and average subset size k-bar,

    Phi = 1 - ( (1/p) sum_f (M/(M-1)) p_f (1 - p_f) ) / ( (k-bar/p) (1 - k-bar/p) ),

where Phi = 1 is identical selection every run and Phi = 0 is chance-level agreement.
A percentile confidence interval is obtained by resampling the runs.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# Path to the decomposition results with per-run feature_idxs, removed_features and
# best_context_vector. Override with the DECOMPOSITION_RESULTS environment variable.
RESULTS = Path(os.environ.get(
    "DECOMPOSITION_RESULTS",
    r"c:\Users\pedbr\OneDrive\Education\Doctor's degree\Research\Papers"
    r"\Decomposition\Results\decomposition\self_tuning_proposed_with_feature_idxs.parquet",
))

DIM = {  # original dimensionality, for ordering and trend analysis
    "libras": 90, "scadi": 205, "lsvt": 310, "madelon_valid": 500, "uji_indoor": 522,
    "har": 561, "hapt": 561, "dlbcl": 5469, "brain_tumor_1": 5920,
    "prostate_tumor_1": 5966, "leukemia_3": 11225, "lungc": 12600,
}


def selected_indices(row):
    """Reconstruct the selected features of one run in the original feature space."""
    fi = np.asarray(row.feature_idxs, dtype=int)
    rem = np.asarray(row.removed_features, dtype=int)
    cv = np.asarray(row.best_context_vector, dtype=int).astype(bool)
    n = fi.size + rem.size
    kept = np.setdiff1d(np.arange(n), rem)
    assert cv.size == fi.size, "context vector length mismatch"
    return kept[fi][cv], n


def selection_matrix(runs):
    """Binary matrix Z (M runs x p features) over the original feature space."""
    subsets, n = [], None
    for _, row in runs.iterrows():
        idx, n_row = selected_indices(row)
        n = n_row if n is None else n
        assert n_row == n, "inconsistent feature count across runs"
        subsets.append(idx)
    Z = np.zeros((len(subsets), n), dtype=np.int8)
    for i, idx in enumerate(subsets):
        Z[i, idx] = 1
    return Z


def nogueira_stability(Z):
    """Point estimate of the Nogueira stability measure."""
    M, p = Z.shape
    pf = Z.mean(axis=0)
    kbar = Z.sum(axis=1).mean()
    numerator = (M / (M - 1)) * pf * (1 - pf)
    denominator = (kbar / p) * (1 - kbar / p)
    return 1 - numerator.mean() / denominator


def mean_pairwise_jaccard(Z):
    """Average Jaccard index over all run pairs, an intuitive complement."""
    M = Z.shape[0]
    total, count = 0.0, 0
    for i in range(M):
        for j in range(i + 1, M):
            inter = np.logical_and(Z[i], Z[j]).sum()
            union = np.logical_or(Z[i], Z[j]).sum()
            total += inter / union if union else 0.0
            count += 1
    return total / count


def jackknife_ci(Z):
    """95% CI via leave-one-run-out jackknife.

    Resampling runs with replacement duplicates identical subsets and spuriously inflates
    agreement, so the jackknife, which never duplicates a run, is used instead.
    """
    M = Z.shape[0]
    theta = np.array([nogueira_stability(np.delete(Z, i, axis=0)) for i in range(M)])
    theta_bar = theta.mean()
    var = (M - 1) / M * np.sum((theta - theta_bar) ** 2)
    se = np.sqrt(var)
    point = nogueira_stability(Z)
    return point - 1.96 * se, point + 1.96 * se


def n_distinct_pools(runs):
    """Number of distinct VIP-retained feature sets across the runs (pool variability)."""
    pools = {
        frozenset(np.setdiff1d(
            np.arange(np.asarray(r.feature_idxs).size + np.asarray(r.removed_features).size),
            np.asarray(r.removed_features, dtype=int)).tolist())
        for _, r in runs.iterrows()
    }
    return len(pools)


def within_pool_stability(runs):
    """Stability among the runs that share the most common VIP-retained pool.

    Isolates the variability due to the genetic search, holding the candidate pool fixed.
    Returns the stability and the number of runs sharing that pool.
    """
    keys, subsets = [], []
    for _, r in runs.iterrows():
        rem = np.asarray(r.removed_features, dtype=int)
        n = np.asarray(r.feature_idxs).size + rem.size
        kept = np.setdiff1d(np.arange(n), rem)
        keys.append(frozenset(kept.tolist()))
        idx, _ = selected_indices(r)
        subsets.append((frozenset(kept.tolist()), idx, n))
    common = pd.Series(keys).value_counts().index[0]
    group = [(idx, n) for key, idx, n in subsets if key == common]
    if len(group) < 2:
        return np.nan, len(group)
    n = group[0][1]
    Z = np.zeros((len(group), n), dtype=np.int8)
    for i, (idx, _) in enumerate(group):
        Z[i, idx] = 1
    return nogueira_stability(Z), len(group)


if __name__ == "__main__":
    data = pd.read_parquet(RESULTS)
    rows = []
    for dataset in sorted(data.dataset.unique(), key=lambda d: DIM.get(d, 0)):
        runs = data[data.dataset == dataset]
        Z = selection_matrix(runs)
        lo, hi = jackknife_ci(Z)
        wp_phi, wp_n = within_pool_stability(runs)
        rows.append({
            "dataset": dataset,
            "features": DIM.get(dataset, Z.shape[1]),
            "mean_subset": Z.sum(axis=1).mean(),
            "jaccard": mean_pairwise_jaccard(Z),
            "stability": nogueira_stability(Z),
            "ci_low": lo,
            "ci_high": hi,
            "pools": n_distinct_pools(runs),
            "within_pool": wp_phi,
            "wp_runs": wp_n,
        })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))
    print(f"\nmean stability across datasets: {out.stability.mean():.3f}")
    print(f"Spearman(stability, features): "
          f"{out[['stability','features']].corr(method='spearman').iloc[0,1]:.3f}")
