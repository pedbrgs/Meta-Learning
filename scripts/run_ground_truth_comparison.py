"""Ground-truth validation of PSTD-selected features, with the indexing bug fixed.

The earlier scripts passed one DataLoader to the CCEA and then to the baselines. CCPSTFG
mutates that object in place (it prunes columns in the VIP stage and permutes them in the
decomposition stage), so the baselines were scored on the reduced, reordered matrix and
their indices no longer referred to the original feature space. Here every method receives
its own freshly loaded DataLoader, and the PSTD selection is reconstructed in the original
space and checked two ways before any metric is computed.

Indexing, verified against the PyCCEA source:
    kept       = sorted(setdiff1d(arange(n), removed_features))   original indices kept by VIP
    feature_idxs                                                   permutation of range(len(kept))
    best_context_vector                                           binary mask over the decomposed order
    selected   = kept[feature_idxs][mask]                          selection in the original space
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split

logging.disable(logging.CRITICAL)

# PyCCEA is expected to be pip-installed in the active environment, as it is on the
# server where the run_*.py scripts execute. For local runs against a source checkout,
# point the PYCCEA_SOURCE environment variable at the directory that contains the pyccea
# package; when unset, the package is imported from the environment. pyccea itself is
# imported lazily inside the worker functions so that --summary needs only pandas.
_source = os.environ.get("PYCCEA_SOURCE")
if _source:
    sys.path.insert(0, _source)

SCRATCH = Path(os.environ.get("GT_OUTPUT_DIR", Path(__file__).parent)).resolve()
SCRATCH.mkdir(parents=True, exist_ok=True)

N_FEATURES = 1000
N_INFORMATIVE = 50
N_SAMPLES = 500
CLASS_SEP = 0.8
FLIP_Y = 0.05
TEST_SIZE = 0.30
MAX_GEN = 200  # budget-limited: the redundant variant does not converge on its own


def build_dataset(seed, n_redundant):
    """Informative block [0:50], redundant block [50:50+r], noise after, shuffle disabled."""
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=CLASS_SEP,
        flip_y=FLIP_Y,
        shuffle=False,
        random_state=seed,
    )
    frame = pd.DataFrame(X, columns=[str(i) for i in range(N_FEATURES)])
    frame["label"] = y
    train, test = train_test_split(
        frame, test_size=TEST_SIZE, random_state=seed, stratify=frame["label"]
    )
    train, test = train.copy(), test.copy()
    train["subset"], test["subset"] = "train", "test"
    data = pd.concat([train, test], ignore_index=True)
    # Absolute path: DataLoader._load does os.path.join(datasets_dir, file), and an
    # absolute file makes that join return the file directly (same trick the run_*.py
    # scripts use on the server). Avoids monkeypatching the loader.
    path = (SCRATCH / f"gt_final_r{n_redundant}_seed{seed}.parquet").resolve()
    data.to_parquet(path)
    return path


def data_config(seed):
    """DataLoader configuration, matching the inline config the server runner uses."""
    return {
        "general": {"splitter_type": "k_fold", "verbose": False,
                    "float_dtype": "float32", "seed": seed},
        "splitter": {"preset": True, "kfolds": 10, "prefold": False},
        "normalization": {"normalize": True, "method": "min_max"},
        "preprocessing": {"drop_na": True, "winsorization": False, "quantiles": [0.01, 0.99]},
    }


def make_loader(path, seed):
    from pyccea.utils.datasets import DataLoader

    name = f"gt_final_{Path(path).stem}"
    DataLoader.DATASETS[name] = {"file": str(path), "task": "classification"}
    loader = DataLoader(dataset=name, conf=data_config(seed))
    loader.get_ready()
    return loader


def ccea_config(seed):
    """CCPSTFG configuration. Same structure the server runner uses, with the corrected
    settings from the thesis: KNN fitness (sensitive to noise features, so selection
    matters) and an aggressive VIP removal quantile (q_max=0.95)."""
    return {
        "coevolution": {
            "subpop_sizes": [30],
            "max_gen": int(os.environ.get("GT_MAX_GEN", MAX_GEN)),
            "max_gen_without_improvement": 100,
            "optimized_resource_allocation": False,
            "max_best_context_vectors": 0,
            "seed": seed,
        },
        "decomposition": {
            "method": "clustering", "drop": True, "max_n_clusters": 10,
            "max_n_pls_components": 10, "removal_quantile_step_size": 0.05,
            "max_removal_quantile": 0.95, "clustering_model_type": "agglomerative_clustering",
        },
        "collaboration": {"method": "best"},
        "wrapper": {"task": "classification", "cache_size": 2000,
                    "model_type": "k_nearest_neighbors", "use_subprocess": False},
        "evaluation": {"fitness_function": "penalty", "eval_function": "balanced_accuracy",
                       "eval_mode": "k_fold", "weights": [1.0, 0.0],
                       "n_workers": int(os.environ.get("GT_WORKERS", 4))},
        "optimizer": {"method": "GA", "selection_method": "generational",
                      "mutation_rate": 0.05, "crossover_rate": 1.0,
                      "tournament_sample_size": 1, "elite_size": 1},
    }


def reconstruct(ccea, path, seed):
    """Recover selected indices in the original space and verify the mapping two ways.

    feature_idxs indexes the post-removal space, so the selection in the original space is
    kept[feature_idxs][mask], not feature_idxs[mask]. The latter is the bug present in
    run_ground_truth_experiment.py, which inflates the overlap with the informative block.
    """
    removed = np.asarray(getattr(ccea, "removed_features", []), dtype=int)
    kept = np.setdiff1d(np.arange(N_FEATURES), removed)
    order = np.asarray(ccea.feature_idxs, dtype=int)
    mask = np.asarray(ccea.best_context_vector, dtype=int).astype(bool)

    # Structural checks.
    assert kept.size == N_FEATURES - removed.size, "kept size mismatch"
    assert np.array_equal(np.sort(order), np.arange(kept.size)), "feature_idxs is not a permutation of range(n_kept)"
    assert mask.size == kept.size, "context vector length does not match kept size"

    selected = kept[order][mask]
    assert selected.size == 0 or (selected.min() >= 0 and selected.max() < N_FEATURES), "selected out of range"
    assert np.unique(selected).size == selected.size, "duplicate indices in selection"

    # Gold-standard check: the data of the reconstructed original columns must equal the
    # columns the CCEA actually optimized (its X_train after prune+permute, at mask).
    check = make_loader(path, seed)
    lhs = check.X_train[:, selected]
    rhs = ccea.data.X_train[:, mask]
    assert lhs.shape == rhs.shape, f"shape mismatch {lhs.shape} vs {rhs.shape}"
    assert np.allclose(lhs, rhs), "reconstructed columns do not match the CCEA's optimized columns"

    return selected


def groups(n_redundant):
    informative = set(range(N_INFORMATIVE))
    redundant = set(range(N_INFORMATIVE, N_INFORMATIVE + n_redundant))
    noise = set(range(N_INFORMATIVE + n_redundant, N_FEATURES))
    return informative, redundant, noise


def metrics(selected, n_redundant):
    informative, redundant, noise = groups(n_redundant)
    selected = set(int(i) for i in selected)
    n_inf, n_red, n_noi = len(selected & informative), len(selected & redundant), len(selected & noise)
    total = len(selected) or 1
    precision = n_inf / total
    recall = n_inf / len(informative)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_selected": len(selected),
        "n_informative": n_inf,
        "n_redundant": n_red,
        "n_noise": n_noi,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "informative_share": n_inf / (n_inf + n_red) if (n_inf + n_red) else np.nan,
    }


def run_pstd(path, seed, n_redundant):
    from pyccea.coevolution import CCPSTFG

    loader = make_loader(path, seed)
    started = time.time()
    ccea = CCPSTFG(data=loader, conf=ccea_config(seed), verbose=False)
    ccea.optimize()
    elapsed = time.time() - started

    selected = reconstruct(ccea, path, seed)
    result = metrics(selected, n_redundant)
    result.update(
        method="CCEA (PSTD)", seconds=elapsed,
        n_kept=int(N_FEATURES - np.asarray(ccea.removed_features).size),
        n_subcomps=int(ccea.n_subcomps), quantile_removed=float(ccea.quantile_to_remove),
    )
    return result


def run_baselines(path, k, seed, n_redundant):
    loader = make_loader(path, seed)  # fresh, never touched by a CCEA
    X, y = loader.X_train, loader.y_train
    out = []
    for name, scorer in (("chi2", chi2), ("mutual_info", mutual_info_classif)):
        support = np.flatnonzero(SelectKBest(scorer, k=k).fit(X, y).get_support())
        result = metrics(support, n_redundant)
        result["method"] = name
        out.append(result)
    rng = np.random.default_rng(seed)
    result = metrics(rng.choice(N_FEATURES, size=k, replace=False), n_redundant)
    result["method"] = "random"
    out.append(result)
    return out


RESULTS_CSV = SCRATCH / "gt_final_results.csv"
COLUMNS = [
    "method", "n_redundant_gen", "run", "n_selected", "n_informative", "n_redundant",
    "n_noise", "precision", "recall", "f1", "informative_share", "seconds", "n_kept",
    "n_subcomps", "quantile_removed",
]


def append_rows(rows):
    """Append rows to the results CSV, writing the header only if the file is new."""
    frame = pd.DataFrame(rows).reindex(columns=COLUMNS)
    header = not RESULTS_CSV.exists()
    frame.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def completed_runs():
    """Set of (n_redundant, seed) pairs already present as a PSTD row, for resuming."""
    if not RESULTS_CSV.exists():
        return set()
    done = pd.read_csv(RESULTS_CSV)
    done = done[done.method == "CCEA (PSTD)"]
    return set(zip(done.n_redundant_gen.astype(int), done.run.astype(int)))


def run_one(n_redundant, seed):
    """One atomic unit: PSTD plus baselines for a single (dataset, seed). Appends at the end."""
    path = build_dataset(seed, n_redundant)
    pstd = run_pstd(path, seed, n_redundant)
    pstd.update(run=seed, n_redundant_gen=n_redundant)
    print(f"[r={n_redundant} seed={seed}] PSTD kept={pstd['n_kept']} sel={pstd['n_selected']} "
          f"inf={pstd['n_informative']} red={pstd['n_redundant']} noise={pstd['n_noise']} "
          f"prec={pstd['precision']:.3f} ({pstd['seconds']/60:.0f}min)", flush=True)

    rows = [pstd]
    for r in run_baselines(path, pstd["n_selected"], seed, n_redundant):
        r.update(run=seed, n_redundant_gen=n_redundant)
        rows.append(r)
        print(f"           {r['method']:<12} inf={r['n_informative']:>3} red={r['n_redundant']:>3} "
              f"noise={r['n_noise']:>3} prec={r['precision']:.3f}", flush=True)

    # Written only after all compute succeeds, so a killed run leaves no partial row.
    append_rows(rows)


def drive(n_seeds, redundant_levels, timeout_min):
    """Launch each (dataset, seed) as its own subprocess with a wall-clock timeout.

    Isolating every run in a child process means a hang or crash kills only that run: the
    driver kills the process tree, logs it, and moves on. Already-completed pairs are
    skipped, so the job resumes cleanly after any interruption.
    """
    import signal
    import subprocess

    pending = [
        (r, s) for r in redundant_levels for s in range(1, n_seeds + 1)
        if (r, s) not in completed_runs()
    ]
    print(f"driver: {len(pending)} run(s) pending, timeout {timeout_min} min each", flush=True)

    for n_redundant, seed in pending:
        if (n_redundant, seed) in completed_runs():  # re-check in case of manual edits
            continue
        print(f"driver: starting r={n_redundant} seed={seed}", flush=True)
        # Put the child in its own process group / session so that on timeout the entire
        # tree can be killed. PyCCEA spawns a worker pool that a plain kill would miss.
        popen_kwargs = {"start_new_session": True} if os.name == "posix" else {}
        proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__)),
             "--one", "--redundant", str(n_redundant), "--seed", str(seed)],
            **popen_kwargs,
        )
        try:
            proc.wait(timeout=timeout_min * 60)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
            print(f"driver: r={n_redundant} seed={seed} TIMED OUT after {timeout_min} min, "
                  f"killed and skipped", flush=True)

    print("driver: finished", flush=True)
    summarize()


def summarize():
    if not RESULTS_CSV.exists():
        print("no results yet")
        return
    frame = pd.read_csv(RESULTS_CSV)
    cols = ["n_selected", "n_informative", "n_redundant", "n_noise", "precision", "f1", "informative_share"]
    print("\n===== MEANS =====")
    for n_redundant in sorted(frame.n_redundant_gen.unique()):
        sub = frame[frame.n_redundant_gen == n_redundant]
        n = (sub.method == "CCEA (PSTD)").sum()
        print(f"\n--- redundant features in data: {n_redundant}  (n={n} seeds) ---")
        print(sub.groupby("method")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--one", action="store_true", help="worker mode: run a single (dataset, seed)")
    parser.add_argument("--redundant", type=int, help="number of redundant features (worker mode)")
    parser.add_argument("--seed", type=int, help="seed (worker mode)")
    parser.add_argument("--seeds", type=int, default=10, help="seeds per dataset (driver mode)")
    parser.add_argument("--levels", type=str, default="0,50", help="redundancy levels, comma-separated")
    parser.add_argument("--timeout-min", type=int, default=90, help="per-run timeout in minutes")
    parser.add_argument("--summary", action="store_true", help="print current means and exit")
    args = parser.parse_args()

    if args.summary:
        summarize()
    elif args.one:
        run_one(args.redundant, args.seed)
    else:
        drive(args.seeds, [int(x) for x in args.levels.split(",")], args.timeout_min)
