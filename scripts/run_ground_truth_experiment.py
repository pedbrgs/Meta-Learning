import gc
import os
import time
import random
import logging
import argparse
import pandas as pd
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2 as chi2_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)

from pyccea.coevolution import CCPSTFG
from pyccea.utils.datasets import DataLoader
from pyccea.evaluation.wrapper import WrapperEvaluation


PERC_INFORMATIVE = 0.05


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def calculate_jaccard(selected_features, informative_features) -> float:
    intersection = set(selected_features).intersection(informative_features)
    union = set(selected_features).union(informative_features)
    return len(intersection) / len(union)


def calculate_hit_rate(selected_features, informative_features) -> float:
    intersection = set(selected_features).intersection(informative_features)
    return len(intersection) / len(informative_features)


# ---------------------------------------------------------------------------
# Data / config helpers
# ---------------------------------------------------------------------------

def build_dataloader(data_path: str, dataset_name: str, data_conf: dict) -> DataLoader:
    DataLoader.DATASETS[dataset_name] = {
        "file": data_path,
        "task": "classification"
    }
    return DataLoader(dataset=dataset_name, conf=data_conf)


def list_datasets(data_dir: str, is_debug: bool) -> list:
    data_stats = []
    for file in os.listdir(data_dir):
        if file.endswith(".parquet") and "synthetic_ground_truth" in file:
            data = pd.read_parquet(os.path.join(data_dir, file))
            num_samples, num_features = data.shape
            dataset_name = file.replace(".parquet", "")
            del data
            gc.collect()
            data_stats.append({
                "data_path": dataset_name,
                "num_samples": num_samples,
                "num_features": num_features
            })
    if not data_stats:
        return []
    df = pd.DataFrame(data_stats)
    df["computational_effort"] = df["num_samples"] + df["num_features"]
    datasets = df.sort_values("computational_effort", ascending=False)["data_path"].tolist()
    if is_debug:
        datasets = [datasets[-1]]
    return datasets


def set_logger() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)


def load_results(root_path: str = "results", output_file: str = "gt_experiments.parquet") -> pd.DataFrame:
    os.makedirs(root_path, exist_ok=True)
    file_path = os.path.join(root_path, output_file)
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return pd.DataFrame(columns=["dataset"])


def save_results(results: pd.DataFrame, root_path: str = "results", output_file: str = "gt_experiments.parquet") -> None:
    os.makedirs(root_path, exist_ok=True)
    results.to_parquet(os.path.join(root_path, output_file), index=False)


def load_baseline_results(root_path: str = "results", output_file: str = "gt_baselines.parquet") -> pd.DataFrame:
    os.makedirs(root_path, exist_ok=True)
    file_path = os.path.join(root_path, output_file)
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return pd.DataFrame(columns=["dataset", "method", "run"])


def save_baseline_results(results: pd.DataFrame, root_path: str = "results", output_file: str = "gt_baselines.parquet") -> None:
    os.makedirs(root_path, exist_ok=True)
    results.to_parquet(os.path.join(root_path, output_file), index=False)


def load_data_conf(random_state: int) -> dict:
    return {
        "general": {
            "splitter_type": "k_fold",
            "verbose": True,
            "float_dtype": "float32",
            "seed": random_state
        },
        "splitter": {
            "preset": True,
            "kfolds": 10,
            "prefold": False
        },
        "normalization": {
            "normalize": True,
            "method": "min_max"
        },
        "preprocessing": {
            "drop_na": True,
            "winsorization": False,
            "quantiles": [0.01, 0.99]
        }
    }


def load_ccea_conf(random_state: int, is_debug: bool) -> dict:
    return {
        "coevolution": {
            "subpop_sizes": [30],
            "max_gen": 1 if is_debug else 1000,
            "max_gen_without_improvement": 2 if is_debug else 10,
            "optimized_resource_allocation": False,
            "max_best_context_vectors": 0,
            "seed": random_state
        },
        "decomposition": {
            "method": "clustering",
            "drop": True,
            "max_n_clusters": 10,
            "max_n_pls_components": 10,
            "removal_quantile_step_size": 0.05,
            "max_removal_quantile": 0.50,
            "clustering_model_type": "agglomerative_clustering"
        },
        "collaboration": {
            "method": "best"
        },
        "wrapper": {
            "task": "classification",
            "cache_size": 2000,
            "model_type": "logistic_regression",
            "use_subprocess": True,
        },
        "evaluation": {
            "fitness_function": "penalty",
            "eval_function": "balanced_accuracy",
            "eval_mode": "k_fold",
            "weights": [0.9, 0.1],
            "n_workers": 6
        },
        "optimizer": {
            "method": "GA",
            "selection_method": "generational",
            "mutation_rate": 0.05,
            "crossover_rate": 1.0,
            "tournament_sample_size": 1,
            "elite_size": 1
        }
    }


# ---------------------------------------------------------------------------
# Stopping criteria
# ---------------------------------------------------------------------------

def cumulative_standard_error(series: pd.Series) -> pd.Series:
    return series.expanding().std() / series.expanding().count().pow(0.5)


def get_completed_datasets(
        results: pd.DataFrame,
        metric_col: str,
        standard_error_threshold: float,
        min_runs: int,
        max_runs: int
    ) -> list:
    errors = results.groupby("dataset")[metric_col].apply(cumulative_standard_error)
    achieved_errors = errors.groupby(level=0).last()
    run_counts = results.groupby("dataset").size()
    cond_error_met = (achieved_errors <= standard_error_threshold) & (run_counts >= min_runs)
    cond_error_failed = (achieved_errors > standard_error_threshold) & (run_counts >= max_runs)
    completed_mask = cond_error_met | cond_error_failed
    return achieved_errors.index[completed_mask].tolist()


def check_stopping_criteria(results: pd.DataFrame, args, dataset_name: str, n_runs: int) -> bool:
    metric_series = results[results["dataset"] == dataset_name][args.metric_col]
    if metric_series.empty:
        return False
    errors = cumulative_standard_error(metric_series)
    if (n_runs >= args.min_runs) and (errors.iloc[-1] <= args.se_thresh):
        logging.info(f"Standard error threshold achieved ({errors.iloc[-1]:.2f}%).")
        logging.info(f"Ending experiments for dataset: {dataset_name}.")
        return True
    if n_runs >= args.max_runs:
        logging.info(f"Maximum number of runs reached ({args.max_runs}).")
        logging.info(f"Ending experiments for dataset: {dataset_name}.")
        return True
    return False


# ---------------------------------------------------------------------------
# CCEA evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_context_vector(ccea, subset: str) -> pd.DataFrame:
    evaluator = WrapperEvaluation(
        task=ccea.conf["wrapper"]["task"],
        model_type=ccea.conf["wrapper"]["model_type"],
        eval_function=ccea.conf["evaluation"]["eval_function"],
        eval_mode="k_fold" if subset == "train" else "hold_out",
        n_classes=ccea.data.n_classes
    )
    _ = evaluator.evaluate(solution=ccea.best_context_vector.copy(), data=ccea.data)
    metrics = pd.DataFrame.from_dict(evaluator.evaluations, orient="index").transpose()
    metrics.columns = [f"{subset}_{col}" for col in metrics.columns]
    return metrics


def literal_eval(array_str: str) -> np.ndarray:
    array_str = array_str.replace('[', '').replace(']', '').replace(',', ' ')
    return np.array([int(float(x)) for x in array_str.split()])


def get_overall_stats(**kwargs) -> pd.DataFrame:
    total_features = int(kwargs["ccea"].data.n_features)
    n_informative = int(PERC_INFORMATIVE * total_features)
    informative_features = np.arange(n_informative)

    removed_features = np.asarray(kwargs["ccea"].removed_features, dtype=int)
    removed_informative = np.intersect1d(removed_features, informative_features)
    logging.info(
        f"VIP removed {len(removed_informative)}/{n_informative} informative features "
        f"({100*len(removed_informative)/n_informative:.1f}%) — "
        f"max achievable hit rate: {100*(n_informative - len(removed_informative))/n_informative:.1f}%"
    )

    context_vector = kwargs["ccea"].best_context_vector.astype(bool)
    feature_indices = getattr(kwargs["ccea"], "best_feature_idxs", kwargs["ccea"].feature_idxs)
    # feature_indices are positions in the post-removal feature space (VIP drops features
    # before the search), so they must be mapped back to original indices through the kept
    # set before comparing with the informative ground truth. Indexing the informative
    # ground truth with feature_indices directly counts post-removal positions as if they
    # were original indices, which inflates the hit rate and Jaccard index.
    kept = np.setdiff1d(np.arange(total_features), removed_features)
    selected_features = kept[np.asarray(feature_indices)][context_vector]

    jaccard_index = calculate_jaccard(selected_features, informative_features)
    hit_rate = calculate_hit_rate(selected_features, informative_features)

    run_stats = {
        "dataset": kwargs["dataset_name"],
        "total_samples": kwargs["ccea"].data.n_examples,
        "total_features": kwargs["ccea"].data.n_features,
        "run": kwargs["run"],
        "n_informative": n_informative,
        "hit_rate": round(hit_rate, 4),
        "jaccard_index": round(jaccard_index, 4),
        "n_subcomps": kwargs["ccea"].n_subcomps,
        "subcomp_sizes": str(kwargs["ccea"].subcomp_sizes),
        "subpop_sizes": str(kwargs["ccea"].subpop_sizes),
        "ccea_conf": str(kwargs["ccea"].conf),
        "data_conf": str(kwargs["ccea"].data.conf),
        "selected_features": selected_features.tolist(),
        "feature_indices": str(kwargs["ccea"].best_feature_idxs),
        "best_context_vector": str(kwargs["ccea"].best_context_vector),
        "best_fitness": round(kwargs["ccea"].best_fitness, 4),
        "convergence_curve": [round(f, 4) for f in kwargs["ccea"].convergence_curve],
        "quantile_to_remove": kwargs["ccea"].quantile_to_remove,
        "n_pls_components": kwargs["ccea"].n_components,
        "vip_threshold": kwargs["ccea"].vip_threshold,
        "removed_features": str(kwargs["ccea"].removed_features),
        "n_iterations": len(kwargs["ccea"].convergence_curve),
        "n_selected_features": kwargs["ccea"].best_context_vector.sum(),
        "n_pre_removed_features": len(kwargs["ccea"].removed_features),
        "init_time": kwargs["init_time"],
        "tuning_time": kwargs["ccea"]._tuning_time,
        "feature_selection_time": kwargs["fs_time"]
    }
    return pd.DataFrame.from_dict(run_stats, orient="index").T


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------

def get_k_candidates(n_features: int) -> np.ndarray:
    max_k = min(n_features, 2000)
    return np.unique(np.logspace(0, np.log10(max_k), 20).astype(int))


def run_filter_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scorer,
    k_candidates: np.ndarray,
    n_splits: int,
    random_state: int
) -> np.ndarray:
    """Score features once, then tune k via cross-validation. Returns sorted selected indices."""
    raw = scorer(X_train, y_train)
    # chi2 returns (statistic, p_value); mutual_info returns scores directly
    feature_scores = raw[0] if isinstance(raw, tuple) else raw
    ranked_idx = np.argsort(feature_scores)[::-1]  # best features first

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=1)

    best_k, best_score = int(k_candidates[0]), -np.inf
    for k in k_candidates:
        top_idx = ranked_idx[:int(k)]
        preds = cross_val_predict(model, X_train[:, top_idx], y_train, cv=cv)
        score = balanced_accuracy_score(y_train, preds)
        if score > best_score:
            best_score = score
            best_k = int(k)

    return np.sort(ranked_idx[:best_k])


def run_random_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_candidates: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int
) -> np.ndarray:
    """Tune k via CV with random subsets (averaged over n_repeats). Returns sorted selected indices."""
    rng = np.random.default_rng(random_state)
    n_features = X_train.shape[1]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=1)

    best_k, best_score = int(k_candidates[0]), -np.inf
    for k in k_candidates:
        repeat_scores = []
        for _ in range(n_repeats):
            idx = rng.choice(n_features, size=int(k), replace=False)
            preds = cross_val_predict(model, X_train[:, idx], y_train, cv=cv)
            repeat_scores.append(balanced_accuracy_score(y_train, preds))
        avg = float(np.mean(repeat_scores))
        if avg > best_score:
            best_score = avg
            best_k = int(k)

    return np.sort(rng.choice(n_features, size=best_k, replace=False))


def compute_baseline_clf_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_idx: np.ndarray,
    n_splits: int,
    random_state: int
) -> dict:
    """Compute train (k-fold CV) and test classification metrics for a given feature subset."""
    model = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=1)
    X_tr = X_train[:, selected_idx]
    X_te = X_test[:, selected_idx]
    n_classes = len(np.unique(y_train))
    avg = "binary" if n_classes == 2 else "macro"

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_preds = cross_val_predict(model, X_tr, y_train, cv=cv)

    model.fit(X_tr, y_train)
    test_preds = model.predict(X_te)

    def _metrics(y_true, y_pred, prefix):
        spec = (
            recall_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
            if avg == "binary" else np.nan
        )
        row = {
            f"{prefix}_accuracy":           round(float(accuracy_score(y_true, y_pred)), 4),
            f"{prefix}_balanced_accuracy":  round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            f"{prefix}_precision":          round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            f"{prefix}_recall":             round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            f"{prefix}_f1_score":           round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            f"{prefix}_specificity":        round(float(spec), 4) if not np.isnan(spec) else None,
        }
        return row

    return {**_metrics(y_train, train_preds, "train"), **_metrics(y_test, test_preds, "test")}


def run_baselines(
    dataloader: DataLoader,
    n_informative: int,
    dataset_name: str,
    run: int,
    random_state: int
) -> pd.DataFrame:
    X_train = dataloader.X_train
    y_train = dataloader.y_train.ravel()
    X_test  = dataloader.X_test
    y_test  = dataloader.y_test.ravel()

    n_features = X_train.shape[1]
    n_splits   = 10
    informative_features = np.arange(n_informative)
    k_candidates = get_k_candidates(n_features)

    methods = [
        ("mutual_info", lambda: run_filter_baseline(X_train, y_train, mutual_info_classif, k_candidates, n_splits, random_state)),
        ("chi2",        lambda: run_filter_baseline(X_train, y_train, chi2_score,          k_candidates, n_splits, random_state)),
        ("random",      lambda: run_random_baseline(X_train, y_train, k_candidates, n_splits, 5, random_state)),
    ]

    rows = []
    for method_name, select_fn in methods:
        logging.info(f"Running baseline: {method_name}")
        start = time.time()
        selected_idx = select_fn()
        fs_time = time.time() - start

        clf_metrics = compute_baseline_clf_metrics(
            X_train, y_train, X_test, y_test, selected_idx, n_splits, random_state
        )
        hit_rate     = calculate_hit_rate(selected_idx, informative_features)
        jaccard_index = calculate_jaccard(selected_idx, informative_features)

        rows.append({
            "dataset":              dataset_name,
            "method":               method_name,
            "run":                  run,
            "n_informative":        n_informative,
            "total_features":       n_features,
            "total_samples":        len(y_train) + len(y_test),
            "n_selected_features":  len(selected_idx),
            "selected_features":    selected_idx.tolist(),
            "hit_rate":             round(hit_rate, 4),
            "jaccard_index":        round(jaccard_index, 4),
            "feature_selection_time": round(fs_time, 4),
            **clf_metrics,
        })
        logging.info(
            f"[{method_name}] k={len(selected_idx)} | "
            f"hit_rate={hit_rate:.4f} | jaccard={jaccard_index:.4f}"
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main loop helpers
# ---------------------------------------------------------------------------

def _parse_seed_from_conf(conf_str: str) -> int:
    """Extract the seed value from a stored data_conf string."""
    import ast
    return int(ast.literal_eval(conf_str)["general"]["seed"])


def _baselines_complete(baselines: pd.DataFrame, results: pd.DataFrame, dataset_name: str) -> bool:
    """True when all 3 baseline methods have been run for every existing CCEA run."""
    if results.empty or baselines.empty or "run" not in baselines.columns:
        return False
    ccea_runs = set(results.loc[results["dataset"] == dataset_name, "run"].tolist())
    if not ccea_runs:
        return False
    for method in ("mutual_info", "chi2", "random"):
        done = set(
            baselines.loc[
                (baselines["dataset"] == dataset_name) & (baselines["method"] == method),
                "run"
            ].tolist()
        )
        if not ccea_runs.issubset(done):
            return False
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    set_logger()
    args = parse_args()

    datasets = list_datasets(data_dir=args.data_dir, is_debug=args.is_debug)
    if not datasets:
        logging.info("Data folder is empty.")
        return

    logging.info(f"Datasets: {datasets}.")
    results   = load_results()
    baselines = load_baseline_results()

    # A dataset is fully completed only when CCEA stopping criteria are met
    # AND baselines have been run for every CCEA run.
    if not results.empty:
        ccea_completed = get_completed_datasets(
            results=results,
            metric_col=args.metric_col,
            standard_error_threshold=args.se_thresh,
            min_runs=args.min_runs,
            max_runs=args.max_runs
        )
        fully_completed = [
            d for d in ccea_completed
            if _baselines_complete(baselines, results, d)
        ]
        datasets = [d for d in datasets if d not in fully_completed]
        logging.info(f"Fully completed datasets: {fully_completed}.")
    logging.info(f"Datasets for experiments: {datasets}.")

    for dataset_name in datasets:
        dataset_file = f"{dataset_name}.parquet"
        data_path = os.path.join(args.data_dir, dataset_file)
        logging.info(f"Starting experiments for dataset: {dataset_name}.")

        # Start from the last run that already has baselines (0 if none yet).
        baseline_ds = baselines[
            (baselines["dataset"] == dataset_name) & (baselines["method"] == "mutual_info")
        ]
        n_runs = int(baseline_ds["run"].max()) if not baseline_ds.empty else 0

        while True:
            n_runs += 1

            # Check whether CCEA already ran this run (it may be ahead of baselines).
            existing_ccea = results[
                (results["dataset"] == dataset_name) & (results["run"] == n_runs)
            ]
            ccea_already_done = not existing_ccea.empty

            if ccea_already_done:
                # Reuse the exact same seed so the dataloader is identical.
                random_state = _parse_seed_from_conf(existing_ccea["data_conf"].values[0])
                logging.info(f"Run #{n_runs} | CCEA already done | reusing seed {random_state}")
            else:
                random_state = random.randint(0, 10_000)
                logging.info(f"Run #{n_runs} | Random state {random_state}")

            # Load data (same seed → same normalisation and splits as original run).
            data_conf  = load_data_conf(random_state=random_state)
            dataloader = build_dataloader(
                data_path=data_path,
                dataset_name=dataset_name,
                data_conf=data_conf
            )
            dataloader.get_ready()
            n_informative = int(PERC_INFORMATIVE * dataloader.n_features)

            # --- CCEA (skip if already stored) ---
            if not ccea_already_done:
                ccea_conf  = load_ccea_conf(random_state=random_state, is_debug=args.is_debug)
                start_time = time.time()
                ccea = CCPSTFG(conf=ccea_conf, data=dataloader, verbose=False)
                init_time  = time.time() - start_time
                logging.info(f"CCEA initialization completed in {init_time/60:.2f} minutes.")

                start_time = time.time()
                ccea.optimize()
                fs_time = time.time() - start_time
                logging.info(f"Feature selection completed in {fs_time/60:.2f} minutes.")

                train_metrics = evaluate_context_vector(ccea, subset="train")
                test_metrics  = evaluate_context_vector(ccea, subset="test")
                run_stats = get_overall_stats(
                    dataset_name=dataset_name,
                    ccea=ccea,
                    run=n_runs,
                    init_time=init_time,
                    fs_time=fs_time
                )
                run_results = pd.concat([run_stats, train_metrics, test_metrics], axis=1)
                results = pd.concat([results, run_results], ignore_index=True)
                save_results(results=results)
                logging.info(
                    f"[CCPSTFG] hit_rate={run_stats['hit_rate'].values[0]} | "
                    f"jaccard={run_stats['jaccard_index'].values[0]}"
                )
                del ccea

            # --- Baselines (skip methods already stored for this run) ---
            done_methods = set(
                baselines.loc[
                    (baselines["dataset"] == dataset_name) & (baselines["run"] == n_runs),
                    "method"
                ].tolist()
            )
            if done_methods < {"mutual_info", "chi2", "random"}:
                # The CCEA above mutates dataloader.X_train in place (VIP prune + feature
                # permutation), so the baselines must not reuse it: SelectKBest would run on
                # the reduced, reordered matrix and return indices in that space rather than
                # in the original one. Rebuild a fresh loader; the same seed reproduces the
                # identical preprocessing and splits.
                baseline_loader = build_dataloader(
                    data_path=data_path,
                    dataset_name=dataset_name,
                    data_conf=data_conf
                )
                baseline_loader.get_ready()
                baseline_run = run_baselines(
                    dataloader=baseline_loader,
                    n_informative=n_informative,
                    dataset_name=dataset_name,
                    run=n_runs,
                    random_state=random_state
                )
                del baseline_loader
                baseline_run = baseline_run[~baseline_run["method"].isin(done_methods)]
                baselines = pd.concat([baselines, baseline_run], ignore_index=True)
                save_baseline_results(baselines)

            del dataloader
            gc.collect()

            if check_stopping_criteria(results, args, dataset_name, n_runs):
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    type=str,   help="data directory path")
    parser.add_argument("--metric-col",  type=str,   help="metric to monitor for standard error threshold")
    parser.add_argument("--se-thresh",   type=float, default=0.03,  help="standard error threshold")
    parser.add_argument("--min-runs",    type=int,   default=5,     help="minimum number of runs per dataset")
    parser.add_argument("--max-runs",    type=int,   default=50,    help="maximum number of runs per dataset")
    parser.add_argument("--is-debug",    action="store_true",       help="debug mode")
    return parser.parse_args()


if __name__ == '__main__':
    run()
