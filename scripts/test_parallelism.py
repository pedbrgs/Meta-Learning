import gc
import os
import time
import random
import logging
import argparse
import pandas as pd

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

from pyccea.coevolution import CCPSTFG
from pyccea.utils.datasets import DataLoader
from pyccea.evaluation.wrapper import WrapperEvaluation


def build_dataloader(data_path: str, dataset_name: str, data_conf: dict) -> DataLoader:
    DataLoader.DATASETS[dataset_name] = {
        "file": data_path,
        "task": "classification"
    }
    dataloader = DataLoader(
        dataset=dataset_name,
        conf=data_conf
    )
    return dataloader


# Initialize the logger globally
logger = logging.getLogger("CCEA_Experiment")

def set_logger() -> None:
    """Configures the global logger."""
    logger.setLevel(logging.INFO)

    # Clear handlers to prevent duplicate logs if function is called twice
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    # Crucial: Stop logs from being intercepted/silenced by the root logger
    logger.propagate = False

# Initialize the logger configuration
set_logger()

def load_results(root_path: str = "results", output_file: str = "experiments.parquet") -> pd.DataFrame:
    os.makedirs(root_path, exist_ok=True)
    file_path = os.path.join(root_path, output_file)
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        return pd.DataFrame(columns=["dataset"])


def save_results(results: pd.DataFrame, root_path: str = "results", output_file: str = "experiments.parquet") -> None:
    os.makedirs(root_path, exist_ok=True)
    file_path = os.path.join(root_path, output_file)
    results.to_parquet(file_path, index=False)


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
            "kfolds": 3,
            "prefold": False
        },
        "normalization": {
            "normalize": True,
            "method": "standard"
        },
        "preprocessing": {
            "drop_na": True,
            "winsorization": True,
            "quantiles": [0.01, 0.99]
        }
    }


def load_ccea_conf(random_state: int, is_debug: bool, n_workers: int) -> dict:
    return {
        "coevolution": {
            "subpop_sizes": [50],
            "max_gen": 1 if is_debug else 1000,
            "max_gen_without_improvement": 2 if is_debug else 10,
            "optimized_resource_allocation": True,
            "max_best_context_vectors": 0,
            "seed" : random_state
        },
        "decomposition": {
            "method": "clustering",
            "drop": True,
            "max_n_clusters": 10,
            "max_n_pls_components": 10,
            "removal_quantile_step_size": 0.05,
            "max_removal_quantile": 0.95,
            "clustering_model_type": "agglomerative_clustering"
        },
        "collaboration": {
            "method": "best"
        },
        "wrapper": {
            "task": "classification",
            "cache_size": 2000,
            "model_type": "random_forest",
            "use_subprocess": True,
        },
        "evaluation": {
            "fitness_function": "penalty",
            "eval_function": "balanced_accuracy",
            "eval_mode": "k_fold",
            "weights": [1.0, 0.0],
            "n_workers": n_workers
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


def cumulative_standard_error(series: pd.Series) -> pd.Series:
    return series.expanding().std() / series.expanding().count().pow(0.5)


def get_completed_tasks(results: pd.DataFrame, args) -> set:
    if results.empty or "n_workers" not in results.columns:
        return set()

    def check_group(group):
        errors = cumulative_standard_error(group[args.metric_col])
        last_error = errors.iloc[-1]
        n_runs = len(group)

        cond_met = (last_error <= args.se_thresh) and (n_runs >= args.min_runs)
        cond_max = (n_runs >= args.max_runs)
        return cond_met or cond_max

    status = results.groupby(["dataset", "n_workers"]).apply(check_group)
    completed = status[status == True].index.tolist()
    return set(completed)


def evaluate_context_vector(ccea, subset: str) -> pd.DataFrame:
    evaluator = WrapperEvaluation(
        task=ccea.conf["wrapper"]["task"],
        model_type=ccea.conf["wrapper"]["model_type"],
        eval_function=ccea.conf["evaluation"]["eval_function"],
        eval_mode="k_fold" if subset == "train" else "hold_out",
        n_classes=ccea.data.n_classes
    )
    _ = evaluator.evaluate(
        solution=ccea.best_context_vector.copy(),
        data=ccea.data
    )
    metrics = pd.DataFrame.from_dict(
        evaluator.evaluations,
        orient="index"
    ).transpose()
    metrics.columns = [f"{subset}_{col}" for col in metrics.columns]
    del evaluator
    gc.collect()
    return metrics


def get_overall_stats(**kwargs) -> dict:
    run_stats = {
        "dataset": kwargs["dataset_name"],
        "n_workers": kwargs["n_workers"],
        "total_samples": kwargs["ccea"].data.n_examples,
        "total_features": kwargs["ccea"].data.n_features,
        "run": kwargs["run"],
        "n_subcomps": kwargs["ccea"].n_subcomps,
        "subcomp_sizes": str(kwargs["ccea"].subcomp_sizes),
        "subpop_sizes": str(kwargs["ccea"].subpop_sizes),
        "ccea_conf": str(kwargs["ccea"].conf),
        "data_conf": str(kwargs["ccea"].data.conf),
        "feature_indices": str(kwargs["ccea"].best_feature_idxs),
        "best_context_vector": str(kwargs["ccea"].best_context_vector),
        "best_fitness": round(kwargs["ccea"].best_fitness, 4),
        "convergence_curve": [round(fitness, 4) for fitness in kwargs["ccea"].convergence_curve],
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


def check_stopping_criteria(results: pd.DataFrame, args: dict, dataset_name: str, n_runs: int, n_workers: int) -> bool:
    mask = (results["dataset"] == dataset_name) & (results["n_workers"] == n_workers)
    metric_series = results[mask][args.metric_col]
    
    if metric_series.empty:
        return False

    errors = cumulative_standard_error(metric_series)
    last_error = errors.iloc[-1]
    
    if (n_runs >= args.min_runs) and (last_error <= args.se_thresh):
        logger.info(f"Stopping threshold reached for {dataset_name} (workers: {n_workers}): {last_error:.4f}")
        return True
    if n_runs >= args.max_runs:
        logger.info(f"Maximum number of runs reached for {dataset_name} (workers: {n_workers})")
        return True
    return False


def run(args: dict) -> None:

    set_logger()

    datasets = ["swarm_behaviour_aligned", "pcam"]
    workers_range = list(range(1, args.max_workers + 1))

    logger.info(f"Datasets: {datasets}.")
    results = load_results(output_file="parallel_scaling_results.parquet")
    completed_tasks = get_completed_tasks(results, args)

    for dataset_name in datasets:

        dataset_file = f"{dataset_name}.parquet"
        data_path = os.path.join(args.data_dir, dataset_file)
        logger.info(f"Starting experiments for dataset: {dataset_name}.")

        for nw in workers_range:

            if (dataset_name, nw) in completed_tasks:
                logger.info(f"Skipping {dataset_name} with {nw} workers: criterion already satisfied.")
                continue

            logger.info(f"Experiment: {dataset_name} | workers: {nw}")

            if not results.empty and "n_workers" in results.columns:
                mask = (results["dataset"] == dataset_name) & (results["n_workers"] == nw)
                n_runs = results[mask]["run"].max() if not results[mask].empty else 0
            else:
                n_runs = 0

            while True:
                n_runs += 1
                random_state = random.randint(0, 10_000)
                logger.info(f"Run #{n_runs} for {dataset_name} (workers = {nw})")

                data_conf = load_data_conf(random_state=random_state)
                dataloader = build_dataloader(data_path, dataset_name, data_conf)
                dataloader.get_ready()

                ccea_conf = load_ccea_conf(random_state, args.is_debug, n_workers=nw)

                start_init = time.time()
                ccea = CCPSTFG(conf=ccea_conf, data=dataloader, verbose=False)
                init_time = time.time() - start_init

                start_fs = time.time()
                ccea.optimize()
                fs_time = time.time() - start_fs

                train_metrics = evaluate_context_vector(ccea, subset="train")
                test_metrics = evaluate_context_vector(ccea, subset="test")

                run_stats = get_overall_stats(
                    dataset_name=dataset_name,
                    n_workers=nw,
                    ccea=ccea,
                    run=n_runs,
                    init_time=init_time,
                    fs_time=fs_time
                )

                run_results = pd.concat([run_stats, train_metrics, test_metrics], axis=1)
                results = pd.concat([results, run_results], ignore_index=True)
                save_results(results, output_file="parallel_scaling_results.parquet")

                del dataloader, ccea
                gc.collect()

                if check_stopping_criteria(results, args, dataset_name, n_runs, nw):
                    break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--metric-col", type=str, default="test_balanced_accuracy")
    parser.add_argument("--se-thresh", type=float, default=0.03)
    parser.add_argument("--min-runs", type=int, default=5)
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--is-debug", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)