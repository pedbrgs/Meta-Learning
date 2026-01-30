import gc
import os
import time
import random
import logging
import argparse
import pandas as pd
import numpy as np
import ast

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

from pyccea.coevolution import CCPSTFG
from pyccea.utils.datasets import DataLoader
from pyccea.evaluation.wrapper import WrapperEvaluation


def calculate_jaccard(selected_features, informative_features) -> float:
    intersection = set(selected_features).intersection(informative_features)
    union = set(selected_features).union(informative_features)
    jaccard_index = len(intersection) / len(union)
    return jaccard_index


def calculate_hit_rate(selected_features, informative_features) -> float:
    intersection = set(selected_features).intersection(informative_features)
    hit_rate = len(intersection)/len(informative_features)
    return hit_rate


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
    data = pd.DataFrame(data_stats)
    data["computational_effort"] = data["num_samples"] + data["num_features"]
    datasets = data.sort_values("computational_effort", ascending=False)["data_path"].values.tolist()
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
    else:
        return pd.DataFrame(columns=["dataset"])


def save_results(results: pd.DataFrame, root_path: str = "results", output_file: str = "gt_experiments.parquet") -> None:
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
            "seed" : random_state
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
            "model_type": "k_nearest_neighbors",
            "use_subprocess": True,
        },
        "evaluation": {
            "fitness_function": "penalty",
            "eval_function": "balanced_accuracy",
            "eval_mode": "k_fold",
            "weights": [1.0, 0.0],
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


def cumulative_standard_error(series: pd.Series) -> pd.Series:
    return series.expanding().std() / series.expanding().count().pow(0.5)


def get_completed_datasets(results: pd.DataFrame, metric_col: str, standard_error_threshold: float, min_runs: int, max_runs: int) -> list:
    if results.empty: return []
    errors = results.groupby("dataset")[metric_col].apply(cumulative_standard_error)
    achieved_errors = errors.groupby(level=0).last()
    run_counts = results.groupby("dataset").size()
    cond_error_met = (achieved_errors <= standard_error_threshold) & (run_counts >= min_runs)
    cond_error_failed = (achieved_errors > standard_error_threshold) & (run_counts >= max_runs)
    completed_mask = cond_error_met | cond_error_failed
    return achieved_errors.index[completed_mask].tolist()


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
    array = np.array([int(float(x)) for x in array_str.split()])
    return array


def get_overall_stats(**kwargs) -> pd.DataFrame:

    total_features = kwargs["ccea"].data.n_features
    features = list(range(int(total_features)))
    n_informative = int(0.01 * total_features)
    informative_features = list(range(int(n_informative)))
    removed_features = kwargs["ccea"].removed_features
    context_vector = kwargs["ccea"].best_context_vector
    feature_indices = kwargs["ccea"].feature_idxs
    remaining_features = np.array(list(set(features).difference(removed_features)))
    selected_features = remaining_features[feature_indices][context_vector.astype(bool)]

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


def check_stopping_criteria(results: pd.DataFrame, args, dataset_name: str, n_runs: int) -> bool:
    metric_series = results[results["dataset"] == dataset_name][args.metric_col]
    if metric_series.empty or len(metric_series) < 2: return False
    errors = cumulative_standard_error(metric_series)
    current_error = errors.iloc[-1]
    if (n_runs >= args.min_runs) and (current_error <= args.se_thresh):
        logging.info(f"Standard error threshold achieved ({errors.iloc[-1]:.2f}%). ")
        logging.info(f"Ending experiments for dataset: {dataset_name}.")
        return True
    if n_runs >= args.max_runs:
        logging.info(f"Maximum number of runs reached ({args.max_runs}). ")
        logging.info(f"Ending experiments for dataset: {dataset_name}.")
        return True
    return False


def run() -> None:

    set_logger()
    args = parse_args()

    datasets = list_datasets(data_dir=args.data_dir, is_debug=args.is_debug)
    if not datasets:
        logging.info("Data folder is empty.")
        return

    logging.info(f"Datasets: {datasets}")
    results = load_results()
    
    completed_datasets = get_completed_datasets(
        results=results, metric_col=args.metric_col,
        standard_error_threshold=args.se_thresh,
        min_runs=args.min_runs, max_runs=args.max_runs
    )
    datasets = [d for d in datasets if d not in completed_datasets]

    for dataset_name in datasets:

        dataset_file = f"{dataset_name}.parquet"
        data_path = os.path.join(args.data_dir, dataset_file)
        logging.info(f"Starting experiments for dataset: {dataset_name}.")

        n_runs = (
            results.loc[results["dataset"] == dataset_name, "run"].max()
            if not results[results["dataset"] == dataset_name].empty
            else 0
        )
        while True:

            n_runs += 1
            random_state = random.randint(0, 10_000)
            logging.info(f"Run #{n_runs} | Random state {random_state}")

            # Load data configuration and dataloader
            data_conf = load_data_conf(random_state=random_state)
            dataloader = build_dataloader(
                data_path=data_path,
                dataset_name=dataset_name,
                data_conf=data_conf
            )
            dataloader.get_ready()
            # Load CCEA configuration
            ccea_conf = load_ccea_conf(random_state=random_state, is_debug=args.is_debug)
            
            # Initialize CCEA
            start_time = time.time()
            ccea = CCPSTFG(conf=ccea_conf, data=dataloader, verbose=False)
            init_time = time.time() - start_time
            logging.info(f"CCEA initialization completed in {(init_time/60):.2f} minutes.")
            # Run feature selection
            start_time = time.time()
            ccea.optimize()
            fs_time = time.time() - start_time
            logging.info(f"Feature selection completed in {(fs_time/60):.2f} minutes.")

            train_metrics = evaluate_context_vector(ccea, subset="train")
            test_metrics = evaluate_context_vector(ccea, subset="test")
            
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
                f"Hit rate (%): {run_stats['hit_rate'].values[0]} | "
                f"Jaccard Index: {run_stats['jaccard_index'].values[0]}"
            )
            
            if check_stopping_criteria(results, args, dataset_name, n_runs):
                break
            
            del dataloader, ccea
            gc.collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help="data directory path")
    parser.add_argument("--metric-col", type=str, help="metric to be monitored for standard error threshold")
    parser.add_argument("--se-thresh", type=float, default=0.03, help="standard error threshold")
    parser.add_argument("--min-runs", type=int, default=5, help="minimum number of runs per dataset")
    parser.add_argument("--max-runs", type=int, default=50, help="maximum number of runs per dataset")
    parser.add_argument("--is-debug", action="store_true", help="debug mode")
    return parser.parse_args()


if __name__ == '__main__':
    run()