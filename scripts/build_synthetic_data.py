import os
import gc
import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def parse_args():
    """Handles command-line arguments for the synthetic dataset generator."""

    parser = argparse.ArgumentParser(description="Synthetic Dataset Generator for Feature Selection Benchmark")
    
    # Dataset size and ground truth parameters
    parser.add_argument("--n-features", type=int, default=100000, help="Total number of features")
    parser.add_argument("--sample-ratio", type=float, default=0.1, help="Ratio of samples relative to features")
    parser.add_argument("--perc-informative", type=float, default=0.30, help="Percentage of informative features")
    
    # Structural complexity parameters (requested)
    parser.add_argument("--n-redundant", type=int, default=0, help="Number of redundant features (linear combinations)")
    parser.add_argument("--n-repeated", type=int, default=0, help="Number of duplicated features")
    parser.add_argument("--n-clusters-per-class", type=int, default=3, help="Clusters per class (complexity of distribution)")
    parser.add_argument("--n-classes", type=int, default=2, help="Number of target classes")
    
    # Data quality and split parameters
    parser.add_argument("--class-sep", type=float, default=0.8, help="How separable classes are (higher is easier)")
    parser.add_argument("--flip-y", type=float, default=0.01, help="Random noise applied to labels")
    parser.add_argument("--test-size", type=float, default=0.30, help="Proportion of dataset for test split")
    
    # Infrastructure parameters
    parser.add_argument("--output-dir", type=str, default="./data/Benchmark/", help="Output directory path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()


def build_synthetic_dataset(args):
    """Generates a synthetic dataset using the arguments provided via CLI."""

    # Derived calculations
    n_samples = int(args.n_features * args.sample_ratio)
    n_informative = int(args.perc_informative * args.n_features)
    weights = [0.55, 0.45] # Fixed class weights based on your previous requirement

    print(f"--- Configuration ---")
    print(f"Features: {args.n_features} | Samples: {n_samples}")
    print(f"Informative Features (Ground Truth): First {n_informative} columns.")
    print(f"Output Directory: {args.output_dir}")

    # Generate synthetic data
    # shuffle=False is critical to keep informative features at indices [0 : n_informative]
    X, y = make_classification(
        n_samples=n_samples,
        n_features=args.n_features,
        n_informative=n_informative,
        n_redundant=args.n_redundant,
        n_repeated=args.n_repeated,
        n_clusters_per_class=args.n_clusters_per_class,
        n_classes=args.n_classes,
        class_sep=args.class_sep,
        flip_y=args.flip_y,
        weights=weights,
        random_state=args.seed,
        shuffle=False 
    )

    # Convert to DataFrame using float32 to optimize RAM usage on EC2
    df = pd.DataFrame(X).astype("float32")
    df["label"] = y.astype("int8")
    
    # Free up memory from raw numpy arrays
    del X, y
    gc.collect()

    # Stratified split to maintain class balance in both train and test sets
    train_data, test_data = train_test_split(
        df, 
        test_size=args.test_size, 
        random_state=args.seed, 
        stratify=df["label"]
    )

    # Tag subsets
    train_data["subset"] = "train"
    test_data["subset"] = "test"
    
    # Combine for storage
    final_df = pd.concat([train_data, test_data], ignore_index=True)
    
    # Ensure directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    filename = f"synthetic_ground_truth_s{n_samples}_f{args.n_features}.parquet"
    full_path = os.path.join(args.output_dir, filename)
    final_df.to_parquet(full_path, index=False)
    
    print(f"--- Generation Complete ---")
    print(f"Saved to: {full_path}")
    print(f"Memory Usage: {final_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")


if __name__ == '__main__':
    # Parse CLI arguments and run the generator
    cli_args = parse_args()
    build_synthetic_dataset(cli_args)