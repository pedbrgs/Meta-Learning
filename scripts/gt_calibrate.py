"""Calibrate a synthetic dataset for the ground-truth feature selection experiment.

The dataset must leave room for feature selection to matter. If a classifier reaches
near-perfect accuracy on the full feature set, the wrapper objective is saturated and
carries no gradient towards the informative features, which is what happened in the
previous run. The calibration therefore checks three quantities per candidate setting:

    all      balanced accuracy using every feature
    signal   balanced accuracy using only the informative features
    noise    balanced accuracy using an equally sized random set of noise features

A usable setting has signal clearly above all, and noise close to chance.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

N_FEATURES = 1000
N_INFORMATIVE = 50
SEED = 42


def generate(n_samples, class_sep, flip_y, n_clusters_per_class, seed=SEED):
    """Informative features occupy columns [0:N_INFORMATIVE] because shuffle is disabled."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=flip_y,
        shuffle=False,
        random_state=seed,
    )
    return X, y


def score(X, y, columns, seed=SEED):
    """Mean balanced accuracy of a 1-NN over stratified 10-fold cross-validation."""
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in splitter.split(X, y):
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X[train_idx][:, columns])
        X_val = scaler.transform(X[val_idx][:, columns])
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X_train, y[train_idx])
        scores.append(balanced_accuracy_score(y[val_idx], model.predict(X_val)))
    return float(np.mean(scores))


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    informative = np.arange(N_INFORMATIVE)
    noise = rng.choice(np.arange(N_INFORMATIVE, N_FEATURES), N_INFORMATIVE, replace=False)
    every = np.arange(N_FEATURES)

    print(f"{'n':>5} {'sep':>5} {'flip':>5} {'clus':>5} | {'all':>6} {'signal':>6} {'noise':>6} "
          f"| {'gap':>6}")
    print("-" * 62)
    for n_samples in (300, 500):
        for class_sep in (0.3, 0.5, 0.8):
            for flip_y in (0.0, 0.05):
                X, y = generate(n_samples, class_sep, flip_y, n_clusters_per_class=2)
                s_all = score(X, y, every)
                s_sig = score(X, y, informative)
                s_noi = score(X, y, noise)
                print(f"{n_samples:>5} {class_sep:>5} {flip_y:>5} {2:>5} | "
                      f"{s_all:>6.3f} {s_sig:>6.3f} {s_noi:>6.3f} | {s_sig - s_all:>6.3f}")
