import pandas as pd
from random import shuffle


def federate_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_clients: int,
    drop_cols_precentage: float = 0.3,
):
    idx = list(X.index)
    shuffle(idx)
    X = X.loc[idx].reset_index(drop=True)
    y = y[idx]

    X_splits = []
    y_splits = []
    n_samples = len(X)
    split_size = n_samples // n_clients

    for i in range(n_clients):
        start = i * split_size
        end = (i + 1) * split_size if i != n_clients - 1 else n_samples
        X_cur = X.iloc[start:end]
        cols_to_drop = X_cur.columns.to_series().sample(
            frac=drop_cols_precentage,
        )

        X_cur = X_cur.drop(columns=cols_to_drop)
        X_splits.append(X_cur)
        y_splits.append(y[start:end])
    return X_splits, y_splits
