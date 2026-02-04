import pandas as pd
import numpy as np
import warnings
from typing import Union


def federate_data(
    X: pd.DataFrame,
    y: np.ndarray,
    clients: Union[int, list[int]],
    drop_feature_percentage: float = 0.3,
    random_state: int = None,
):
    rng = np.random.default_rng(random_state)

    if isinstance(clients, int):
        client_sizes = [len(X) // clients] * (clients - 1)
        client_sizes.append(len(X) - sum(client_sizes))
    elif isinstance(clients, list) and all(isinstance(x, int) for x in clients):
        client_sizes = clients
    else:
        raise TypeError(f"Argument must be int or list[int], not {type(clients)}")

    if sum(client_sizes) < len(X):
        warnings.warn(
            f"Sum of client sizes ({sum(client_sizes)}) is less than sample count in data ({len(X)}).\nSome samples will be left out."
        )
    if sum(client_sizes) > len(X):
        raise ValueError(
            f"Sum of client sizes ({sum(client_sizes)}) is greater than sample count in data ({len(X)}).\nReduce client sizes to match sample count in data."
        )

    idx = list(X.index)
    rng.shuffle(idx)
    X = X.loc[idx].reset_index(drop=True)
    y = y[idx]

    X_splits: list[pd.DataFrame] = []
    y_splits: list[np.ndarray] = []

    current_index = 0

    for client_size in client_sizes:
        start = current_index
        end = current_index + client_size
        current_index += client_size

        X_cur = X.iloc[start:end]
        cols_to_drop = X_cur.columns.to_series().sample(
            frac=drop_feature_percentage,
            random_state=random_state,
        )

        X_cur = X_cur.drop(columns=cols_to_drop)
        X_splits.append(X_cur)
        y_splits.append(y[start:end])
    return X_splits, y_splits
