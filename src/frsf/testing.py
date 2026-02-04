import warnings
import numpy as np
import pandas as pd
from typing import Union


def create_dummy_data(
    n_samples: int,
    n_features: int,
    cencor_chance: float = 0.15,
    drop_feature_percentage: float = 0.0,
    random_state: Union[int, np.random.Generator] = None,
) -> tuple[pd.DataFrame, np.ndarray]:

    rng = np.random.default_rng(random_state)

    X = rng.standard_normal(size=(n_samples, n_features))
    cencor_features = list(
        rng.choice(
            n_features,
            int(n_features * drop_feature_percentage),
            replace=False,
        )
    )
    X[:, cencor_features] = np.nan
    X = pd.DataFrame(X)

    dtype = [
        ("Status", "?"),
        ("Survival_in_days", "<f8"),
    ]
    status = rng.standard_normal(size=n_samples) > cencor_chance
    survival = rng.integers(1, 1000, n_samples).astype(float)
    if np.all(status == False):
        status[0] = True

    y = np.zeros(n_samples, dtype=dtype)
    y["Status"] = status
    y["Survival_in_days"] = survival

    return X, y


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
