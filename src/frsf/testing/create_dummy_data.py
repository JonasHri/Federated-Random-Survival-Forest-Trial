import numpy as np
import pandas as pd
import pytest
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
