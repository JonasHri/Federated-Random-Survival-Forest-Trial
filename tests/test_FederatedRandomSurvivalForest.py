from frsf.models import LocalRandomSurvivalForest, FederatedRandomSurvivalForest
from frsf.testing import create_dummy_data, federate_data
from frsf.preprocessing import align_schema
import pytest
import numpy as np


@pytest.mark.parametrize("n_samples", [2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("n_features", [1, 2, 4, 8, 16])
def test_fit_predict(n_samples, n_features):
    random_state = 0
    n_clients = 3
    X, y = create_dummy_data(
        n_samples * n_clients,
        n_features,
        random_state=random_state,
    )
    X_list, y_list = federate_data(
        X,
        y,
        n_clients,
        random_state=random_state,
    )
    X_list = [align_schema(X_fed, X.columns) for X_fed in X_list]

    local_models = []

    for X_fed, y_fed in zip(X_list, y_list):
        if np.all(y_fed["Status"] == False):
            y_fed["Status"][0] = True
        local_model = LocalRandomSurvivalForest(random_state=random_state)
        local_model = local_model.fit(X_fed, y_fed)

    return
