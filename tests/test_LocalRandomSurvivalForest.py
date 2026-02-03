# %%
from frsf.models import LocalRandomSurvivalForest
from frsf.testing import create_dummy_data
import numpy as np
import pytest


@pytest.mark.parametrize("n_samples", [2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("n_features", [1, 2, 4, 8, 16, 32])
def test_training(n_samples, n_features):

    X, y = create_dummy_data(n_samples, n_features, drop_feature_percent=0.33)
    dropped_features = set(X.columns[X.loc[0].isna()])

    model = LocalRandomSurvivalForest()

    model.fit(X, y)
    model.predict(X)

    assert model.site_size == len(X), "site size"
    assert model.local_features == (
        set(range(n_features)) - dropped_features
    ), "local_features"
    for estimator in model.estimators_:
        assert estimator.sample_weight == len(X) / 100, "estimator weight"


def test_save_load(tmp_path):

    n_samples, n_features = 128, 32

    X, y = create_dummy_data(n_samples, n_features, drop_feature_percent=0.33)

    model = LocalRandomSurvivalForest()
    model.fit(X, y)

    path = tmp_path / "model.model"
    model.save(path)
    loaded_model = LocalRandomSurvivalForest.load(path)

    np.testing.assert_array_equal(
        model.predict(X),
        loaded_model.predict(X),
        err_msg="predict",
    )
    np.testing.assert_array_equal(
        model.predict_survival_function(X),
        loaded_model.predict_survival_function(X),
        err_msg="predict_survival_function",
    )

    assert model.site_size == loaded_model.site_size, "site_size"
    assert model.local_features == loaded_model.local_features, "local_features"
    for estimator, loaded_estimator in zip(model.estimators_, loaded_model.estimators_):
        assert (
            estimator.sample_weight == loaded_estimator.sample_weight
        ), "sample_weight"


# %%
