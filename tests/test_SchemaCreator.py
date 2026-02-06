from frsf.preprocessing import SchemaCreator, SchemaAligner
from frsf.testing import create_dummy_data, federate_data
import pytest


@pytest.mark.parametrize("random_state", [0, 1, 2, 3, 4, 5, 6, 7])
def test_alignment(random_state):
    n_samples = 1000
    n_features = 10
    X, y = create_dummy_data(n_samples, n_features, random_state=random_state)

    X_list, _ = federate_data(
        X, y, 5, drop_feature_percentage=0.3, random_state=random_state
    )

    schema = SchemaCreator().fit_transform([X_fed.columns for X_fed in X_list])

    X_aligned_list = []

    for X_fed in X_list:
        X_aligned = SchemaAligner().fit_transform(X_fed, schema)
        X_aligned_list.append(X_aligned)

    for X_aligned in X_aligned_list[1:]:
        assert (X_aligned.columns == X_aligned_list[0].columns).all()
