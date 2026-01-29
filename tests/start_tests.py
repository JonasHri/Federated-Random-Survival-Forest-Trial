# %%
from frsf.preprocessing import federate_data, align_schema
from frsf.models import (
    LocalRandomSurvivalForest,
    FederatedRandomSurvivalForest,
)
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.preprocessing import OneHotEncoder

X_base, Y_base = load_veterans_lung_cancer()
client_count = 3
X_splits_small, Y_splits = federate_data(X_base, Y_base, client_count)
X_splits = [align_schema(X, X_base.columns.to_list()) for X in X_splits_small]
Xt_splits = [OneHotEncoder().fit_transform(x_split) for x_split in X_splits]

models = []
for i in range(client_count):
    X, y = Xt_splits[i], Y_splits[i]
    model = LocalRandomSurvivalForest().fit(X, y)
    models.append(model)
    print(f"local prediction of client {i} with {len(model.estimators_)} estimators:")
    print(model.predict(X.iloc[:10]))

# %%

federated_model = FederatedRandomSurvivalForest(models, n_estimators=300)
federated_model.update_local_models()
# example prediction using federated model

print(
    f"federated prediction with {len(federated_model.estimators_)} estimators on data from client 1:"
)
print(federated_model.predict(Xt_splits[0].iloc[:10]))

# preditctions on updated local models
for i, (model, data) in enumerate(zip(models, Xt_splits)):
    print(f"local prediction of client {i} with {len(model.estimators_)} estimators:")
    print(model.predict(data.iloc[:10]))
