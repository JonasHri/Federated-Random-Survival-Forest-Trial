from sksurv.ensemble import RandomSurvivalForest
from sksurv.tree import SurvivalTree
import numpy as np
from typing import Literal, Optional, Self
from numpy.typing import ArrayLike
import joblib
import pandas as pd


class LocalRandomSurvivalForest(RandomSurvivalForest):
    def __init__(
        self,
        update_method: Literal["constant", "all"] = "all",
        update_weighting: Literal["equal", "site_size"] = "equal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.local_features: set = set()
        self.federated_estimators: list[SurvivalTree] = []
        self.site_size: int = 0
        self.update_method = update_method
        self.update_weighting = update_weighting
        self.tree_origin: Literal["local", "federated"] = "local"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight: Optional[ArrayLike] = None,
    ) -> Self:
        self.site_size: int = len(X)
        self.local_features = set(X.columns[~X.isna().all()].tolist())
        self.all_features = X.columns.tolist()
        super().fit(X, y, sample_weight=sample_weight)
        for estimator in self.estimators_:
            estimator.sample_weight = self.site_size / self.n_estimators

        return self

    def set_federated_estimators(self, estimators: list[SurvivalTree]):
        self.federated_estimators = estimators
        return self

    def use_local_estimators(self):
        if self.tree_origin == "local":
            return self

        self.tree_origin = "local"

        self.estimators_ = self.local_estimators
        self.n_estimators = len(self.estimators_)

    def use_federated_estimators(self, random_state: Optional[int] = None):
        if self.tree_origin == "federated":
            return self

        if random_state is None:
            random_state = self.random_state

        self.local_estimators = self.estimators_
        self.tree_origin = "federated"

        if self.update_method == "all":
            self.estimators_ = self.federated_estimators

        elif self.update_method == "constant":
            if self.update_weighting == "equal":
                weights: list[float] = [1.0] * len(self.federated_estimators)
            elif self.update_weighting == "site_size":
                weights: list[float] = [
                    estimator.sample_weight for estimator in self.federated_estimators
                ]
            self.estimators_ = np.random.default_rng(random_state).choice(
                self.federated_estimators,
                size=self.n_estimators,
                p=np.array(weights) / sum(weights),
            )

        self.n_estimators = len(self.estimators_)
        return self

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> Self:
        return joblib.load(path)


class FederatedRandomSurvivalForest(RandomSurvivalForest):
    def __init__(
        self,
        local_models: list[LocalRandomSurvivalForest],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.all_features: np.ndarray = local_models[0].all_features

        self.local_models: list[LocalRandomSurvivalForest] = []
        self.estimators_: list[SurvivalTree] = []
        self.tree_features: list[set] = []
        for model in local_models:
            self.add_local_model(model)

    def add_local_model(self, local_model: LocalRandomSurvivalForest):
        local_tree_features = []
        for estimator in local_model.estimators_:
            tree_features = estimator.tree_.feature
            tree_features_names = set(
                [self.all_features[i] for i in tree_features if i >= 0]
            )
            local_tree_features.append(tree_features_names)

        self.local_models.append(local_model)
        self.estimators_.extend(local_model.estimators_)
        self.tree_features.extend(local_tree_features)
        self.n_estimators = len(self.estimators_)
        return self

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Federated model cannot be fit directly.")

    def predict(self, X):
        raise NotImplementedError("Federated model cannot predict directly.")

    def distribute_trees(self, random_state: Optional[int] = None):
        for model in self.local_models:
            valid_estimators = []
            for estimator, feat_set in zip(self.estimators_, self.tree_features):
                # find all estimators that only use features available locally
                if feat_set.issubset(model.local_features):
                    valid_estimators.append(estimator)

            model.set_federated_estimators(valid_estimators)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> Self:
        return joblib.load(path)
