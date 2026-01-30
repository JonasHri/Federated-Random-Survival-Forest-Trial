from sksurv.ensemble import RandomSurvivalForest
from sksurv.tree import SurvivalTree
from .LocalRandomSurvivalForest import LocalRandomSurvivalForest
from random import sample
import numpy as np
from typing import Literal, Self
import joblib


class FederatedRandomSurvivalForest(RandomSurvivalForest):
    def __init__(
        self,
        local_models: list[LocalRandomSurvivalForest],
        local_update_method: Literal["constant", "all"] = "all",
        local_update_weighting: Literal["equal", "site_size"] = "equal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.local_update_method = local_update_method
        self.local_update_weighting = local_update_weighting
        self.feature_names_in_: np.ndarray = local_models[0].feature_names_in_

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
                [self.feature_names_in_[i] for i in tree_features if i >= 0]
            )
            local_tree_features.append(tree_features_names)

        self.local_models.append(local_model)
        self.estimators_.extend(local_model.estimators_)
        self.tree_features.extend(local_tree_features)
        self.n_estimators = len(self.estimators_)
        return self

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Federated model cannot be fit directly.")

    def update_local_models(self, random_state: int = None):
        for model in self.local_models:
            valid_estimators = []
            for estimator, feat_set in zip(self.estimators_, self.tree_features):
                # find all estimators that only use features available locally
                if feat_set.issubset(model.local_features):
                    valid_estimators.append(estimator)

            if self.local_update_method == "all":
                model.estimators_ = valid_estimators
                model.n_estimators = len(model.estimators_)

            elif self.local_update_method == "constant":
                if self.local_update_weighting == "equal":
                    weights: list[float] = [1.0] * len(model.estimators_)
                elif self.local_update_weighting == "site_size":
                    weights: list[float] = [
                        estimator.sample_weight for estimator in valid_estimators
                    ]
                model.estimators_ = np.random.default_rng(random_state).choice(
                    valid_estimators,
                    size=model.n_estimators,
                    p=np.array(weights) / sum(weights),
                )

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def from_file(cls, path) -> Self:
        return joblib.load(path)
