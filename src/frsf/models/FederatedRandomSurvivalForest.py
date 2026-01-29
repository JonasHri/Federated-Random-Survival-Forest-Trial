from sksurv.ensemble import RandomSurvivalForest
from .LocalRandomSurvivalForest import LocalRandomSurvivalForest
from typing import List
from random import sample


class FederatedRandomSurvivalForest(RandomSurvivalForest):
    def __init__(self, local_models: List[LocalRandomSurvivalForest], **kwargs):
        super().__init__(**kwargs)
        self.local_models = local_models
        self.features = local_models[0].all_features
        self.feature_names_in_ = local_models[0].feature_names_in_
        self.estimators_ = []
        for model in local_models:
            self.estimators_.extend(model.estimators_)
        self.n_estimators = len(self.estimators_)

        self.estimator_features = []
        for estimator in self.estimators_:
            tree_features = estimator.tree_.feature
            tree_features_names = set(
                [self.features[i] for i in tree_features if i >= 0]
            )
            self.estimator_features.append(tree_features_names)

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Federated model cannot be fit directly.")

    def update_local_models(self, local_size: int = None):
        for model in self.local_models:
            valid_estimators = []
            for estimator, feat_set in zip(self.estimators_, self.estimator_features):
                # find all estimators that only use features available locally
                if feat_set.issubset(model.local_features):
                    valid_estimators.append(estimator)

            if local_size is None:
                model.estimators_ = valid_estimators
            else:
                model.estimators_ = sample(valid_estimators, local_size)
            model.n_estimators = len(model.estimators_)
