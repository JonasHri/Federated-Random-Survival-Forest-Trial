from sksurv.ensemble import RandomSurvivalForest
import pandas as pd
import joblib
from typing import Self


class LocalRandomSurvivalForest(RandomSurvivalForest):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        sample_weight=None,
    ) -> Self:
        self.site_size: int = len(X)
        self.local_features = set(X.columns[~X.isna().all()].tolist())
        super().fit(X, y, sample_weight=sample_weight)
        for estimator in self.estimators_:
            estimator.sample_weight = self.site_size / self.n_estimators

        return self

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def from_file(cls, path) -> Self:
        return joblib.load(path)
