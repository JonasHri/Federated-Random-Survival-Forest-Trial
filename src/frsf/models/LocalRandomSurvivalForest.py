from sksurv.ensemble import RandomSurvivalForest
import pandas as pd


class LocalRandomSurvivalForest(RandomSurvivalForest):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, sample_weight=None):
        self.all_features = X.columns.tolist()
        self.local_features = set(X.columns[~X.isna().all()].tolist())
        return super().fit(X, y, sample_weight=sample_weight)
