# %%
import pandas as pd
from typing import Optional, Dict
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from typing import Self


class SchemaAligner(BaseEstimator, TransformerMixin):

    def fit(self, full_schema: list[str], column_map: Optional[Dict] = None):
        self.full_schema = full_schema
        self.column_map = column_map
        return self

    def transform(self, Data: pd.DataFrame):
        if self.column_map is None:
            local_columns = Data.columns.tolist()
            column_map = {column: column for column in local_columns}
        else:
            column_map = self.column_map

        Data_renamed = Data.rename(columns=column_map)
        Data_reindexed = Data_renamed.reindex(columns=self.full_schema)
        # Data_reindexed.attrs["aligned"] = True
        return Data_reindexed

    def fit_transform(
        self,
        Data: pd.DataFrame,
        full_schema: list[str],
        column_map: Optional[Dict] = None,
    ):
        self.fit(full_schema, column_map)
        return self.transform(Data)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> Self:
        return joblib.load(path)


class SchemaCreator(BaseEstimator, TransformerMixin):

    def fit_transform(self, local_features: list[list[str]]):
        local_features = [set(features) for features in local_features]
        schema = list(set.union(*local_features))
        schema = sorted(schema)
        return schema
