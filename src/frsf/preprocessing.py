# %%
import pandas as pd
from typing import Optional
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from typing import Self
import numpy as np


class SchemaAligner(BaseEstimator, TransformerMixin):

    def fit(self, full_schema: list[str], column_map: Optional[dict[str, str]] = None):
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
        column_map: Optional[dict[str, str]] = None,
    ):
        self.fit(full_schema, column_map)
        return self.transform(Data)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> Self:
        return joblib.load(path)


class SchemaCreator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        anonymize: bool = False,
        extra_columns: int = 0,
        extra_column_prefix: str = "extra_",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.anonymize = anonymize
        self.extra_columns = extra_columns
        self.extra_column_prefix = extra_column_prefix
        self.random_state = random_state
        self.extra_columns_used = 0

    def fit_transform(
        self,
        local_features: list[list[str]],
        local_column_maps: Optional[list[dict]] = None,
    ):
        local_features = [set(features) for features in local_features]
        if local_column_maps is None:
            local_column_maps = [dict() for _ in range(len(local_features))]

        local_column_maps_closure = [
            {feature: feature for feature in features} for features in local_features
        ]

        local_column_maps_full: list[dict[str, str]] = []
        for column_map, column_map_closure in zip(
            local_column_maps, local_column_maps_closure
        ):
            local_column_maps_full.append(column_map_closure | column_map)

        schema = set()
        for local_map in local_column_maps_full:
            schema.update(local_map.values())
        schema = sorted(schema)

        extra_columns = [
            f"{self.extra_column_prefix}{i}" for i in range(self.extra_columns)
        ]
        schema.extend(extra_columns)

        if self.anonymize:
            new_schema = [f"feature_{i}" for i in range(len(schema))]
            schema = (
                np.random.RandomState(self.random_state).permutation(schema).tolist()
            )
        else:
            new_schema = schema

        self.schema = new_schema
        self.schema_column_map = dict(zip(schema, new_schema))

        for local_map in local_column_maps_full:
            for key, value in local_map.items():

                local_map[key] = self.schema_column_map[value]

        return self.schema, local_column_maps_full

    def add_client(self, features: list[str], column_map: Optional[dict] = None):
        if column_map is None:
            column_map = dict()

        column_map_closure = {feature: feature for feature in features}

        local_column_map_full: dict[str, str] = column_map_closure | column_map

        needed_columns = set(local_column_map_full.values()) - set(
            self.schema_column_map.keys()
        )
        if len(needed_columns) > self.extra_columns - self.extra_columns_used:
            raise ValueError(
                "Not enough extra columns to accommodate new client features."
            )

        keys_to_remove = []

        for key, value in local_column_map_full.items():
            if value in self.schema_column_map.keys():
                continue
            extra_column_name = f"{self.extra_column_prefix}{self.extra_columns_used}"
            keys_to_remove.append(extra_column_name)
            self.schema_column_map[value] = self.schema_column_map[extra_column_name]
            self.extra_columns_used += 1
            local_column_map_full[key] = extra_column_name

        for key, value in local_column_map_full.items():
            local_column_map_full[key] = self.schema_column_map[value]

        for key in keys_to_remove:
            del self.schema_column_map[key]

        return self.schema, local_column_map_full

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path) -> Self:
        return joblib.load(path)
