import pandas as pd
from typing import List, Optional, Dict
from typing import Optional


def align_schema(
    Data: pd.DataFrame, full_schema: List[str], column_map: Optional[Dict] = None
):
    if column_map is None:
        local_columns = Data.columns.tolist()
        column_map = {column: column for column in local_columns}

    Data_renamed = Data.rename(columns=column_map)

    return Data_renamed.reindex(columns=full_schema)
