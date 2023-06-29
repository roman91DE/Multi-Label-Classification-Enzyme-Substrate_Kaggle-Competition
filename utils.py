import pandas as pd
from functools import partial

PATH_TRAIN = "./data/train.csv"
PATH_TEST = "./data/test.csv"

DTYPES_FEATURES = {
          "id": "uint64",
          "fr_COO": "category",
          "fr_COO2": "category",
      }

DTYPES_TARGETS = {
          "EC1": "bool",
          "EC2": "bool",
          "EC3": "bool",
          "EC4": "bool",
          "EC5": "bool",
          "EC6": "bool"
}

DROP_COLS = ["EC3", "EC4", "EC5", "EC6"]





def _load_data(datapath: str, dtypes: dict, drop_cols: list) -> pd.DataFrame:
  return pd.read_csv(
      filepath_or_buffer=datapath,
      dtype=dtypes,
      index_col="id"
    ).drop(columns=drop_cols, axis=1)


GetTrainDF = partial(_load_data, datapath=PATH_TRAIN, dtypes=dict(**DTYPES_TARGETS, **DTYPES_FEATURES), drop_cols=DROP_COLS)
GetTestDF = partial(_load_data, datapath=PATH_TEST, dtypes=DTYPES_FEATURES, drop_cols=[])