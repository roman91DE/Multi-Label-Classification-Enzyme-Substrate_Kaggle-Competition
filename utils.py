import pandas as pd
from dataclasses import dataclass
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


@dataclass
class ClassificationScore:
    TP: int = 0 # True Positive
    FP: int = 0 # False Positive
    TN: int = 0 # True Negative
    FN: int = 0 # False Negative

    def num_cases(self):
        return self.TP + self.FP + self.TN + self.FN
    
    def accuracy(self):
        """how many cases were correctly classified"""
        return (self.TP + self.TN) / self.num_cases()
    
    def precision(self):
        """how many positive predictions were correct"""
        return self.TP / (self.TP + self.FP)

    def specificity(self):
        """how many negative cases were correctly classified"""
        return self.TN / (self.TN + self.FP)
    
    def recall(self):
        """how many positive cases were correctly classified"""
        return self.TP / (self.TP + self.FN)
    
    def f1(self):
        """harmonic mean of precision and recall"""
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def roc_auc(self):
        """Receiver Operating Characteristic Area Under the Curve"""
        return (self.TP / (self.TP + self.FN)) - (self.FP / (self.FP + self.TN))
    
def get_ClassificationScore(true_vals, pred_vals) -> ClassificationScore:
    cs = ClassificationScore()
    for target, prediction in zip(true_vals, pred_vals):
        if target == 1 and prediction == 1:
            cs.TP += 1
        elif target == 0 and prediction == 0:
            cs.TN += 1
        elif target == 0 and prediction == 1:
            cs.FP += 1
        elif target == 1 and prediction == 0:
            cs.FN += 1
    return cs