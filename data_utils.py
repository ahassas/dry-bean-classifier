from torch.utils.data import TensorDataset, DataLoader
import torch

from typing import List, Tuple
import pandas as pd
import numpy as np

class Dataset:

    def __init__(self, path, preprocessor, batch_size, shuffle):
        self.path = path
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load_data(self):
        data = pd.read_csv(self.path)
        return data

    def prepare_data(self):
        data = self.load_data()
        X, y = self.preprocessor.apply_preprocess(data)

        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(y))
        dataset = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return dataset


# Fixed class order -> the same one-hot index for every run (train, valid, test)
CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]

# Size-related columns are right-skewed (Area goes from ~20k to ~250k),
# so we compress them with log before the network sees them.
SKEWED = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
          "ConvexArea", "EquivDiameter"]


class Preprocessor:

    def __init__(self, feature_columns: List, label_column: str, **kwargs):
        self.feature_columns = feature_columns
        self.label_column = label_column
        # log on size features: tested in experiments, gave no gain over plain
        # standardization, so it is off by default
        self.log_size = False

        self.__dict__.update(kwargs)

    def apply_preprocess(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = self.select_features(data)
        X = self.preprocess_features(X)

        y = self.prepare_labels(data)

        return X, y

    def select_features(self, data: pd.DataFrame) -> np.ndarray:
        return data[self.feature_columns].values

    def preprocess_features(self, data: np.ndarray) -> np.ndarray:
        # Only stateless transforms here. A fresh Preprocessor is used at inference,
        # so anything learned from train (mean/std) would be lost.
        # Standardization is done inside the model instead (see model.py).
        X = data.astype(np.float32)
        if self.log_size:
            for i, col in enumerate(self.feature_columns):
                if col in SKEWED:
                    X[:, i] = np.log(X[:, i])
        return X

    def prepare_labels(self, data: pd.DataFrame) -> np.ndarray:
        idx = data[self.label_column].map(CLASSES.index).values
        return np.eye(len(CLASSES), dtype=np.float32)[idx]   # one-hot
