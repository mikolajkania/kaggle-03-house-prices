import os
import pickle

import pandas as pd


class CSVLoader:

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def load(self) -> tuple[pd.DataFrame, pd.Series]:
        data_df: pd.DataFrame = pd.read_csv(self.path)
        return data_df.drop(columns=['Id', 'SalePrice'], axis=1), data_df['SalePrice']


class TrainingDataHandler:
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def load(self):
        with open(self._path_to_file(), 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y']

    def write(self, X: pd.DataFrame, y: pd.Series):
        train_data = {
            'X': X,
            'y': y
        }

        with open(self._path_to_file(), 'wb') as f:
            pickle.dump(train_data, f)

    def _path_to_file(self):
        return os.path.join(self.path)
