import os
import pickle

import pandas as pd


class CSVLoader:

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def load(self) -> pd.DataFrame:
        data: pd.DataFrame = pd.read_csv(self.path)
        return data


class TrainingDataPersister:
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def load(self):
        with open(self._path_to_file(), 'rb') as f:
            data = pickle.load(f)
        return data['X'], data['y']

    def write(self, data: pd.DataFrame, y: pd.Series):
        train_data = {
            'X': data,
            'y': y
        }

        with open(self._path_to_file(), 'wb') as f:
            pickle.dump(train_data, f)

    def _path_to_file(self):
        return os.path.join(self.path)
