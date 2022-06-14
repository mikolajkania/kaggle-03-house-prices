import pandas as pd
from sklearn.linear_model import LinearRegression


class ModelResolver:
    @staticmethod
    def get(name: str):
        if name == 'LinearRegression':
            return LinearRegression()


class ModelHandler:

    def __init__(self, name: str) -> None:
        super().__init__()
        self._model = ModelResolver.get(name)

    def fit(self, data: pd.DataFrame, y: pd.Series):
        self._model.fit(data, y)

    def predict(self, data: pd.DataFrame):
        return self._model.predict(data)

    def get(self):
        return self._model

    def load(self):
        pass

    def write(self):
        pass
