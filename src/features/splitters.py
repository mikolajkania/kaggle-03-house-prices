import pandas as pd

from sklearn.model_selection import train_test_split


class DataSplitter:

    @staticmethod
    def split(data: pd.DataFrame, y: pd.Series, train_ratio: float):
        return train_test_split(data, y, train_size=train_ratio, shuffle=True, random_state=42)
