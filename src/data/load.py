import pandas as pd


class DataLoader:

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path

    def load(self) -> pd.DataFrame:
        train_df: pd.DataFrame = pd.read_csv(self.path)
        return train_df
