import pandas as pd


class StringEncoder:

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def encode(data: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(data).reset_index(drop=True)

    # TODO not used now
    @staticmethod
    def _get_string_columns(data: pd.DataFrame):
        return [col for col in data.columns if pd.api.types.is_string_dtype(data[col].dtype)]
