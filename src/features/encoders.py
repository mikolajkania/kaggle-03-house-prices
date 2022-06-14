import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder


class CategoricalEncoder:

    def __init__(self):
        self.str_cols = dict()
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.NAN)

    def fit(self, data: pd.DataFrame):
        self.str_cols = self._get_string_columns(data)

        # will create special values for unknown features in test set
        self.encoder.fit(data[self.str_cols])

    def transform(self, data: pd.DataFrame):
        data[self.str_cols] = self.encoder.transform(data[self.str_cols])

    @staticmethod
    def _get_string_columns(data: pd.DataFrame):
        return [col for col in data.columns if pd.api.types.is_string_dtype(data[col].dtype)]
