import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder


class CategoricalEncoder:

    def encode(self, train: pd.DataFrame, val: pd.DataFrame):
        str_cols = self._get_string_columns(train)

        # will create special values for unknown features in test set
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.NAN)
        encoder.fit(train[str_cols])

        train[str_cols] = encoder.transform(train[str_cols])
        val[str_cols] = encoder.transform(val[str_cols])

    @staticmethod
    def _get_string_columns(data: pd.DataFrame):
        return [col for col in data.columns if pd.api.types.is_string_dtype(data[col].dtype)]
