import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer


# Pearson correlation
#   0.3 to 0.5 - weak correlation
#   0.5 to 0.7 - moderate correlation
#   0.7 to 0.9 - strong correlation
#   0.9 to 1 - very strong correlation
class CorrelationHandler:

    def __init__(self, mode: str, data: pd.DataFrame, y: pd.Series):
        self.mode = mode
        self.corr_data = data.assign(SalePrice=y.values)

    def drop_na_columns(self):
        self.corr_data.drop(columns=['Id', 'SalePrice'], axis=1)

    def handle(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        if self.mode == 'none':
            pass
        elif self.mode == 'no_corr':
            self._drop_below_weak_correlation(train, val)
        else:
            raise Exception(f'Not valid correlation mode={self.mode}')

    def _drop_below_weak_correlation(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        corr = self.corr_data.corr()
        below_weak_corr_cols = corr[abs(corr['SalePrice']) <= 0.3].index
        print(f'Dropping columns: {below_weak_corr_cols}')

        train.drop(columns=below_weak_corr_cols, axis=1, inplace=True)
        train.reset_index(drop=True, inplace=True)

        val.drop(columns=below_weak_corr_cols, axis=1, inplace=True)
        val.reset_index(drop=True, inplace=True)

    def _drop_below_moderate_correlation(self) -> None:
        pass

    def _drop_below_strong_correlation(self) -> None:
        pass


class MissingDataHandler:

    def __init__(self, mode: str, data: pd.DataFrame) -> None:
        self.imputer = SimpleImputer(strategy=mode, missing_values=np.NAN)
        self.imputer.fit(data)

    def fill(self, data: pd.DataFrame):
        data[:] = self.imputer.transform(data)

    @staticmethod
    def drop_high_na_columns(data: pd.DataFrame) -> None:
        # based on train EDA
        data.drop(columns=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
        data.reset_index(drop=True, inplace=True)


class OutliersHandler:
    @staticmethod
    def remove_outliers(data: pd.DataFrame):
        pass


# TODO should some columns be just boolean?
# TODO should some columns be replaced with bins?
# TODO scale features?
class ColumnTransformer:
    pass


# TODO explore options to change distribution of variables within column
class DistributionTransformer:
    pass
