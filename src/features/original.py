import numpy as np
import pandas as pd

from scipy.stats import boxcox
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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

    def transform(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
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

    def __init__(self):
        self.outliers = dict()

    def fit(self, data: pd.DataFrame):
        for col in data.columns:
            col_std = data[col].std()
            col_mean = data[col].mean()
            cut_off = col_std * 3

            self.outliers[col] = {}
            self.outliers[col]['lower'], self.outliers[col]['upper'] = col_mean - cut_off, col_mean + cut_off

    def transform(self, data: pd.DataFrame):
        for col in data.columns:
            lower = self.outliers[col]['lower']
            upper = self.outliers[col]['upper']

            skew_before = data[col].skew()
            data[col] = np.where(data[col] < lower, lower, data[col])
            data[col] = np.where(data[col] > upper, upper, data[col])
            skew_after = data[col].skew()

            print(f'[Outliers removal] For {col}: skew before={skew_before}, and after={skew_after}')


# TODO should some columns be just boolean?
# TODO should some columns be replaced with bins?
# TODO should some columns be joined?
class FeatureTransformer:
    pass


class FeatureScaler:

    def __init__(self):
        self.min_max_scaler = StandardScaler()

    def fit(self, data: pd.DataFrame):
        self.min_max_scaler.fit(data)

    def transform(self, train: pd.DataFrame, val: pd.DataFrame):
        train[:] = self.min_max_scaler.transform(train)
        val[:] = self.min_max_scaler.transform(val)


class DistributionTransformer:
    @staticmethod
    def transform(data: pd.Series, lmbda: float):
        skew_before = data.skew()
        transformed_data = boxcox(x=data, lmbda=lmbda)
        skew_after = pd.Series(transformed_data).skew()
        print(f'[Target distribution] For {data.name}: skew before={skew_before}, and after={skew_after}')
        return transformed_data
