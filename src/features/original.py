import numpy as np
import pandas as pd

from scipy.special import boxcox1p
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


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

    def transform(self, data: pd.DataFrame) -> None:
        if self.mode == 'none':
            pass
        elif self.mode == 'no_corr':
            self._drop_below_weak_correlation(data)
        else:
            raise Exception(f'Not valid correlation mode={self.mode}')

    def _drop_below_weak_correlation(self, data: pd.DataFrame) -> None:
        corr = self.corr_data.corr()
        below_weak_corr_cols = corr[abs(corr['SalePrice']) <= 0.3].index
        print(f'Dropping columns: {below_weak_corr_cols}')

        data.drop(columns=below_weak_corr_cols, axis=1, inplace=True)
        data.reset_index(drop=True, inplace=True)

    def _drop_below_moderate_correlation(self) -> None:
        pass

    def _drop_below_strong_correlation(self) -> None:
        pass


class MissingDataHandler:

    def __init__(self, mode: str, data: pd.DataFrame) -> None:
        self.imputer = SimpleImputer(strategy=mode, missing_values=np.NAN)
        self.imputer.fit(data)

    def transform(self, data: pd.DataFrame):
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

    def transform(self, data: pd.DataFrame, verbose=False):
        for col in data.columns:
            lower = self.outliers[col]['lower']
            upper = self.outliers[col]['upper']

            skew_before = data[col].skew()
            data[col] = np.where(data[col] < lower, lower, data[col])
            data[col] = np.where(data[col] > upper, upper, data[col])
            skew_after = data[col].skew()

            if verbose:
                print(f'[Outliers removal] For {col}: skew before={skew_before}, and after={skew_after}')


class BinsTransformer:
    def __init__(self, col: str, bins_cnt: int = 8):
        self.col = col
        self.bins_cnt = bins_cnt
        self.bin_edges = list()

    def fit(self, data: pd.DataFrame):
        data[self.col] = data[self.col].astype(int)
        bins = pd.qcut(data[self.col], q=self.bins_cnt, labels=False, precision=0, retbins=True)

        self.bin_edges = list(zip(bins[1], bins[1][1:]))

    def transform(self, data: pd.DataFrame):
        for idx, edge in enumerate(self.bin_edges):
            if idx == 0:
                data.loc[data[self.col] < edge[1], self.col] = idx
            if idx == len(self.bin_edges) - 1:
                data.loc[data[self.col] >= edge[0], self.col] = idx
            else:
                data.loc[(data[self.col] >= edge[0]) & (data[self.col] < edge[1]), self.col] = idx


class TypeTransformer:

    @staticmethod
    def transform(data: pd.DataFrame):
        data['MSSubClass'] = data['MSSubClass'].astype(str)
        data['OverallQual'] = data['OverallQual'].astype(str)
        data['OverallCond'] = data['OverallCond'].astype(str)
        data['YrSold'] = data['YrSold'].astype(str)
        data['MoSold'] = data['MoSold'].astype(str)


class FeatureScaler:

    def __init__(self):
        # using MinMaxScaler instead of StandardScaler to work with DistributionTransformer
        # it needs positive values
        self.scalar = MinMaxScaler()

    def fit(self, data: pd.DataFrame):
        self.scalar.fit(data)

    def transform(self, data: pd.DataFrame):
        data[:] = self.scalar.transform(data)


class DistributionTransformer:
    @staticmethod
    def transform(data: pd.Series, lmbda: float, verbose=False):
        skew_before = data.skew()
        # using this special value as ordinary boxcox required positive data only (not even 0s)
        transformed_data = boxcox1p(data, lmbda)
        skew_after = pd.Series(transformed_data).skew()
        if verbose:
            print(f'[Target distribution] For {data.name}: skew before={skew_before}, and after={skew_after}')
        return transformed_data

    @staticmethod
    def transform_df(data: pd.DataFrame, lmbda: float, verbose=False):
        for col in data.columns:
            data[col] = DistributionTransformer.transform(data[col], lmbda, verbose)
