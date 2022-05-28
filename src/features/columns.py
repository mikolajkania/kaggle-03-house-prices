import pandas as pd

from sklearn.impute import SimpleImputer


# Pearson correlation
#   0.3 to 0.5 - weak correlation
#   0.5 to 0.7 - moderate correlation
#   0.7 to 0.9 - strong correlation
#   0.9 to 1 - very strong correlation
class CorrelationHandler:

    @staticmethod
    def drop_no_correlation(self, data: pd.DataFrame) -> None:
        pass

    @staticmethod
    def drop_weak_correlation(self, data: pd.DataFrame) -> None:
        pass

    @staticmethod
    def drop_moderate_correlation(self, data: pd.DataFrame) -> None:
        pass

    @staticmethod
    def drop_strong_correlation(self, data: pd.DataFrame) -> None:
        pass

    @staticmethod
    def drop_very_strong_correlation(self, data: pd.DataFrame) -> None:
        pass


class MissingDataHandler:
    def fill(self, mode: str, data: pd.DataFrame):
        if mode == 'freq':
            self._replace_most_freq(data)
        else:
            raise Exception('Not supported mode!')

    @staticmethod
    def _replace_most_freq(data: pd.DataFrame):
        data[:] = SimpleImputer(strategy='most_frequent').fit_transform(data)

    @staticmethod
    def _replace_median(data: pd.DataFrame):
        pass

    @staticmethod
    def drop_na_columns(data: pd.DataFrame) -> None:
        data.drop(columns=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
        data.reset_index(drop=True, inplace=True)


class OutliersHandler:
    @staticmethod
    def remove_outliers(data: pd.DataFrame):
        pass


# TODO should some columns be just boolean?
# TODO should some columns be replaced with bins?
class ColumnTransformer:
    pass


# TODO explore options to change distribution of variables within column
class DistributionTransformer:
    pass
