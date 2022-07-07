import pandas as pd


class FeatureCreator:

    @staticmethod
    def create_new(data: pd.DataFrame):
        FeatureCreator._extend_porch(data)

    @staticmethod
    def _create_binary(data: pd.DataFrame, feature: str):
        data[feature + 'binary'] = data[feature].apply(lambda x: 1 if x > 0 else 0)
        data[feature + 'binary'].astype(bool)

    @staticmethod
    def _extend_porch(data: pd.DataFrame):
        data['PorchArea_artificial'] = (data['OpenPorchSF'] + data['EnclosedPorch'] +
                                        data['3SsnPorch'] + data['ScreenPorch'])
