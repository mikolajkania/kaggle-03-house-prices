import pandas as pd


class FeatureCreator:

    @staticmethod
    def create_new(data: pd.DataFrame):
        FeatureCreator._extend_porch(data)

        data['TotalUtilityArea_artificial'] = (data['GrLivArea'] + data['TotalBsmtSF'] + data['GarageArea'])

    @staticmethod
    def _extend_porch(data: pd.DataFrame):
        data['PorchArea_artificial'] = (data['OpenPorchSF'] + data['EnclosedPorch'] +
                                        data['3SsnPorch'] + data['ScreenPorch'])
