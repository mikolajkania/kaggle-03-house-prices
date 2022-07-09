import pandas as pd


class FeatureCreator:

    @staticmethod
    def create_new(data: pd.DataFrame):
        data['PorchArea_artificial'] = (data['OpenPorchSF'] + data['EnclosedPorch'] +
                                        data['3SsnPorch'] + data['ScreenPorch'])

        data['TotalUtilityArea_artificial'] = (data['GrLivArea'] + data['TotalBsmtSF'] +
                                                data['GarageArea'])

        data['HomeQual_artificial'] = (data['OverallQual'] + data['OverallCond'])

        data['OtherRoomsAbvGr_artificial'] = data['TotRmsAbvGrd'] - data['KitchenAbvGr'] - data['BedroomAbvGr']
