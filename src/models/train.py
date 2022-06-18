import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


class ModelResolver:

    @staticmethod
    def of(name: str):
        return ModelHandler(ModelResolver._final(name),
                            ModelResolver._grid_estimator(name),
                            ModelResolver._grid_param(name))

    @staticmethod
    def _final(name: str):
        # best results from grid search

        if name == 'LinearRegression':
            return LinearRegression()
        elif name == 'RandomForestRegressor':
            return RandomForestRegressor(bootstrap=False, max_depth=19, max_features='sqrt',
                                         n_estimators=1800, random_state=42)
        elif name == 'XGBRegressor':
            return xgb.XGBRegressor(random_state=42, seed=42, booster='dart', max_depth=None, n_estimators=30)
        else:
            raise Exception(f'Unsupported model name={name}')

    @staticmethod
    def _grid_estimator(name: str):
        if name == 'LinearRegression':
            return LinearRegression()
        elif name == 'RandomForestRegressor':
            return RandomForestRegressor(random_state=42)
        elif name == 'XGBRegressor':
            return xgb.XGBRegressor(random_state=42, seed=42)
        else:
            raise Exception(f'Unsupported model name={name}')

    @staticmethod
    def _grid_param(name: str) -> dict:
        if name == 'LinearRegression':
            return {}
        elif name == 'RandomForestRegressor':
            return {
                'n_estimators': list(range(100, 2000, 100)),
                'max_depth': list(range(10, 20, 1)),
                'min_samples_split': list(range(2, 10, 2)),
                'min_samples_leaf': list(range(1, 5, 1)),
                'max_features': ['sqrt', 'log2', 1.0],
                'bootstrap': [True, False],
                'random_state': [42]

            }
        elif name == 'XGBRegressor':
            return {
                'n_estimators': [90, 80, 70, 60, 50, 40, 30, 20, 10],
                'max_depth': [None],
                # 'n_estimators': list(range(100, 2000, 100)),
                # 'max_depth': [None, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                'booster': ['gbtree', 'gblinear', 'dart'],
                'random_state': [42],
                'seed': [42]

            }
        else:
            raise Exception(f'Unsupported model name={name}')


class ModelHandler:

    def __init__(self, final_estimator, grid_estimator, grid_param: dict):
        self.final_estimator = final_estimator
        self.grid_estimator = grid_estimator
        self.grid_param = grid_param

    def fit(self, data: pd.DataFrame, y: pd.Series):
        self.final_estimator.fit(data, y)

    def predict(self, data: pd.DataFrame):
        return self.final_estimator.predict(data)

    def grid_search(self, data: pd.DataFrame, y: pd.Series) -> None:
        print('Grid search started')
        grid = GridSearchCV(self.grid_estimator, self.grid_param, cv=4, n_jobs=6, verbose=10)
        found = grid.fit(data, y)
        print(f'Best cv score={found.best_score_}')
        print(f'Best cv params={found.best_params_}')
        print(f'Best cv estimator={found.best_estimator_}')

    def get(self):
        return self.final_estimator

    def load(self):
        pass

    def write(self):
        pass
