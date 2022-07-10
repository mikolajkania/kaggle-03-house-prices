import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


class ModelResolver:

    @staticmethod
    def of(name: dict):
        return ModelHandler(ModelResolver._final(name),
                            ModelResolver._grid_base_estimator(name),
                            ModelResolver._grid_param(name))

    @staticmethod
    def _final(name: str):
        if name == 'XGBRegressor':
            return xgb.XGBRegressor(booster='gbtree', max_depth=5,
                                    n_estimators=450, learning_rate=0.1,
                                    random_state=42, seed=42)
        elif name == 'LGBMRegressor':
            return LGBMRegressor(boosting_type='gbdt', learning_rate=0.006, max_bin=500, max_depth=8,
                                    n_estimators=6000, num_leaves=10, random_state=42)
        elif name == 'LinearRegression':
            return LinearRegression()
        elif name == 'RandomForestRegressor':
            return RandomForestRegressor(bootstrap=False, max_depth=19, max_features='sqrt',
                                    n_estimators=1800, random_state=42)
        elif name == 'Ridge':
            return Ridge(alpha=3, solver='sag', tol=0.01, random_state=42)
        else:
            raise Exception(f'Unsupported model name={name}')

    @staticmethod
    def _grid_base_estimator(name: str):
        if name == 'LinearRegression':
            return LinearRegression()
        elif name == 'RandomForestRegressor':
            return RandomForestRegressor(random_state=42)
        elif name == 'XGBRegressor':
            return xgb.XGBRegressor(random_state=42, seed=42)
        elif name == 'LGBMRegressor':
            return LGBMRegressor(random_state=42)
        elif name == 'Ridge':
            return Ridge(random_state=42)
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
                # 'n_estimators': [150, 140, 130, 120, 110, 100, 70, 50, 30],
                # 'max_depth': [None],
                'n_estimators': list(range(100, 6200, 500)),
                'max_depth': [None, 5],
                'booster': ['gbtree'],
                'random_state': [42],
                'seed': [42],
                'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.3]
            }
        elif name == 'LGBMRegressor':
            return {
                'boosting_type': ['gbdt'],
                'num_leaves': [9, 10, 11],
                'max_depth': [7, 8, 9],
                'learning_rate': [0.006, 0.01],
                'n_estimators': [5000, 6000],
                'max_bin': [400, 500, 600],
                'random_state': [42],
                # from LightGBM tuning docs
                # 'bagging_freq': [0, 5],
                # 'bagging_fraction': [1.0, 0.75],
                # 'bagging_seed': [42]
            }
        elif name == 'Ridge':
            return {
                # 'alpha': [0.001, 0.01, 0.1, 1.0, 10],
                # 'tol': [0.1, 0.01, 0.001, 0.001],
                'alpha': [1.0, 1.5, 3],
                'tol': [0.006, 0.01, 0.03],
                'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'lsqr', 'sag', 'lbfgs'],
                'random_state': [42]
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
        grid = GridSearchCV(self.grid_estimator, self.grid_param, cv=4, n_jobs=4, verbose=10)
        found = grid.fit(data, y)
        print(f'Best cv score={found.best_score_}')
        print(f'Best cv params={found.best_params_}')
        print(f'Best cv estimator={found.best_estimator_}')

        print('\n\n')
        print('Best parameters set found on development set:')
        print(grid.best_params_)
        print()
        print('Grid scores on development set:')
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
        print()

    def get(self):
        return self.final_estimator

    def load(self):
        pass

    def write(self):
        pass
