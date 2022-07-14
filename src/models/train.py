import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score


class ModelResolver:

    @staticmethod
    def of(name: str):
        return ModelHandler(ModelResolver._final(name),
                            ModelResolver._grid_base_estimator(name),
                            ModelResolver._grid_param(name))

    @staticmethod
    def _final(name: str):
        if name == 'XGBRegressor':
            return xgb.XGBRegressor(learning_rate=0.01, n_estimators=5000, max_depth=3, reg_alpha=0.001,
                                    booster='gbtree', seed=42, random_state=42)
        elif name == 'LGBMRegressor':
            return LGBMRegressor(boosting_type='gbdt', learning_rate=0.01, max_bin=200, max_depth=-1,
                                 reg_alpha=0.006, n_estimators=6000, num_leaves=5,
                                 bagging_fraction=0.75, bagging_freq=10, bagging_seed=42,
                                 random_state=42)
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
                # 'n_estimators': list(range(1000, 5100, 500)),
                # 'max_depth': [None, 3, 5],
                # 'booster': ['gbtree'],
                # 'learning_rate': [0.001, 0.01],
                # 'reg_alpha': [0.00001, 0.0001, 0.001],
                # 'random_state': [42],
                # 'seed': [42],
                'n_estimators': list(range(3000, 5100, 500)),
                'max_depth': [2, 3, 4],
                'booster': ['gbtree'],
                'learning_rate': [0.003, 0.006, 0.01, 0.03],
                'reg_alpha': [0.0006, 0.001, 0.003],
                'random_state': [42],
                'seed': [42]
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

    def __init__(self, final_estimator, base_grid_estimator, grid_param: dict):
        self.final_estimator = final_estimator
        self.base_grid_estimator = base_grid_estimator
        self.grid_param = grid_param

    def fit(self, data: pd.DataFrame, y: pd.Series):
        self.final_estimator.fit(data, y)

    def predict(self, data: pd.DataFrame):
        return self.final_estimator.predict(data)

    def grid_search(self, data: pd.DataFrame, y: pd.Series) -> None:
        print('Grid search started')
        # GridSearch unified scoring API always maximizes the score, so scores which need to
        # be minimized are negated in order for the unified scoring API to work correctly
        # In other words - the higher value, the better (but all of them will be negative)
        grid = GridSearchCV(self.base_grid_estimator, self.grid_param, cv=2, n_jobs=4,
                            verbose=10, scoring='neg_mean_squared_error')
        found = grid.fit(data, y)

        print(f'Best cv score={found.best_score_}')
        print(f'Best cv params={found.best_params_}')
        print(f'Best cv estimator={found.best_estimator_}')
        print()
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
        print()

        # Results for best parameters
        results = cross_val_score(found.best_estimator_, data, y, cv=5, scoring='r2')
        print(f'With best grid search model, in CV, mean R2 score={results.mean()}')
        print(f'With best grid search model R2 score={found.best_estimator_.score(data, y)}')

        mse = -(cross_val_score(found.best_estimator_, data, y, cv=5, scoring='neg_mean_squared_error').mean())
        print(f'With best grid search model CV MSE={mse}')
        print(f'With best grid search model CV RMSE={np.sqrt(mse)}')

    def get(self):
        return self.final_estimator
