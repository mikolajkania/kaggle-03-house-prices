import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


class ModelResolver:

    @staticmethod
    def of(name: str):
        return ModelHandler(ModelResolver._basic(name), ModelResolver._final(name), ModelResolver._param_grid(name))

    @staticmethod
    def _basic(name: str):
        if name == 'LinearRegression':
            return LinearRegression()
        elif name == 'RandomForestRegressor':
            return RandomForestRegressor()
        else:
            raise Exception(f'Unsupported model name={name}')

    @staticmethod
    def _final(name: str):
        # best results from grid search

        if name == 'LinearRegression':
            return LinearRegression()
        elif name == 'RandomForestRegressor':
            return RandomForestRegressor(bootstrap=False, max_depth=19, max_features='sqrt', n_estimators=1800)
        else:
            raise Exception(f'Unsupported model name={name}')

    @staticmethod
    def _param_grid(name: str) -> dict:
        if name == 'RandomForestRegressor':
            return {
                'n_estimators': list(range(100, 2000, 100)),
                'max_depth': list(range(10, 20, 1)),
                'min_samples_split': list(range(2, 10, 2)),
                'min_samples_leaf': list(range(1, 5, 1)),
                'max_features': ['sqrt', 'log2', 1.0],
                'bootstrap': [True, False]
            }


class ModelHandler:

    def __init__(self, basic_estimator, final_estimator, param_grid: dict):
        self.basic_estimator = basic_estimator
        self.final_estimator = final_estimator
        self.param_grid = param_grid

    def fit(self, data: pd.DataFrame, y: pd.Series):
        self.final_estimator.fit(data, y)

    def predict(self, data: pd.DataFrame):
        return self.final_estimator.predict(data)

    def grid_search(self, data: pd.DataFrame, y: pd.Series) -> None:
        print('Grid search started')
        grid = GridSearchCV(self.basic_estimator, self.param_grid, cv=4, n_jobs=6, verbose=10)
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
