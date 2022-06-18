import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


class ModelResolver:

    @staticmethod
    def of(name: str):
        return ModelHandler(name, ModelResolver._name(name), ModelResolver._param_grid(name))

    @staticmethod
    def _name(name: str):
        if name == 'LinearRegression':
            return LinearRegression()
        elif name == 'RandomForestRegressor':
            # best results from grid search
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

    def __init__(self, name: str, estimator, param_grid: dict):
        self.name = name
        self.estimator = estimator
        self.param_grid = param_grid

    def fit(self, data: pd.DataFrame, y: pd.Series):
        self.estimator.fit(data, y)

    def predict(self, data: pd.DataFrame):
        return self.estimator.predict(data)

    def grid_search(self, data: pd.DataFrame, y: pd.Series) -> None:
        print('Grid search started')
        grid = GridSearchCV(self.estimator, self.param_grid, cv=4, n_jobs=6, verbose=10)
        found = grid.fit(data, y)
        print(f'Best cv score={found.best_score_}')
        print(f'Best cv params={found.best_params_}')
        print(f'Best cv estimator={found.best_estimator_}')

    def get(self):
        return self.estimator

    def load(self):
        pass

    def write(self):
        pass
