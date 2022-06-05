import json

import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error


class ModelEvaluator:

    @staticmethod
    def metrics(model: RegressorMixin, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series) -> dict:

        y_pred = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        r2_train = model.score(X_train, y_train)
        print(f'R2 score on training set: {r2_train}')
        r2_val = model.score(X_val, y_val)
        print(f'R2 score on validation set: {r2_val}')
        rmse_train = mean_squared_error(y_train, y_pred, squared=False)
        print(f'RMSE predictions: {rmse_train}')
        rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
        print(f'RMSE validation: {rmse_val}')

        return {
            'r2_train': r2_train,
            'r2_val': r2_val,
            'rmse_train': rmse_train,
            'rmse_test': rmse_val
        }

    @staticmethod
    def save_metrics(path: str, metrics: dict):
        with open(path, 'w') as f:
            f.write(json.dumps(metrics))

    def kaggle_submission(self):
        pass
