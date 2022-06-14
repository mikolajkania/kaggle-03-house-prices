import json

import pandas as pd

from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_log_error


class ModelEvaluator:

    @staticmethod
    def metrics(model: RegressorMixin, data: pd.DataFrame, y: pd.Series):
        y_pred = model.predict(data)

        r2 = model.score(data, y)
        rmsle = mean_squared_log_error(y, y_pred, squared=False)

        return r2, rmsle

    @staticmethod
    def save_metrics(path: str, metrics: dict):
        print(metrics)
        with open(path, 'w') as f:
            f.write(json.dumps(metrics))

    def kaggle_submission(self):
        pass
