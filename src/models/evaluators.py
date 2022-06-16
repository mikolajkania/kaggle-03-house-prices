import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score

from src.models.train import ModelHandler


class ModelEvaluator:

    @staticmethod
    def metrics(model: ModelHandler, data: pd.DataFrame, y: pd.Series):
        y_pred = model.get().predict(data)

        r2 = model.get().score(data, y)
        rmsle = mean_squared_log_error(y, y_pred, squared=False)

        return r2, rmsle

    @staticmethod
    def save_metrics(path: str, metrics: dict):
        print(metrics)
        with open(path, 'w') as f:
            f.write(json.dumps(metrics))

    @staticmethod
    def cv_metrics(model: ModelHandler, data: pd.DataFrame, y: pd.Series):
        cv_scores = cross_val_score(model.get(), data, y, scoring='neg_mean_squared_log_error', cv=5)
        cv_scores_adjusted = np.sqrt(-cv_scores)
        return np.mean(cv_scores_adjusted), np.std(cv_scores_adjusted), list(cv_scores_adjusted)

    @staticmethod
    def kaggle_submission(predict_df: pd.DataFrame, model: ModelHandler, data: pd.DataFrame):
        y_pred = model.predict(data)

        output = pd.DataFrame({'Id': predict_df.Id, 'SalePrice': y_pred})
        output.to_csv('submission-next-test.csv', index=False)

        print('Submission created.')
