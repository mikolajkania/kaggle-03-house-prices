import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score

from src.models.train import ModelHandler


class ModelEvaluator:

    def __init__(self, model: ModelHandler):
        self.model = model

    def metrics(self, data: pd.DataFrame, y: pd.Series):
        y_pred = self.model.get().predict(data)

        r2 = self.model.get().score(data, y)
        rmsle = mean_squared_log_error(y, y_pred, squared=False)

        return r2, rmsle

    def feature_importance(self, data: pd.DataFrame):
        if hasattr(self.model.get(), 'feature_importances_'):
            importance = zip(self.model.get().feature_importances_, data.columns)
            importance_sorted = sorted(importance, key=lambda x: x[0], reverse=True)
            for score, col in importance_sorted:
                print('Feature: %s, score: %.5f' % (col, score))

    @staticmethod
    def save_metrics(path: str, metrics: dict):
        print(metrics)
        with open(path, 'w') as f:
            f.write(json.dumps(metrics))

    def cv_metrics(self, data: pd.DataFrame, y: pd.Series):
        cv_scores = cross_val_score(self.model.get(), data, y, scoring='neg_mean_squared_log_error', cv=5)
        cv_scores_adjusted = np.sqrt(-cv_scores)
        return np.mean(cv_scores_adjusted), np.std(cv_scores_adjusted), list(cv_scores_adjusted)

    def kaggle_submission(self, predict_df: pd.DataFrame, data: pd.DataFrame):
        y_pred = self.model.predict(data)

        output = pd.DataFrame({'Id': predict_df.Id, 'SalePrice': y_pred})
        output.to_csv('submission-next-test.csv', index=False)

        print('Submission created.')
