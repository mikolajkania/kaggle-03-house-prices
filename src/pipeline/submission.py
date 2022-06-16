import os
import sys

import yaml

sys.path.extend(os.pardir)

from src.data.handlers import CSVLoader
from src.models.train import ModelHandler
from src.models.evaluators import ModelEvaluator
from src.models.preproc import preprocess, extract_preproc_config

# PARAMS

params_path = sys.argv[1]
params = yaml.safe_load(open(params_path))

# CODE

train_csv = CSVLoader(params['prepare']['data']['train']['path'])
all_df = train_csv.load()

predict_csv = CSVLoader(params['prepare']['data']['predict']['path'])
predict_df = predict_csv.load()
X_pred = predict_df.drop(columns=['Id'], axis=1)

y = all_df['SalePrice']
X = all_df.drop(columns=['Id', 'SalePrice'], axis=1)

preproc_config = extract_preproc_config(params)
preprocess(X, X_pred, y, preproc_config)

model = ModelHandler(params['prepare']['train']['name'])
model.fit(X, y)

evaluator = ModelEvaluator()
r2_train, rmsle_train = evaluator.metrics(model, X, y)

evaluator.save_metrics(path=params['dvc']['metrics']['final']['path'], metrics={
    'r2_train': r2_train,
    'rmsle_train': rmsle_train,
})
evaluator.kaggle_submission(predict_df, model, X_pred)