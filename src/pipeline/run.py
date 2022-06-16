import os
import sys

import yaml

sys.path.extend(os.pardir)

from src.data.handlers import CSVLoader
from src.models.train import ModelHandler
from src.models.helpers import preprocess, extract_preproc_config
from src.features.splitters import DataSplitter
from src.models.evaluators import ModelEvaluator

# PARAMS

params_path = sys.argv[1]
params = yaml.safe_load(open(params_path))

# INIT

os.makedirs(params['dvc']['auto']['dir'], exist_ok=True)

# CODE

# Single training

train_csv = CSVLoader(params['prepare']['data']['train']['path'])
all_df = train_csv.load()

y = all_df['SalePrice']
X = all_df.drop(columns=['Id', 'SalePrice'], axis=1)

splitter = DataSplitter()
X_train, X_val, y_train, y_val = splitter.split(X, y, 0.75)

preproc_config = extract_preproc_config(params)
print(f'Options used for preprocessing: {preproc_config}')
preprocess(X_train, X_val, y_train, preproc_config)

model = ModelHandler(params['prepare']['train']['name'])
model.fit(X_train, y_train)

evaluator = ModelEvaluator()
r2_train, rmsle_train = evaluator.metrics(model, X_train, y_train)
r2_val, rmsle_val = evaluator.metrics(model, X_val, y_val)

# Cross validation

preprocess(X, None, y, preproc_config)
cv_mean, cv_std, cross_val = evaluator.cv_metrics(model, X, y)

# Persisting metrics

evaluator.save_metrics(path=params['dvc']['metrics']['train']['path'], metrics={
    'r2_train': r2_train,
    'r2_val': r2_val,
    'rmsle_train': rmsle_train,
    'rmsle_val': rmsle_val,
    'cv_rmsle_mean_score': cv_mean,
    'cv_rmsle_std': cv_std
})
