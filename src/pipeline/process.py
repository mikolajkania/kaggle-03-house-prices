import json
import os
import sys

import yaml

sys.path.extend(os.pardir)

from src.models.train import ModelHandler
from src.features.splitters import DataSplitter
from src.models.evaluators import ModelEvaluator
from src.features.vectorizes import StringEncoder
from src.features.columns import MissingDataHandler
from src.data.handlers import CSVLoader, TrainingDataHandler

# PARAMS

params_path = sys.argv[1]
params = yaml.safe_load(open(params_path))

# INIT

os.makedirs(params['dvc']['auto']['dir'], exist_ok=True)

# CODE

csv_data = CSVLoader(params['prepare']['data']['input']['path'])
X, y = csv_data.load()

missing_data = MissingDataHandler()
missing_data.drop_na_columns(X)
mode = params['prepare']['preproc']['missing']
missing_data.fill(mode, X)

str_encoder = StringEncoder()
X = str_encoder.encode(data=X)

train_data = TrainingDataHandler(params['dvc']['auto']['path'])
train_data.write(X, y)

splitter = DataSplitter()
X_train, X_val, y_train, y_val = splitter.split(X, y, 0.8)

model = ModelHandler('LinearRegression')
model.fit(X_train, y_train)

evaluator = ModelEvaluator()
metrics_dict = evaluator.metrics(model.get(), X_train, y_train, X_val, y_val)

with open(params['dvc']['metrics']['path'], 'w') as f:
    f.write(json.dumps(metrics_dict))
