import os
import sys

import yaml

sys.path.extend(os.pardir)

from src.features.vectorizes import StringEncoder
from src.features.columns import MissingDataHandler
from src.data.handlers import CSVLoader, TrainingDataHandler

# PARAMS

params_path = sys.argv[1]
params = yaml.safe_load(open(params_path))

# INIT

os.makedirs(params['dvc']['dir'], exist_ok=True)

# CODE

csv_data = CSVLoader(params['prepare']['data']['input']['path'])
X_train, y_train = csv_data.load()

missing_data = MissingDataHandler()
missing_data.drop_na_columns(X_train)
mode = params['prepare']['preproc']['missing']
missing_data.fill(mode, X_train)

str_encoder = StringEncoder()
X_train = str_encoder.encode(data=X_train)

train_data = TrainingDataHandler(params['dvc']['path'])
train_data.write(X_train, y_train)
