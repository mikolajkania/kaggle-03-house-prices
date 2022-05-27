import os
import sys
import yaml
import pickle

sys.path.extend(os.pardir)

from src.data.handlers import CSVHandler, TrainingDataHandler

# PARAMS

params_path = sys.argv[1]
params = yaml.safe_load(open(params_path))['prepare']['data']

# INIT

os.makedirs(params['dvc']['dir'], exist_ok=True)

# CODE

csv_data = CSVHandler(params['input']['path'])
all_df = csv_data.load()

y_train = all_df['SalePrice']
all_df.drop(columns=['Id', 'SalePrice'], axis=1)
X_train = all_df.copy()

train_data = TrainingDataHandler(params['dvc']['path'])
train_data.write(X_train, y_train)
