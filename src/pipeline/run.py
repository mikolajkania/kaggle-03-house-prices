import os
import sys

import yaml

sys.path.extend(os.pardir)

from src.models.train import ModelHandler
from src.features.splitters import DataSplitter
from src.models.evaluators import ModelEvaluator
from src.features.encoders import CategoricalEncoder
from src.features.original import MissingDataHandler, CorrelationHandler, OutliersHandler, DistributionTransformer, \
    FeatureScaler
from src.data.handlers import CSVLoader

# PARAMS

params_path = sys.argv[1]
params = yaml.safe_load(open(params_path))

# INIT

os.makedirs(params['dvc']['auto']['dir'], exist_ok=True)

# CODE

csv_data = CSVLoader(params['prepare']['data']['input']['path'])
all_df = csv_data.load()

y = all_df['SalePrice']
X = all_df.drop(columns=['Id', 'SalePrice'], axis=1)

splitter = DataSplitter()
X_train, X_val, y_train, y_val = splitter.split(X, y, 0.75)

category_encoder = CategoricalEncoder()
category_encoder.encode(train=X_train, val=X_val)

missing_data = MissingDataHandler(mode=params['prepare']['preproc']['missing'], data=X_train)
missing_data.fill(X_train)
missing_data.fill(X_val)
missing_data.drop_high_na_columns(X_train)
missing_data.drop_high_na_columns(X_val)

if params['prepare']['preproc']['outliers_removal']:
    outliers = OutliersHandler()
    outliers.fit(X_train)
    outliers.transform(X_train)
    outliers.transform(X_val)

correlation = CorrelationHandler(mode=params['prepare']['preproc']['corr_threshold'], data=X_train, y=y_train)
correlation.transform(X_train, X_val)

scaler = FeatureScaler()
scaler.fit(X_train)
scaler.transform(X_train, X_val)

target_transform = params['prepare']['preproc']['target_transform']
if target_transform['enabled']:
    y_train = DistributionTransformer.transform(data=y_train, lmbda=target_transform['lambda'])
    y_val = DistributionTransformer.transform(data=y_val, lmbda=target_transform['lambda'])

model = ModelHandler('LinearRegression')
model.fit(X_train, y_train)

evaluator = ModelEvaluator()
metrics_dict = evaluator.metrics(model.get(), X_train, y_train, X_val, y_val)
evaluator.save_metrics(path=params['dvc']['metrics']['path'], metrics=metrics_dict)
