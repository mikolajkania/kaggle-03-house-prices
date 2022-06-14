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

train_csv = CSVLoader(params['prepare']['data']['train']['path'])
all_df = train_csv.load()

y = all_df['SalePrice']
X = all_df.drop(columns=['Id', 'SalePrice'], axis=1)

splitter = DataSplitter()
X_train, X_val, y_train, y_val = splitter.split(X, y, 0.75)

category_encoder = CategoricalEncoder()
category_encoder.fit(data=X_train)
category_encoder.transform(data=X_train)
category_encoder.transform(data=X_val)

missing_data = MissingDataHandler(mode=params['prepare']['preproc']['missing'], data=X_train)
missing_data.transform(data=X_train)
missing_data.transform(data=X_val)
missing_data.drop_high_na_columns(data=X_train)
missing_data.drop_high_na_columns(data=X_val)

if params['prepare']['preproc']['outliers_removal']:
    outliers = OutliersHandler()
    outliers.fit(data=X_train)
    outliers.transform(data=X_train, verbose=False)
    outliers.transform(data=X_val)

correlation = CorrelationHandler(mode=params['prepare']['preproc']['corr_threshold'], data=X_train, y=y_train)
correlation.transform(data=X_train)
correlation.transform(data=X_val)

scaler = FeatureScaler()
scaler.fit(X_train)
scaler.transform(data=X_train)
scaler.transform(data=X_val)

target_transform = params['prepare']['preproc']['target_transform']
if target_transform['enabled']:
    DistributionTransformer.transform_df(data=X_train, lmbda=target_transform['lambda'], verbose=False)
    DistributionTransformer.transform_df(data=X_val, lmbda=target_transform['lambda'])

model = ModelHandler('LinearRegression')
model.fit(X_train, y_train)

evaluator = ModelEvaluator()
r2_train, rmsle_train = evaluator.metrics(model.get(), X_train, y_train)
r2_val, rmsle_val = evaluator.metrics(model.get(), X_val, y_val)
evaluator.save_metrics(path=params['dvc']['metrics']['train']['path'], metrics={
    'r2_train': r2_train,
    'r2_val': r2_val,
    'rmsle_train': rmsle_train,
    'rmsle_val': rmsle_val
})
