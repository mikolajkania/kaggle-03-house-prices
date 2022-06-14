import os
import sys

import yaml
import pandas as pd


sys.path.extend(os.pardir)

from src.models.train import ModelHandler
from src.models.evaluators import ModelEvaluator
from src.features.encoders import CategoricalEncoder
from src.features.original import MissingDataHandler, CorrelationHandler, OutliersHandler, DistributionTransformer, \
    FeatureScaler
from src.data.handlers import CSVLoader

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

category_encoder = CategoricalEncoder()
category_encoder.fit(data=X)
category_encoder.transform(data=X)
category_encoder.transform(data=X_pred)

missing_data = MissingDataHandler(mode=params['prepare']['preproc']['missing'], data=X)
missing_data.transform(data=X)
missing_data.transform(data=X_pred)
missing_data.drop_high_na_columns(data=X)
missing_data.drop_high_na_columns(data=X_pred)

if params['prepare']['preproc']['outliers_removal']:
    outliers = OutliersHandler()
    outliers.fit(data=X)
    outliers.transform(data=X)
    outliers.transform(data=X_pred)

correlation = CorrelationHandler(mode=params['prepare']['preproc']['corr_threshold'], data=X, y=y)
correlation.transform(data=X)
correlation.transform(data=X_pred)

scaler = FeatureScaler()
scaler.fit(data=X)
scaler.transform(data=X)
scaler.transform(data=X_pred)

target_transform = params['prepare']['preproc']['target_transform']
if target_transform['enabled']:
    DistributionTransformer.transform_df(data=X, lmbda=target_transform['lambda'])
    DistributionTransformer.transform_df(data=X_pred, lmbda=target_transform['lambda'])

model = ModelHandler('LinearRegression')
model.fit(X, y)

evaluator = ModelEvaluator()
r2_final, rmsle_final = evaluator.metrics(model.get(), X, y)
evaluator.save_metrics(path=params['dvc']['metrics']['final']['path'], metrics={
    'r2_final': r2_final,
    'rmsle_final': rmsle_final,
})

y_pred = model.predict(X_pred)

output = pd.DataFrame({'Id': predict_df.Id, 'SalePrice': y_pred})
output.to_csv('submission-next.csv', index=False)

print('Submission created.')