import os
import sys

import yaml

sys.path.extend(os.pardir)

from src.data.handlers import CSVLoader
from src.models.train import ModelResolver
from src.features.original import TypeTransformer
from src.models.preproc import preprocess, extract_preproc_config
from src.features.artificial import FeatureCreator
from src.features.splitters import DataSplitter
from src.models.evaluators import ModelEvaluator

from sklearn.ensemble import StackingRegressor, VotingRegressor

# PARAMS

params_path = sys.argv[1]
params = yaml.safe_load(open(params_path))

# INIT

os.makedirs(params['dvc']['auto']['dir'], exist_ok=True)

# CODE

# Single training

train_csv = CSVLoader(params['train']['data']['train']['path'])
all_df = train_csv.load()

if params['train']['preproc']['create_features']:
    features = FeatureCreator()
    features.create_new(data=all_df)

if params['train']['preproc']['transform_types']:
    typeTrans = TypeTransformer()
    typeTrans.transform(data=all_df)

y = all_df['SalePrice']
X = all_df.drop(columns=['Id', 'SalePrice'], axis=1)

splitter = DataSplitter()
X_train, X_val, y_train, y_val = splitter.split(X, y, 0.75)

preproc_config = extract_preproc_config(params)
print(f'Options used for preprocessing: {preproc_config}')
preprocess(X_train, X_val, y_train, preproc_config)

estimators_names = params['train']['estimator']['names']
if len(estimators_names) == 1:
    model = ModelResolver.of(name=estimators_names[0])
    model.fit(X_train, y_train)
    evaluator = ModelEvaluator(model.get())
else:
    estimators = []
    for est in estimators_names:
        model = ModelResolver.of(name=est).get()
        estimators.append((est, model))
    # stacked_reg = StackingRegressor(estimators=estimators[1:],
    #                                 final_estimator=estimators[0][1],
    #                                 passthrough=False)
    vote_reg = VotingRegressor(estimators=estimators,
                               weights=[0.2, 0.35, 0.45])
    vote_reg.fit(X_train, y_train)
    evaluator = ModelEvaluator(vote_reg)

if params['train']['eval']['feature_importance']:
    evaluator.feature_importance(data=X_train)
r2_train, rmsle_train = evaluator.metrics(X_train, y_train)
r2_val, rmsle_val = evaluator.metrics(X_val, y_val)

# Cross validation

if params['train']['steps']['cv']:
    preprocess(X, None, y, preproc_config)
    cv_mean, cv_std, cross_val = evaluator.cv_metrics(X, y)

# Grid search
if params['train']['steps']['grid']:
    model.grid_search(X, y)

# Persisting metrics

metrics = {
    'r2_train': r2_train,
    'r2_val': r2_val,
    'rmsle_train': rmsle_train,
    'rmsle_val': rmsle_val,
}
if params['train']['steps']['cv']:
    metrics['cv_rmsle_mean_score'] = cv_mean
    metrics['cv_rmsle_std'] = cv_std

evaluator.save_metrics(path=params['dvc']['metrics']['train']['path'], metrics=metrics)
