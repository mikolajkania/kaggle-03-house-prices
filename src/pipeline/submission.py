import os
import sys

import yaml

sys.path.extend(os.pardir)

from src.data.handlers import CSVLoader
from src.models.train import ModelResolver
from src.models.evaluators import ModelEvaluator
from src.features.original import TypeTransformer
from src.models.preproc import preprocess, extract_preproc_config
from src.features.artificial import FeatureCreator

from sklearn.ensemble import StackingRegressor, VotingRegressor

# PARAMS

params_path = sys.argv[1]
params = yaml.safe_load(open(params_path))

# CODE

train_csv = CSVLoader(params['train']['data']['train']['path'])
all_df = train_csv.load()

predict_csv = CSVLoader(params['train']['data']['predict']['path'])
predict_df = predict_csv.load()
X_pred = predict_df.drop(columns=['Id'], axis=1)

if params['train']['preproc']['create_features']:
    features = FeatureCreator()
    features.create_new(data=all_df)
    features.create_new(data=X_pred)

if params['train']['preproc']['transform_types']:
    typeTrans = TypeTransformer()
    typeTrans.transform(data=all_df)
    typeTrans.transform(data=X_pred)

y = all_df['SalePrice']
X = all_df.drop(columns=['Id', 'SalePrice'], axis=1)

preproc_config = extract_preproc_config(params)
preprocess(X, X_pred, y, preproc_config)

estimators_names = params['train']['estimator']['names']
if len(estimators_names) == 1:
    model = ModelResolver.of(name=estimators_names[0])
    model.fit(X, y)
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
                               weights=[0.15, 0.35, 0.45, 0.05])
    vote_reg.fit(X, y)
    evaluator = ModelEvaluator(vote_reg)

if params['train']['eval']['feature_importance']:
    evaluator.feature_importance(X)
r2_train, rmsle_train = evaluator.metrics(X, y)

evaluator.save_metrics(path=params['dvc']['metrics']['final']['path'], metrics={
    'r2_train': r2_train,
    'rmsle_train': rmsle_train,
})
evaluator.kaggle_submission(predict_df, X_pred)
