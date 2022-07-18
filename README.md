# kaggle-03-house-prices
- https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

# Repository template
- https://drivendata.github.io/cookiecutter-data-science/

# Python package
- python -m venv envs/kaggle03-py3.9
- pip install pip-tools
- pip-compile requirements.in
- pip install -r requirements.txt

# Results

In many Kaggle solutions you don't see reasoning that brought data scientist to final output: it might be hard research, but also coping from other developers.

My main idea for this competition was to get high score, but to not rely only on one, sacred Jupyter Notebook. Tracking, results persistence, reproducibility were among crucial objectives that I set before entering it.

Jupyter Notebooks were of course present - they are great tool for performing analysis of datasets or models. They can be found in this repo as well.

Notable results based on few popular algorithms can be found below (the lower, the better):

|                             Notable submissions                             |  CV mean error | Public score (error) |        Branch         |
|:---------------------------------------------------------------------------:|---------------:|---------------------:|:---------------------:|
| Model voting with weights: RidgeCV (0.1), XGB (0.55), LGBM (0.3), RF (0.05) |        0.11944 |              0.12556 |      exp-voting       |
|                    Models stacking: Ridge, XGB, LGBM, RF                    |        0.12343 |              0.12665 |     exp-stacking      |
|                            LightGBM with tuning                             |        0.12640 |              0.12627 |     exp-lgbm-tune     |
|                         XGBoost with further tuning                         |        0.12394 |              0.13038 |    exp-xgb-tune-2     |
|                      LightGBM with feature engineering                      |        0.12966 |              0.13045 |     exp-features      |
|                                  LightGBM                                   |        0.12995 |              0.13199 |       exp-lgbm        |
|                       XGBoost - learning rate tuning                        |        0.13692 |              0.14386 | exp-xgb-learning-rate |
|                                   XGBoost                                   |        0.13692 |              0.14386 |        exp-xgb        |
|                               Random Forests                                |        0.14055 |              0.14455 |         main          |
|                             Logistic regression                             |        0.15459 |              0.34370 |         main          |