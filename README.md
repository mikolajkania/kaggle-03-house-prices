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

The main idea for this competition was to get high score, but by using repeatable process, instead of only experimenting in Jupyter Notebook. Tracking, results persistence, reproducibility were among crucial objectives that I set before entering it.

Notable results based on few popular algorithms can be found below (the lower, the better):

|          Notable submissions          |  CV mean error | Public score (error) |        Branch         |
|:-------------------------------------:|---------------:|---------------------:|:---------------------:|
| Model voting: RidgeCV, LGBM, XGB, RF  |        0.11944 |              0.12556 |      exp-voting       |
| Models stacking: Ridge, LGBM, XGB, RF |        0.12343 |              0.12665 |     exp-stacking      |
|         LightGBM with tuning          |        0.12640 |              0.12627 |     exp-lgbm-tune     |
|      XGBoost with further tuning      |        0.12394 |              0.13038 |    exp-xgb-tune-2     |
|   LightGBM with feature engineering   |        0.12966 |              0.13045 |     exp-features      |
|               LightGBM                |        0.12995 |              0.13199 |       exp-lgbm        |
|    XGBoost - learning rate tuning     |        0.13692 |              0.14386 | exp-xgb-learning-rate |
|                XGBoost                |        0.13692 |              0.14386 |        exp-xgb        |
|            Random Forests             |        0.14055 |              0.14455 |         Main          |
|          Logistic regression          |        0.15459 |              0.34370 |         Main          |