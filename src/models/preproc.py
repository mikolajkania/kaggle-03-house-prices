import os
import sys

import pandas as pd

sys.path.extend(os.pardir)

from src.features.encoders import CategoricalEncoder
from src.features.original import MissingDataHandler, CorrelationHandler, OutliersHandler, DistributionTransformer, \
    FeatureScaler, BinsTransformer


def extract_preproc_config(params: dict):
    return {
        'missing_imputer': params['train']['preproc']['missing'],
        'outliers_removal': params['train']['preproc']['outliers_removal'],
        'corr_threshold_removal': params['train']['preproc']['corr_threshold'],
        'distribution_lambda': params['train']['preproc']['distribution_lambda'],
        'years_bins': params['train']['preproc']['bins_years']['cols'],
        'years_bins_cnt': params['train']['preproc']['bins_years']['cnt']
    }


def preprocess(data: pd.DataFrame, val: pd.DataFrame, y: pd.Series, preproc_options: dict):
    category_encoder = CategoricalEncoder()
    category_encoder.fit(data=data)
    category_encoder.transform(data=data)
    if val is not None:
        category_encoder.transform(data=val)

    missing_data = MissingDataHandler(mode=preproc_options['missing_imputer'], data=data)
    missing_data.transform(data=data)
    if val is not None:
        missing_data.transform(data=val)
    missing_data.drop_high_na_columns(data=data)
    if val is not None:
        missing_data.drop_high_na_columns(data=val)

    if preproc_options['outliers_removal']:
        outliers_removal = OutliersHandler()
        outliers_removal.fit(data=data)
        outliers_removal.transform(data=data, verbose=False)
        if val is not None:
            outliers_removal.transform(data=val)

    for col in preproc_options['years_bins']:
        year_bins = BinsTransformer(col, preproc_options['years_bins_cnt'])
        year_bins.fit(data)
        year_bins.transform(data)
        if val is not None:
            year_bins.transform(val)

    if preproc_options['corr_threshold_removal'] is not None:
        correlation = CorrelationHandler(mode=preproc_options['corr_threshold_removal'], data=data, y=y)
        correlation.transform(data=data)
        if val is not None:
            correlation.transform(data=val)

    scaler = FeatureScaler()
    scaler.fit(data)
    scaler.transform(data=data)
    if val is not None:
        scaler.transform(data=val)

    if preproc_options['distribution_lambda']:
        lmbda = preproc_options['distribution_lambda']
        DistributionTransformer.transform_df(data=data, lmbda=float(lmbda), verbose=False)
        if val is not None:
            DistributionTransformer.transform_df(data=val, lmbda=float(lmbda))
