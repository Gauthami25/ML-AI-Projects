# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from features.process_features import standardize

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
                                           

RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')

# Raw data directories for Malawi data
MWI_DIR = os.path.join(RAW_DATA_DIR, 'KCP2017_MP', 'KCP_ML_MWI')

MWI_HOUSEHOLD = os.path.join(MWI_DIR, 'MWI_2012_household.dta')
MWI_INDIVIDUAL = os.path.join(MWI_DIR, 'MWI_2012_individual.dta')

COUNTRY_LIST = ['mwi', 'mwi-competition']
CATEGORY_LIST = ['train', 'test', 'questions']


def get_country_filepaths(country):
    if country not in COUNTRY_LIST:
        raise ValueError("{} not one of the countries we cover, which are {}".format(country, COUNTRY_LIST))

    country_dir = os.path.join(DATA_DIR, country)

    if not os.path.exists(country_dir):
        os.makedirs(country_dir)

    return (os.path.join(country_dir, 'train.pkl'),
            os.path.join(country_dir, 'test.pkl'),
            os.path.join(country_dir, 'questions.json'))


def split_features_labels_weights(path,
                                  weights=['wta_pop', 'wta_hh'],
                                  weights_col=['wta_pop'],
                                  label_col=['poor']):
    '''Split data into features, labels, and weights dataframes'''
    data = pd.read_pickle(path)
    return (data.drop(weights + label_col, axis=1),
            data[label_col],
            data[weights_col])


def load_data(path, selected_columns=None, ravel=True, standardize_columns='numeric'):
    X, y, w = split_features_labels_weights(path)
    if selected_columns is not None:
        X = X[[col for col in X.columns.values if col in selected_columns]]
    if standardize_columns == 'numeric':
        standardize(X)
    elif standardize_columns == 'all':
        standardize(X, numeric_only=False)
    if ravel is True:
        y = np.ravel(y)
        w = np.ravel(w)
    return (X, y, w)
