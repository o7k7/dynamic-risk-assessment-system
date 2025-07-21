from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import io
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

FEATURE_COLS = ['corporation', 'lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET_COL = 'exited'

dataset_csv_path = os.path.join(config['output_folder_path'], config['training_data_filename'])
model_path = os.path.join(config['output_model_path'], config['training_model_filename'])


def read_data() -> pd.DataFrame:
    try:
        with open(dataset_csv_path, 'r') as f:
            content = f.read()
            df = pd.read_csv(io.StringIO(content))

            return df
    except Exception as e:
        raise Exception(f"Unable to read training data : {e}")


#################Function for training the model
def train_model():
    # use this logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='warn', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    df = read_data()

    X = df.loc[:, FEATURE_COLS].values.reshape(-1, 2)
    y = df[TARGET_COL].values.reshape(-1, 1).ravel()

    # fit the logistic regression to your data
    model = logit.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
