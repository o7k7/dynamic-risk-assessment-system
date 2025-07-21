from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

from ingestion import merge_multiple_dataframe
from training import FEATURE_COLS, TARGET_COL

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'], config['training_model_filename'])


def read_test_data():
    df = merge_multiple_dataframe(config['test_data_path'])
    return df


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    try:
        with open(model_path,'r') as f:
            model = pickle.load(f)
            df = read_test_data()
            x = df.loc[:, FEATURE_COLS].values.reshape(-1, 2)
            Y = df[TARGET_COL].values.reshape(-1, 1).ravel()

            predicted = model.predict(x)

            f1score = metrics.f1_score(predicted, Y)
            create_log_for_f1_score(f1score)
    except:
        raise Exception('Could not load model')


def create_log_for_f1_score(score):
    fullpath = os.path.join(config['output_model_path'], 'latestscore.txt')

    with open(fullpath,'w') as f:
        f.write(score)

