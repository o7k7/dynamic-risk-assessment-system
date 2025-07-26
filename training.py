import pandas as pd
import numpy as np
import pickle
import os
import io
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

###################Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

FEATURE_COLS = ['lastmonth_activity', 'lastyear_activity', 'number_of_employees']
TARGET_COL = 'exited'

dataset_csv_path = os.path.join(config['output_folder_path'], config['training_data_filename'])
model_path = os.path.join(config['output_model_path'], config['trained_model_filename'])


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
    df = read_data()

    X = df.loc[:, FEATURE_COLS]
    y = df[TARGET_COL]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', FEATURE_COLS),
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(multi_class='auto', solver='liblinear'))
    ])

    model_pipeline.fit(X, y)
    # write the trained model to your workspace in a file called trainedmodel.pkl
    with open(model_path, 'wb') as file:
        pickle.dump(model_pipeline, file)


if __name__ == '__main__':
    train_model()