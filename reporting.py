import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions
from scoring import read_test_data

###############Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)


##############Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    df = read_test_data()

    X_test = df.drop(columns=['exited', 'corporation'], axis=1)
    y_true = df['exited']

    y_pred = model_predictions(X_test)
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    plot_path = os.path.join(config['output_model_path'], 'confusionmatrix.png')

    display.plot()

    plt.savefig(plot_path)


if __name__ == '__main__':
    score_model()
