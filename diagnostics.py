import logging
import pickle
import subprocess
import sys
import time
from typing import List

import pandas as pd
import numpy as np
import timeit
import os
import json

##################Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

prod_model_path = os.path.join(config['prod_model_path'], config['trained_model_filename'])
dataset_csv_path = os.path.join(config['output_folder_path'], config['training_data_filename'])
test_data_path = os.path.join(config['test_data_path'])


##################Function to get model predictions
def model_predictions(df: pd.DataFrame) -> List:
    try:
        with open(prod_model_path, 'rb') as f:
            model = pickle.load(f)
            predictions = model.predict(df)

            return predictions.tolist()
    except Exception as e:
        logging.error(logging.ERROR, f"Model prediction error: {e}")
        return []


##################Function to get summary statistics
def dataframe_summary():
    df = pd.read_csv(dataset_csv_path)

    numeric_df = df.select_dtypes(include='number')

    means = numeric_df.mean()
    medians = numeric_df.median()
    std_devs = numeric_df.std()

    summary_list = []
    for col in numeric_df.columns:
        summary_list.append({
            'column': col,
            'mean': means[col],
            'median': medians[col],
            'std_dev': std_devs[col]
        })

    return summary_list


def check_missing_data():
    df = pd.read_csv(dataset_csv_path)

    missing_percentages = (df.isnull().sum() / len(df)) * 100

    return missing_percentages.tolist()


##################Function to get timings
def execution_time():
    timings = []

    ingestion_start_time = time.time()
    subprocess.run(['python', 'ingestion.py'], capture_output=True)
    ingestion_end_time = time.time()
    ingestion_timing = ingestion_end_time - ingestion_start_time
    timings.append(ingestion_timing)

    training_start_time = time.time()
    subprocess.run(['python', 'training.py'], capture_output=True)
    training_end_time = time.time()
    training_timing = training_end_time - training_start_time
    timings.append(training_timing)

    return timings


##################Function to check dependencies
def outdated_packages_list() -> str:
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--outdated'],
            capture_output=True,
            text=True,
            check=True  # This will raise an exception if the command fails
        )

        if result.stdout.strip():
            print(result.stdout)
            return result.stdout
        else:
            print("All packages are up-to-date.")
            return 'All packages are up-to-date.'
    except Exception as e:
        logging.error(f"Unable to find outdated packages: {e}")
        return None


if __name__ == '__main__':
    execution_time()
    model_predictions(pd.DataFrame({
        'lastmonth_activity': [120, 75, 5, 500],
        'lastyear_activity': [1400, 300, 25, 2500],
        'number_of_employees': [2, 15, 22, 50]
    }))

    dataframe_summary()
    check_missing_data()
    outdated_packages_list()
