from flask import Flask, session, jsonify, request, abort
import pandas as pd
import numpy as np
import pickle
import json
import os
from dotenv import load_dotenv

from diagnostics import model_predictions, dataframe_summary, check_missing_data, execution_time, outdated_packages_list
from scoring import score_model

load_dotenv()

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = os.getenv('API_SECRET_KEY')

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    if not request.is_json:
        abort(400, description="Invalid request: Missing JSON body.")

    data = request.get_json(force=True)
    file_path = data['file_path']

    if file_path is None:
        abort(400, description="Invalid request: 'file_path' field is missing.")
    try:
        df = pd.read_csv(file_path).drop(columns=['exited', 'corporation'], axis=1)
        predictions = model_predictions(df=df)

        return jsonify({
            'predictions': predictions
        })
    except Exception as e:
        abort(400, description="Unable to convert to dataframe")
        return None


#######################Scoring Endpoint
@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    # check the score of the deployed model
    return jsonify({'score': score_model()})


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # check means, medians, and modes for each column
    summary = dataframe_summary()
    return jsonify({'summary': summary})


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    # check timing and percent NA values
    missing_data_percentages = check_missing_data()
    execution_timings = execution_time()
    dependency_summary = outdated_packages_list()

    return jsonify({
        'execution_timings': execution_timings,
        'missing_data_percentages': missing_data_percentages,
        'dependency_summary': dependency_summary,
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
