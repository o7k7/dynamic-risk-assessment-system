import json
import logging
import os
import subprocess
import sys

LOG_FILE = os.path.join('fullprocess.log')

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode='a', # 'a' for append
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info(f"Starting fullprocess.py script")
with open('config.json', 'r') as f:
    config = json.load(f)

def run_process():
    source_data_path = config['input_folder_path']
    prod_deployment_path = config['prod_deployment_path']
    ingested_files_record = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    score_file = os.path.join(prod_deployment_path, 'latestscore.txt')
    ##################Check and read new data
    #first, read ingestedfiles.tx
    with open(ingested_files_record, 'r') as f:
        ingested_files = set(f.read().splitlines())

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    source_files = set(os.listdir(source_data_path)) # Avoid duplicates
    new_files = source_files - ingested_files


    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if not new_files:
        logging.info("No new data found. Terminating the process.")
        sys.exit()

    logging.info(f"New data found at {source_data_path}. Running ingestion.")
    subprocess.run(['python', 'ingestion.py'], check=True)

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    logging.info("Checking for model drift")
    # Get the score from the deployed model
    with open(score_file, 'r') as f:
        last_score = float(f.read())
    logging.info(f"Last deployed score = {last_score}")

    subprocess.run(['python', 'scoring.py', config['input_folder_path']], check=True)
    with open(score_file, 'r') as f:
        new_score = float(f.read())
    logging.info(f"New model score = {new_score}")

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if new_score >= last_score:
        logging.info("No model drift detected. Exiting process.")
        sys.exit()

    logging.info(f"Model drift detected. New score: {new_score}. Last score: {last_score}).")



    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    logging.info("Re-deploying model.")
    subprocess.run(['python', 'training.py'], check=True)
    subprocess.run(['python', 'deployment.py'], check=True)

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
    logging.info(" Running diagnostics and reporting")
    subprocess.run(['python', 'reporting.py'], check=True)
    subprocess.run(['python', 'apicalls.py'], check=True)

    logging.info("Full process completed.")




if __name__ == '__main__':
    run_process()


