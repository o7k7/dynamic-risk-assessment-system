from logging import Logger
from typing import List

import pandas as pd
import io
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


def read_and_log_files_of_input_folder() -> List[pd.DataFrame]:
    df_list = []
    file_names = []
    if os.path.exists(input_folder_path) and os.path.isdir(input_folder_path):
        for filename in os.listdir(input_folder_path):
            if filename.endswith(".csv"):
                full_path = os.path.join(input_folder_path, filename)
                file_names.append(filename)
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()

                        df_list.append(pd.read_csv(io.StringIO(content)))
                except Exception as e:
                    Logger.error(f"Could not read file {full_path}: {e}")

        create_record(file_names)

        return df_list
    else:
        raise Exception(f"folder {input_folder_path} doesn't exist")


#############Function for data ingestion
def merge_multiple_dataframe() -> pd.DataFrame:
    df_list = read_and_log_files_of_input_folder()
    if len(df_list) == 1:
        return df_list[0]

    merged_df = df_list[0]

    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, how='outer', on=df.columns.tolist())

    return merged_df


def write_output(df: pd.DataFrame):
    if os.path.exists(output_folder_path) and os.path.isdir(output_folder_path):
        full_path = os.path.join(output_folder_path, "finaldata.csv")
        try:
            with open(full_path, 'w') as f:
                f.write(df.to_csv(index=False))
        except Exception as e:
            raise Exception(f"Unable to write merged dataframes: {e}")


def create_record(list_of_files_read: List[str]):
    full_path = os.path.join(output_folder_path, "ingestedfiles.txt")
    try:
        with open(full_path, 'w') as f:
            f.write(datetime.now().isoformat() + '\n')
            for file_name in list_of_files_read:
                f.write(file_name + '\n')
    except Exception as e:
        raise Exception(f"Unable to write records: {e}")


if __name__ == '__main__':
    final_df = merge_multiple_dataframe()
    deduped_df = final_df.drop_duplicates()
    write_output(deduped_df)
