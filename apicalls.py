import json
import os

import requests

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

prediction_url = f"{URL}/prediction"
scoring_url = f"{URL}/scoring"
summary_stats_url = f"{URL}/summarystats"
diagnostics_url = f"{URL}/diagnostics"

# Call each API endpoint and store the responses
response1 = requests.post(prediction_url, json={"file_path": "./testdata/testdata.csv"})

response2 = requests.get(scoring_url)

response3 = requests.get(summary_stats_url)

response4 = requests.get(diagnostics_url)

# combine all API responses
responses = {
    "prediction": response1.json(),
    "scoring": response2.json(),
    "summary_stats": response3.json(),
    "diagnostics": response4.json()
}

# write the responses to your workspace
with open('apireturns2.txt', 'w') as f:
    json.dump(responses, f, indent=4)

print("apireturn.txt file is generated.")