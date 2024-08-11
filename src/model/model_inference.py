import json
import requests

# Online Inference
data = {
    "dataframe_split": {"columns": ['buying', 'maint', 'persons', 'lug_boot', 'safety'], "data": [[ 1, 1,4,1, 1]]},
    "params": {"model_name": "RandomForestClassifier"},
}

headers = {"Content-Type": "application/json"}
endpoint = "http://127.0.0.1:5001/invocations"

r = requests.post(endpoint, data=json.dumps(data), headers=headers)
print(r.status_code)
print(r.text)