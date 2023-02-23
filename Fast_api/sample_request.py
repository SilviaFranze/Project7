import json
import requests
import joblib

input_data = joblib.load("../input_data.joblib")

def get_client(x_test):
    features_list=[]
    values_list=[]
    for k,v in x_test.sample().to_dict().items():
        features_list.append(k)
        for key,val in v.items():
            values_list.append(val)

    return dict(zip(features_list, values_list))

response = requests.post("http://localhost:8000/predict", json=get_client(input_data))
print(response.text)