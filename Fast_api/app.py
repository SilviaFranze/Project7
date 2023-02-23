# 1. Library imports
import joblib
import uvicorn
from fastapi import FastAPI
from model_pred import Train,predict_proba
import numpy as np
import pickle
import pandas as pd
import json
import requests


# 2. Create the app object
app = FastAPI()

scaler = joblib.load("../Data&output/standardscaler.joblib")
light_classif = joblib.load("../Data&output/lightgbmodel.joblib")
features = joblib.load("../Data&output/features.joblib")
input_data = joblib.load("../Data&output/input_data.joblib")


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_pro(data: Train):
    data = data.dict()
    prediction, probability = light_classif.predict_proba([data[i] for i in features])

    return {
        'prediction': prediction,
        'probability': probability
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn app:app --reload
