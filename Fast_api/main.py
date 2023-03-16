# 1. Library imports
import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from lightgbm import LGBMClassifier

app = FastAPI()
input_data_scaled = joblib.load("../Data&output/test_scaled_data_ok.joblib")
light_classif = joblib.load("../Data&output/lightgbmodel.joblib")


# utiliser le get pour choisir l id

@app.get('/prediction/{id_client}')
def read_id(id_client: int):
	"""
	Endpoint to	"""
	client = input_data_scaled.loc[input_data_scaled['SK_ID_CURR'] == {id_client}].iloc[:,1:]
	pd.DataFrame(light_classif.predict_proba(client))
	return {'message': id_client,
			'decision': 'accordé'}


if __name__ == '__main__':
	uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn main:app --reload

