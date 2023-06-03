
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, request, jsonify
import joblib
import pandas as pd

#Load customer data
input_data_scaled = joblib.load("/home/silviafranze/X_tst_light.joblib")   #   X_tst_sld_skid.joblib was the original one
# Load the LightGBM model
lgbm_classif = joblib.load("/home/silviafranze/lightgbmodel.joblib")

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/')
def home():
    return 'Welcome to the Homepage'

@app.route('/prediction/<int:id_client>', methods =['GET'])
def prediction(id_client):

    '''
    Endpoint to get the client id and return the prediction based on a pre trained LightGBM model
    '''

    # Select customer data specified by ID
    # selected_customer = input_data_scaled.loc[id_client].to_numpy().reshape(1,-1)
    selected_customer = input_data_scaled.loc[id_client].values.reshape(1,-1)


    # makes the prediction about a specific client
    prediction = lgbm_classif.predict_proba(selected_customer)[:,0][0]

    # determines whether the application was accepted or rejected on the basis of the 0.90 threshold
    if prediction > 0.90:
        decision = "accepted"
    else:
        decision = "refused"

    # returns a dictionary with the client ID and the decision made
    response = {"Customer id": id_client,
                "Decision": decision}

    return jsonify(response)


    # return f'This will be the prediction score app!{id_client}'
