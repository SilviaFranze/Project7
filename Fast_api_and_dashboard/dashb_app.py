import streamlit as st
import requests
import shap
import pandas as pd
import joblib

# loading light gbm model, shap explainer, client ids, and data for SHAP
input_data_scaled = joblib.load("../Data&output/X_tst_sld_skid.joblib")   # customer data
lgbm_classif = joblib.load("../Data&output/lightgbmodel.joblib")
explainer = joblib.load("../Data&output/shap_explainer.joblib")
client_data_for_shap = joblib.load("../Data&output/test_data.joblib")
client_ids = joblib.load("../Data&output/list_id_clients.joblib")

st.title("Scoring prediction")
st.write('Select the customer\'s ID to make a prediction on their loan request.')


# URL FastAPI
api_url = "http://127.0.0.1:8000/prediction"

# Client id selection through a list
st.subheader("Client selection:")
selected_client_id = str(st.selectbox("Select client ID", client_ids))

# Crea un pulsante per fare la richiesta GET all'API
if st.button('Predict'):
    # Effettua la richiesta GET passando l'id del cliente come parametro
    response = requests.get(api_url + "/" + selected_client_id)

    # Estrapola la decisione dalla risposta dell'API
    decision = response.json()['Decision']

    # Mostra la decisione all'utente
    st.write('The prediction for the client num', selected_client_id, 'is', decision)




    # streamlit run .\dashb_app.py
