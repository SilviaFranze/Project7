import streamlit as st
import requests
import shap
import pandas as pd
import joblib

# loading light gbm model, shap explainer, client ids, and data for SHAP
input_data_scaled = joblib.load("Data&output/X_tst_sld_skid_long.joblib")   # customer data
lgbm_classif = joblib.load("Data&output/lightgbmodel.joblib")     # ../Data&output   pour le run en local
explainer = joblib.load("Data&output/shap_explainer.joblib")
client_data_for_shap = joblib.load("Data&output/test_data.joblib")     # /home/silviafranze pour le run sur python anywhere  # substitute them with the actual functioning dataset, to calculate the explainer etc
client_ids = joblib.load("Data&output/list_id_clients_long.joblib")

st.title("Scoring prediction")
st.write('Select the customer\'s ID to make a prediction on their loan request.')


# URL FastAPI
api_url = "https://silviafranze.pythonanywhere.com/prediction"

# Client id selection through a list
st.subheader("Client selection:")
selected_client_id = str(st.selectbox("Select client ID", client_ids))

# button to make the choice from the list of client ids to make the  GET request to the API
if st.button('Predict'):
    # Effettua la richiesta GET passando l'id del cliente come parametro
    response = requests.get(api_url + "/" + selected_client_id)

    # Retrieves the decision from the API respnse 
    decision = response.json()['Decision']

    # writes the final decision
    st.write('The decision for the client num', selected_client_id, 'is', decision)


st.title("Global importance of features")



    # streamlit run .\dashb_app.py