import streamlit as st
import requests
import shap
import pandas as pd
import joblib
from main import prediction_client

st.title("Scoring prediction")

# URL dell'API
API_URL = "http://127.0.0.1:8000/prediction"


# loading light gbm model and shap explainer and client ids
lgbm_classif = joblib.load("../Data&output/lightgbmodel.joblib")
explainer = joblib.load("../Data&output/shap_explainer.joblib")
client_ids = joblib.load("../Data&output/list_id_clients")
client_data_for_shap = joblib.load("../Data&output/test_data.joblib")

# streamlit run .\dashb_app.py

# Creazione dell'interfaccia utente con Streamlit
def app():
    # Scoring client
    st.title("Credit Scoring Dashboard")
    
    # Client id selection
    st.subheader("Client selection:")
    selected_client_id = st.selectbox("Select client ID", client_ids)
    
    # Credit prediction
    st.subheader("Credit Prediction:")
    prediction = prediction_client(selected_client_id)
    st.write(f"La previsione per il cliente {selected_client_id} è {prediction}.")
    
    # Feature importance SHAP
    st.subheader("Feature Importance graphics:")
    client_data = pd.read_csv(f"client_data/{selected_client_id}.csv")
    shap_values = explainer(client_data)
    shap.summary_plot(shap_values, client_data)
    st.pyplot(bbox_inches="tight")