#import shap
# from streamlit_shap import st_shap
import shap
import joblib
import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt

lightgbmodel =  joblib.load("Streamlit_module/lightgbmodelsh.joblib")
explainer = shap.TreeExplainer(lightgbmodel)

def generate_force_plot(client_id, data, model_explainer):
    """
    Genera un force plot per un ID cliente specifico.
    
    Parametri:
    - client_id: ID del cliente di interesse
    - data: DataFrame contenente i dati dei clienti (deve avere una colonna 'SK_ID_CURR')
    - model_explainer: Oggetto SHAP TreeExplainer già addestrato
    
    Restituisce:
    - Un force plot per l'ID cliente specificato  
    """
    client_id = int(client_id)
    client_data = data.loc[[client_id]].drop('SK_ID_CURR', axis=1, errors='ignore')

    # Calcola i valori SHAP per il cliente specifico
    client_shap_values = model_explainer.shap_values(client_data)
    
    # Ottiene il valore atteso dal model_explainer
    base_value = model_explainer.expected_value

    # Genera il force plot
    shap.initjs()  # Necessario per SHAP plots interattivi
    shap.force_plot(base_value[0], client_shap_values[0], client_data, matplotlib=True, show=False)

    # Mostra il plot in Streamlit
    st.pyplot(plt)


def generate_waterfall_plot(client_id, data, explainer):
    """
    Genera un waterfall plot per un ID cliente specifico.
    
    Parametri:
    - client_id: ID del cliente di interesse
    - data: DataFrame contenente i dati dei clienti (deve avere una colonna 'SK_ID_CURR')
    - explainer: Oggetto SHAP Explainer già addestrato
    
    Restituisce:
    - Un waterfall plot per l'ID cliente specificato
    """
    client_id = int(client_id)
    client_data = data.loc[[client_id]].drop('SK_ID_CURR', axis=1)

    # Calcola i valori SHAP per l'istanza specifica
    shap_values = explainer.shap_values(client_data)

    # Calcola il valore base del modello
    expected_value = explainer.expected_value[0]
    
    # Crea e mostra il Waterfall Plot
    shap.waterfall_plot(shap.Explanation(values=shap_values[0][0], 
                                         base_values=expected_value, 
                                         data=client_data.values[0], 
                                         feature_names=client_data.columns.tolist()), show=False)

    # Mostra il plot in Streamlit
    st.pyplot(plt)