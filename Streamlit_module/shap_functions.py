import shap
from streamlit_shap import st_shap
import streamlit as st 

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
    
    # Seleziona i dati del cliente specificato dall'ID
    client_data = data[data['SK_ID_CURR'] == client_id].drop('SK_ID_CURR', axis=1)
    
    # Calcola i valori SHAP per il cliente specifico
    if client_data.empty or client_data.ndim != 2:
        st.error(f"Client data is empty or not 2D! Shape: {client_data.shape}")
    else:
        client_shap_values = model_explainer.shap_values(client_data)
    
    # Ottiene il valore atteso dal model_explainer
    base_value = model_explainer.expected_value

    # Genera e restituisce il force plot
    return shap.force_plot(base_value[0], client_shap_values[0], client_data)



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
    # Seleziona i dati per il cliente specifico
    example_instance = data[data['SK_ID_CURR'] == client_id].drop('SK_ID_CURR', axis=1)
    
    # Calcola i valori SHAP per l'istanza specifica
    shap_values = explainer.shap_values(example_instance)
    
    # Calcola il valore base del modello
    expected_value = explainer.expected_value[0]  # Stiamo lavorando con la classe negativa
    
    # Visualizza l'ID del Cliente
    print("SHAP Waterfall Plot per il Cliente ID:", client_id)
    
    # Crea e mostra il Waterfall Plot
    shap.plots.waterfall(shap.Explanation(values=shap_values[0][0], 
                                          base_values=expected_value, 
                                          data=example_instance.values[0], 
                                          feature_names=example_instance.columns.tolist()))