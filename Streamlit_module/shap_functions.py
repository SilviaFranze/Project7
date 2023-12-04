# from streamlit_shap import st_shap
import shap
import joblib
import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

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

    print ('baaaaaaaaaaaaaseeeeeeeeeeeeeeeee valueeeeeeeeeeeeee in force plot')
    print(type(base_value[0]))

    # Genera il force plot
    shap.initjs()  # Necessario per SHAP plots interattivi
    shap.force_plot(base_value[0], client_shap_values[0], client_data, matplotlib=True, show=False)

    # Mostra il plot in Streamlit
    st.pyplot(plt)


import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def generate_waterfall_plot(client_id, data, explainer):
    client_id = int(client_id)
    example_instance = data.loc[[client_id]].drop('SK_ID_CURR', axis=1, errors='ignore')
    shap_values = explainer.shap_values(example_instance)
    
    # Prendiamo i valori SHAP per il cliente specifico
    shap_singolo = shap_values[0][0]

    # Seleziona le 10 feature con il maggiore impatto assoluto
    indices = np.argsort(-np.abs(shap_singolo))[:10]
    feature_names = example_instance.columns[indices].tolist()
    shap_singolo_top = shap_singolo[indices]

    # Preparazione dei dati per il grafico waterfall
    cumulative = np.cumsum(shap_singolo_top)

    # Crea la figura
    fig, ax = plt.subplots()

    # Disegna le barre
    bars = ax.bar(range(len(shap_singolo_top)), shap_singolo_top, align='center', color=np.where(shap_singolo_top >= 0, 'C0', 'C3'))

    # Aggiungi i valori SHAP come testo sopra le barre
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -12),  # spostamento leggero verso l'alto o verso il basso
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=8, color='black')

    # Aggiungi linee orizzontali per collegare le barre
    for i in range(1, len(shap_singolo_top)):
        plt.plot([i-1, i], [cumulative[i-1], cumulative[i-1]], color='gray')

    # Imposta i nomi delle feature sull'asse x
    plt.xticks(range(len(shap_singolo_top)), feature_names, rotation=45, ha='right')

    # Aggiungi titoli e etichette
    plt.xlabel('Features')
    plt.ylabel('SHAP Value')
    plt.title(f'Waterfall Plot for Client {client_id} - Top 10 Impactful Features')

    # Visualizza il plot in Streamlit
    st.pyplot(fig)






'''
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
    client_shap_values = explainer.shap_values(client_data)[0].flatten()

    expected_value = explainer.expected_value[0]
    print ('expecteeeeeeeeeeeed valueeeeeeeeeeeeeeeeee')
    print (expected_value)

    

    # Utilizza shap.waterfall_plot con i parametri corretti
    shap.plots.waterfall(shap.Explanation(values = expected_value, 
                                         base_values=client_shap_values, 
                                         data=client_data.values, 
                                         feature_names=client_data.columns.tolist()), show=False)

  
    # client_data.values[0], client_data.columns.tolist(), show=False)              , features= client_data, feature_names=client_data.columns.tolist(), show=False

    
    print ('expecteeeeeeeeeeeed valueeeeeeeeeeeeeeeeee')
    #print (expected_value)
    print('shap values')
    print(shap_values[0][0])
    print('data values')
    print(client_data.values[0])
    '''
    # shap_values_array = shap_values[0][0] if isinstance(shap_values, list) else shap_values

    #def generate_waterfall_plot(shap_values, data_values, feature_names):
    # Assicurati che shap_values, data_values e feature_names siano nel formato corretto
    # shap_explanation = shap.Explanation(values=shap_values, data=client_data.values, feature_names=client_data.columns.tolist())
    # shap_explanation = shap.Explanation(values=shap_values_array, data=client_data.values[0], feature_names=client_data.columns.tolist())


    # Genera il Waterfall Plot
    # shap.waterfall_plot(shap_explanation, show=False)
   
    # Crea e mostra il Waterfall Plot
    

    
