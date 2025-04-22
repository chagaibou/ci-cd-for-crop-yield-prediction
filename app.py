import streamlit as st
import joblib
import numpy as np

# Charger le modèle
model = joblib.load('./model.plk')

# Définir la fonction de prédiction
def predict(features):
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction[0]

# Titre de l'application
st.title('Prédiction de Rendement des Cultures')

# Formulaire de saisie des caractéristiques
st.sidebar.header('Saisissez les caractéristiques')
rainfall_mm = st.sidebar.number_input('Precipitations (mm)', min_value=0.0, max_value=2000.0, value=100.0)
soil_quality_index = st.sidebar.number_input('Indice de Qualité du Sol', min_value=1.0, max_value=10.0, value=1.0)
farm_size_hectares = st.sidebar.number_input('Taille de la Ferme (hectares)', min_value=0.0, max_value=1000.0, value=10.0)
sunlight_hours = st.sidebar.number_input('Heures d\'Ensoleillement', min_value=0.0, max_value=24.0, value=8.0)
fertilizer_kg = st.sidebar.number_input('Quantité d\'Engrais (kg)', min_value=0.0, max_value=3000.0, value=50.0)

# Bouton de prédiction
if st.sidebar.button('Prédire'):
    features = [rainfall_mm, soil_quality_index, farm_size_hectares, sunlight_hours, fertilizer_kg]
    prediction = predict(features)
    st.success(f'Le rendement prédit est de {prediction:.2f} unités.')
