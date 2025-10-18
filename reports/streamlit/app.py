import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk

# Configuración de la página
st.set_page_config(layout="wide", page_title="Visualizador y Predictor")

# --- Carga del Modelo ---
@st.cache_resource
def load_model():
    """Carga el modelo XGBoost tuneado."""
    try:
        model = joblib.load('../models/xgb_tuned_model.joblib')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- Barra Lateral para Entradas de Predicción ---
st.sidebar.header('Parámetros para Predicción')

def user_input_features():
    """Recopila las características de entrada del usuario desde la barra lateral."""
    # Ejemplo de características. Reemplaza con las características reales de tu modelo.
    feature1 = st.sidebar.slider('Característica 1', 0.0, 100.0, 50.0)
    feature2 = st.sidebar.slider('Característica 2', 0, 1000, 250)
    feature3 = st.sidebar.selectbox('Característica 3 (Categórica)', ('Opción A', 'Opción B', 'Opción C'))
    
    # Mapeo de la característica categórica a un valor numérico
    feature3_map = {'Opción A': 0, 'Opción B': 1, 'Opción C': 2}
    feature3_numeric = feature3_map[feature3]

    data = {
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3_numeric
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Contenido Principal ---
st.title('Plataforma de Visualización y Predicción MLOps')

tab1, tab2 = st.tabs(["Visualización de Mapa", "Predicción de Modelo"])

# --- Pestaña de Visualización de Mapa ---
with tab1:
    st.header('Mapa de Puntos de Interés')

    # Datos de ejemplo para puntos de interés (reemplazar con tus datos reales)
    poi_data = pd.DataFrame({
        'name': ['POI 1', 'POI 2', 'POI 3', 'POI 4'],
        'latitude': [40.4168, 40.4200, 40.4150, 40.4185],
        'longitude': [-3.7038, -3.7000, -3.7020, -3.7050]
    })

    st.write("Mostrando puntos de interés en el mapa:")
    st.map(poi_data[['latitude', 'longitude']], zoom=14)

    st.subheader("Mapa Avanzado con Pydeck")
    # Configuración de la vista inicial del mapa
    view_state = pdk.ViewState(
        latitude=poi_data['latitude'].mean(),
        longitude=poi_data['longitude'].mean(),
        zoom=13,
        pitch=50,
    )

    # Capa para los puntos de interés
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=poi_data,
        get_position='[longitude, latitude]',
        get_color='[200, 30, 0, 160]',
        get_radius=100,
        pickable=True
    )

    # Tooltip
    tooltip = {
        "html": "<b>{name}</b>",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }

    # Renderizar el mapa
    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip
    )
    st.pydeck_chart(deck)


# --- Pestaña de Predicción ---
with tab2:
    st.header('Predicción con Modelo XGBoost')

    if model is not None:
        st.write('Introduce los parámetros en la barra lateral y presiona "Predecir".')
        
        st.subheader('Parámetros de Entrada:')
        st.write(input_df)

        if st.button('Predecir'):
            # Asegúrate de que las columnas de input_df coincidan con las que el modelo espera
            # Es posible que necesites reordenar o renombrar columnas aquí.
            # Por ejemplo: `input_df = input_df[model.feature_names_in_]`
            
            prediction = model.predict(input_df)
            
            st.subheader('Resultado de la Predicción')
            st.success(f'La predicción del modelo es: {prediction[0]:.4f}')
    else:
        st.error('Error: No se pudo cargar el modelo. Asegúrate de que el archivo `models/xgb_tuned_model.joblib` exista.')