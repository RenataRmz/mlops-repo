import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import json
import glob
import os
from sklearn.neighbors import BallTree
from geopy.geocoders import Nominatim
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from shapely.ops import transform
import unicodedata

# --- Inicialización de estado de sesión para resultados (AÑADIDO) ---
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'show_prediction_results' not in st.session_state:
    st.session_state.show_prediction_results = False
# --- FIN Inicialización de estado de sesión ---


# --- Construcción de rutas absolutas ---
# Obtener el directorio del script actual para construir rutas robustas
# /.../mlops-repo/reports/streamlit
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Subir dos niveles para llegar a la raíz del proyecto /.../mlops-repo
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../'))

# Construir rutas a los archivos desde la raíz del proyecto
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/xgboost_tuned_pipeline.joblib')
FEATURES_PATH = os.path.join(PROJECT_ROOT, 'data/processed/final_features.json')
POIS_PATH_PATTERN = os.path.join(PROJECT_ROOT, 'data/processed/csv/**/*.csv')
CENSUS_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/processed/INEGI/colonia')

# --- Configuración de Iconos para el Mapa ---
ICON_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/icon-atlas.png"
ICON_MAPPING = {
    "marker": {"x": 0, "y": 0, "width": 128, "height": 128, "mask": True},
    "school": {"x": 128, "y": 0, "width": 128, "height": 128, "mask": True},
    "hospital": {"x": 0, "y": 128, "width": 128, "height": 128, "mask": True},
    "park": {"x": 128, "y": 128, "width": 128, "height": 128, "mask": True},
    "bus": {"x": 0, "y": 192, "width": 64, "height": 64, "mask": True},
    "train": {"x": 64, "y": 192, "width": 64, "height": 64, "mask": True},
}
# Mapeo unificado para garantizar que todos los POIs se dibujen correctamente como marcadores
POI_ICON_MAP = {
    'escuelas_publicas': 'marker',
    'escuelas_privadas_con_coordenadas': 'marker',
    'hospitales_y_centros_de_salud_con_coordenadas': 'marker',
    'metrobus_estaciones_con_coordenadas': 'marker',
    'stc_metro_estaciones_utm14n_con_coordenadas': 'marker',
    'areas_verdes_filtrado': 'marker',
}
# Configuración de la página
st.set_page_config(layout="wide", page_title="Predictor de Precios de Renta")

# --- Carga del Modelo y Features ---
@st.cache_resource
def load_model():
    """Carga el modelo XGBoost tuneado."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se pudo cargar el modelo. Se buscó en: {MODEL_PATH}")
        return None

@st.cache_data
def load_features(path):
    """Carga la lista de características del modelo desde un archivo JSON."""
    try:
        with open(path, 'r') as f:
            features = json.load(f)
        return features
    except FileNotFoundError:
        st.error(f"Error: No se pudo encontrar el archivo de características en {path}")
        return None

@st.cache_data
def load_pois(path_pattern):
    """Carga puntos de interés desde archivos CSV."""
    poi_files = glob.glob(path_pattern, recursive=True)
    pois = {}
    for f in poi_files:
        if "lineas" in f:  # Excluir archivos no deseados
            continue
        try:
            key = os.path.splitext(os.path.basename(f))[0].lower()
            df = pd.read_csv(f)
            
            # Normalización y estandarización del nombre de las columnas
            df.columns = [col.lower() for col in df.columns]

            # 1. Renombrar y asegurar columnas de coordenadas
            if 'latitud' in df.columns and 'longitud' in df.columns:
                df = df.rename(columns={'latitud': 'latitude', 'longitud': 'longitude'})
                
                # 2. Renombrar y estandarizar la columna de nombre
                if 'nombre' in df.columns:
                    df = df.rename(columns={'nombre': 'name'})
                else:
                    # Si no hay columna 'nombre', crear una con el tipo de POI
                    df['name'] = key.replace('_', ' ').title()

                # Eliminar filas sin coordenadas
                df.dropna(subset=['latitude', 'longitude'], inplace=True)
                
                # SOLO guardar si el DataFrame resultante no está vacío
                if not df.empty:
                     pois[key] = df
            else:
                 pass

        except Exception as e:
            st.warning(f"No se pudo cargar o procesar el archivo {f}: {e}")
    return pois

@st.cache_data
def load_census_data(municipality, path_pattern):
    """Carga los datos del censo para un municipio específico y los transforma a EPSG:4326."""
    try:
        # Estandarizar nombre del municipio para buscar el archivo
        municipality_std = unicodedata.normalize('NFKD', municipality).encode('ascii', 'ignore').decode('utf-8').lower().replace(' ', '_')
        
        search_path = os.path.join(path_pattern, f"*_{municipality_std}.csv")
        file_list = glob.glob(search_path)
        
        if not file_list:
            st.warning(f"No se encontró archivo de censo para '{municipality}'.")
            return None

        df = pd.read_csv(file_list[0])

        # Transformar coordenadas de EPSG:6372 a EPSG:4326
        if {'lon', 'lat'}.issubset(df.columns):
            df[['lon', 'lat']] = df[['lon', 'lat']].astype(float)
            transformer = Transformer.from_crs("epsg:6372", "epsg:4326", always_xy=True)
            lon_wgs, lat_wgs = transformer.transform(df['lon'].values, df['lat'].values)
            
            # Crear GeoDataFrame
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(lon_wgs, lat_wgs), crs="EPSG:4326"
            )
            return gdf
        else:
            st.error("El archivo del censo no contiene las columnas 'lon' y 'lat' requeridas.")
            return None
            
    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo de datos del censo: {e}")
        return None

@st.cache_resource
def build_poi_indices(_pois_data):
    """Construye índices BallTree para búsquedas espaciales rápidas en los datos de POI."""
    poi_indices = {}
    R_EARTH_M = 6_371_000.0  # Radio de la Tierra en metros

    for key, df in _pois_data.items():
        if not df.empty:
            # Convertir coordenadas a radianes para haversine
            coords_rad = np.radians(df[['latitude', 'longitude']].astype(float).values)
            tree = BallTree(coords_rad, metric="haversine")
            poi_indices[key] = {'tree': tree, 'R_EARTH_M': R_EARTH_M}
    return poi_indices

@st.cache_resource
def build_census_index(_census_gdf):
    """Construye un BallTree para los datos del censo de un municipio."""
    if _census_gdf is None or _census_gdf.empty:
        return None
    
    # Convertir coordenadas del censo a radianes para BallTree
    coords_list = _census_gdf.apply(lambda p: (p.geometry.y, p.geometry.x), axis=1).tolist()
    census_coords_rad = np.radians(coords_list)
    tree = BallTree(census_coords_rad, metric="haversine")
    return tree

model = load_model()
# La ruta es relativa al script de streamlit
final_features = load_features(FEATURES_PATH)
# Cargar todos los CSV de puntos de interés
pois_data = load_pois(POIS_PATH_PATTERN)
# NO cargamos datos del censo aquí, se hará dinámicamente
# Construir los índices para los POIs cargados
poi_indices = build_poi_indices(pois_data)


# --- Inicialización de geocoder y estado de sesión ---
@st.cache_resource
def get_geolocator():
    """Inicializa el geolocator de Nominatim."""
    return Nominatim(user_agent="mlops_rent_predictor")

geolocator = get_geolocator()

# Inicializar estado de sesión para latitud y longitud si no existen
if 'lat' not in st.session_state:
    st.session_state.lat = 19.4326
if 'lon' not in st.session_state:
    st.session_state.lon = -99.1332


# --- Barra Lateral para Entradas de Predicción ---
st.sidebar.header('Parámetros del Inmueble para Predicción')

def user_input_features():
    """Recopila las características de entrada del usuario y calcula las distancias a los POIs."""
    address = st.sidebar.text_input('Dirección (ej: Palacio de Bellas Artes, CDMX)')
    
    if st.sidebar.button('Buscar Dirección'):
        with st.spinner('Geocodificando dirección...'):
            try:
                # Añadir "Mexico City, Mexico" para mejorar la precisión
                full_address = f"{address}, Mexico City, Mexico"
                location = geolocator.geocode(full_address)
                if location:
                    st.session_state.lat = location.latitude
                    st.session_state.lon = location.longitude
                    st.sidebar.success(f"Dirección encontrada: {location.address}")
                else:
                    st.sidebar.error("No se pudo encontrar la dirección. Por favor, intente de nuevo o ajuste la latitud/longitud manualmente.")
            except Exception as e:
                st.sidebar.error(f"Error de geocodificación: {e}")

    st.sidebar.info('La latitud y longitud se actualizan con la búsqueda de dirección. También puede ajustarlas manualmente.')

    lat = st.sidebar.number_input('Latitud', -90.0, 90.0, st.session_state.lat, format="%.4f", key="lat_input")
    lon = st.sidebar.number_input('Longitud', -180.0, 180.0, st.session_state.lon, format="%.4f", key="lon_input")
    
    # Actualizar el estado de la sesión si el usuario cambia los números manualmente
    st.session_state.lat = lat
    st.session_state.lon = lon

    # Selección de municipio para datos de censo
    alcaldias_cdmx = [
        "Alvaro Obregon", "Azcapotzalco", "Benito Juarez", "Coyoacan", 
        "Cuajimalpa de Morelos", "Cuauhtemoc", "Gustavo A. Madero", "Iztacalco", 
        "Iztapalapa", "La Magdalena Contreras", "Miguel Hidalgo", "Milpa Alta", 
        "Tlahuac", "Tlalpan", "Venustiano Carranza", "Xochimilco"
    ]
    municipality = st.sidebar.selectbox("Seleccione la Alcaldía", alcaldias_cdmx, index=2) # Default a Benito Juarez

    # Características que el usuario puede introducir directamente
    recamaras = st.sidebar.slider('Recámaras', 0, 10, 2)
    estacionamiento = st.sidebar.slider('Estacionamientos', 0, 5, 1)
    lote_m2 = st.sidebar.slider('Superficie del Lote (m²)', 20, 1000, 150)
    es_amueblado = st.sidebar.checkbox('¿Es amueblado?', value=False)
    es_penthouse = st.sidebar.checkbox('¿Es penthouse?', value=False)

    # --- Cálculo dinámico de distancias a POIs ---
    # Convertir la coordenada del inmueble a radianes
    prop_coords_rad = np.radians(np.array([[lat, lon]]))
    
    distance_features = {}
    # Mapeo de claves de POI a nombres de feature de distancia esperados por el modelo
    poi_to_feature_map = {
        "escuelas_publicas": "dist_m_escuelas_publicas",
        "hospitales_y_centros_de_salud_con_coordenadas": "dist_m_hospitales_y_centros_de_salud_con_coordenadas",
        "areas_verdes_filtrado": "dist_m_areas_verdes_filtrado",
        "metrobus_estaciones_con_coordenadas": "dist_m_metrobus_estaciones_con_coordenadas",
        "stc_metro_estaciones_utm14n_con_coordenadas": "dist_m_stc_metro_estaciones_utm14n_con_coordenadas",
        "escuelas_privadas_con_coordenadas": "dist_m_escuelas_privadas_con_coordenadas"
    }

    for poi_key, feature_name in poi_to_feature_map.items():
        if poi_key in poi_indices:
            tree_info = poi_indices[poi_key]
            dist_rad, _ = tree_info['tree'].query(prop_coords_rad, k=1)
            dist_m = dist_rad[0][0] * tree_info['R_EARTH_M']
            distance_features[feature_name] = dist_m
        else:
            # Si no hay datos para un POI, usar un valor grande o NaN
            distance_features[feature_name] = np.nan 

    # --- Búsqueda del punto de censo más cercano ---
    census_features = {}
    census_gdf = load_census_data(municipality, CENSUS_DATA_PATH)
    census_tree = build_census_index(census_gdf)

    if census_gdf is not None and census_tree is not None:
        # Buscar el punto más cercano usando el árbol cacheado
        dist_rad, ind = census_tree.query(prop_coords_rad, k=1)
        nearest_idx = ind[0][0]
        
        # Extraer características del punto más cercano
        nearest_census_point = census_gdf.iloc[nearest_idx]
        census_cols = ["P_6A11", "PRO_OCUP_C", "VPH_SINRTV", "VPH_1CUART", "VIVPAR_UT", "VPH_NODREN", "VPH_AGUAFV", "VPH_SINTIC"]
        for col in census_cols:
            if col in nearest_census_point:
                value = nearest_census_point[col]
                # Limpiar el valor: reemplazar '*' y convertir a numérico
                if isinstance(value, str):
                    value = value.replace('*', '0')
                
                # Forzar la conversión a numérico, usando 0.0 si falla
                numeric_value = pd.to_numeric(value, errors='coerce')
                census_features[col] = 0.0 if pd.isna(numeric_value) else numeric_value
            else:
                census_features[col] = 0.0 # Usar 0.0 si la columna no existe
        
        if any(pd.isna(list(census_features.values()))):
             st.warning("Algunas características del censo no se encontraron en el punto más cercano.")

    else:
        st.warning("No se pudieron cargar los datos del censo para la alcaldía seleccionada. Usando valores por defecto.")
        # Usar valores por defecto si no hay datos
        census_features = {
            "P_6A11": 0.05, "PRO_OCUP_C": 3.5, "VPH_SINRTV": 0.1, "VPH_1CUART": 0.08,
            "VIVPAR_UT": 0.95, "VPH_NODREN": 0.02, "VPH_AGUAFV": 0.01, "VPH_SINTIC": 0.15
        }

    data = {
        "recamaras": recamaras,
        "estacionamiento": estacionamiento,
        "lote_m2": lote_m2,
        "es_amueblado": 1 if es_amueblado else 0,
        "es_penthouse": 1 if es_penthouse else 0,
        **census_features,
        **distance_features
    }
    
    # Crear el DataFrame y asegurarse de que el orden de las columnas coincida con final_features
    if final_features:
        features_df = pd.DataFrame(data, index=[0])
        # Añadir columnas faltantes con 0 si es necesario (aunque no debería pasar con el dict)
        for col in final_features:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[final_features] # Reordenar columnas
    else:
        features_df = pd.DataFrame()

    return features_df, lat, lon, distance_features, census_features

st.sidebar.header('Visualización de Mapa')

# --- INSERCIÓN DE VERIFICACIÓN DE POIS CARGADOS (Limpio) ---
loaded_poi_keys = [key.replace('_', ' ').title() for key, df in pois_data.items() if not df.empty]
if loaded_poi_keys:
    st.sidebar.caption(f"**POIs cargados y listos:** {', '.join(loaded_poi_keys)}")
# --- FIN DE INSERCIÓN ---


show_pois = {}
if pois_data:
    st.sidebar.write("Mostrar Puntos de Interés:")
    for key in pois_data.keys():
        # Solo mostrar el checkbox si el DataFrame NO está vacío
        if not pois_data[key].empty:
             # Se eliminan los miles de separadores de puntos del texto del checkbox
             show_pois[key] = st.sidebar.checkbox(f"{key.replace('_', ' ').title()} ({len(pois_data[key])} puntos)", value=False)

if final_features:
    input_df, lat, lon, distance_features, census_features = user_input_features()
else:
    st.stop()

# --- Contenido Principal ---
st.title('Plataforma de Sugerencia de Precio de Renta')

tab1, tab2 = st.tabs(["Ubicación del Inmueble", "Predicción de Precio"])

# --- Pestaña de Visualización de Mapa ---
with tab1:
    st.header('Mapa de Ubicación y Puntos de Interés')

    # Datos para el punto del inmueble introducido
    property_location_data = pd.DataFrame({
        'name': ['Inmueble Introducido'],
        'latitude': [lat],
        'longitude': [lon],
        'icon_id': ['marker'],
        'fill_color': [[255, 30, 0, 255]] # Rojo para el relleno
    })

    st.write("Mostrando la ubicación del inmueble y puntos de interés seleccionados.")
    
    # Configuración de la vista inicial del mapa centrada en el inmueble
    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=14,
        pitch=50,
    )

    # Capa para el punto del inmueble (ICONLAYER)
    property_layer = pdk.Layer(
        'IconLayer',
        data=property_location_data,
        get_position='[longitude, latitude]',
        get_icon='icon_id',
        get_size=5,
        size_scale=7, # Tamaño intermedio (POIs usan 5)
        get_color='fill_color',
        pickable=True,
        icon_atlas=ICON_URL,
        icon_mapping=ICON_MAPPING,
    )

    # Tooltip para el inmueble y los POIs
    tooltip = {
        "html": "<b>{name}</b>",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }

    layers = [property_layer] # Inmueble introducido

    # Colores para las capas de POIs (relleno)
    poi_colors = {
        'escuelas_publicas': [30, 144, 255, 255], # DodgerBlue (Azul)
        'escuelas_privadas_con_coordenadas': [255, 165, 0, 255], # Orange (Naranja)
        'hospitales_y_centros_de_salud_con_coordenadas': [220, 20, 60, 255], # Crimson (Rojo Oscuro)
        'metrobus_estaciones_con_coordenadas': [50, 205, 50, 255], # LimeGreen (Verde Brillante)
        'stc_metro_estaciones_utm14n_con_coordenadas': [255, 105, 180, 255], # HotPink (Rosa)
        'areas_verdes_filtrado': [34, 139, 34, 255], # ForestGreen (Verde Bosque)
    }

    # --- Lógica para la capa de Iconos (POIs) ---
    visible_pois_dfs = []
    for key, show in show_pois.items():
        if show and key in pois_data and not pois_data[key].empty:
            poi_df = pois_data[key].copy()
            
            # Asignar el ID del ícono
            poi_df['icon_id'] = POI_ICON_MAP.get(key, 'marker')
            
            # Obtener el color de relleno
            fill_color = poi_colors.get(key, [128, 128, 128, 255])
            
            # Asignar el color de relleno a la columna 'fill_color'
            poi_df['fill_color'] = [fill_color] * len(poi_df)
            
            visible_pois_dfs.append(poi_df)

    # Si hay POIs para mostrar, crear la IconLayer
    if visible_pois_dfs:
        combined_pois_df = pd.concat(visible_pois_dfs, ignore_index=True)
        
        icon_layer = pdk.Layer(
            'IconLayer',
            data=combined_pois_df,
            get_icon='icon_id',
            get_size=5,
            size_scale=5,
            get_position='[longitude, latitude]',
            get_color='fill_color',
            pickable=True,
            icon_atlas=ICON_URL,
            icon_mapping=ICON_MAPPING,
        )
        layers.append(icon_layer)
    # --- FIN Lógica IconLayer ---


    # Renderizar el mapa
    deck = pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=view_state,
        layers=layers,
        tooltip=tooltip
    )
    st.pydeck_chart(deck)


# --- Pestaña de Predicción ---
with tab2:
    st.header('Predicción de Precio de Renta')

    if model is not None:
        st.write('Los parámetros introducidos para el inmueble son:')
        
        # Mostrar solo las características que el usuario introdujo
        user_inputs = {
            'recamaras': input_df['recamaras'].iloc[0],
            'estacionamiento': input_df['estacionamiento'].iloc[0],
            'lote_m2': input_df['lote_m2'].iloc[0],
            'es_amueblado': 'Sí' if input_df['es_amueblado'].iloc[0] == 1 else 'No',
            'es_penthouse': 'Sí' if input_df['es_penthouse'].iloc[0] == 1 else 'No'
        }
        st.json(user_inputs)
        
        st.info("Las características de contexto (distancias, censo) se calculan dinámicamente desde la ubicación del inmueble.")

        # Mostrar distancias calculadas
        st.write("Distancias calculadas a Puntos de Interés (en metros):")
        dist_view = {
            key.replace('dist_m_', '').replace('_', ' ').title(): f"{value:,.0f} m"
            for key, value in distance_features.items()
        }
        st.json(dist_view)

        # Mostrar características del censo
        st.write("Características del Censo (punto más cercano):")
        st.json(census_features)

        if st.button('Predecir Precio de Renta'):
            try:
                # El DataFrame 'input_df' ya tiene las columnas en el orden correcto
                prediction = model.predict(input_df)
                
                # --- ALMACENAR RESULTADO EN SESSION STATE (MODIFICADO) ---
                st.session_state.prediction_result = prediction[0]
                st.session_state.show_prediction_results = True
                # --- FIN ALMACENAR RESULTADO ---

            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")
                st.warning("Asegúrate de que las características de entrada coincidan con las que el modelo fue entrenado.")
                st.session_state.show_prediction_results = False # Reiniciar en caso de error
        
        # --- MOSTRAR RESULTADO PERSISTENTE (AÑADIDO) ---
        if st.session_state.show_prediction_results and st.session_state.prediction_result is not None:
            st.subheader('Resultado de la Predicción')
            st.success(f'El precio de renta sugerido es: ${st.session_state.prediction_result:,.2f}')
        # --- FIN MOSTRAR RESULTADO PERSISTENTE ---

    else:
        st.error('Error: No se pudo cargar el modelo. Asegúrate de que el archivo `models/xgb_tuned_model.joblib` exista.')
