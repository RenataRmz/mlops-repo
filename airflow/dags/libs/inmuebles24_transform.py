import os
import re
import glob
import unicodedata
from datetime import datetime

import pandas as pd
import numpy as np

try:
    import unidecode
except ImportError:
    # Fallback mínimo si no existe el paquete unidecode
    unidecode = None


def _estandarizar_texto(texto: str) -> str:
    if pd.isna(texto):
        return ""
    s = str(texto).replace("\n", " ").replace(",", " ").strip().lower()
    if unidecode:
        s = unidecode.unidecode(s)
    else:
        # Fallback para quitar acentos si no hay unidecode
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    return s


def _limpiar_y_convertir_precio(precio_str: str, tipo_cambio_usd_mxn: float = 18.66) -> float | None:
    if pd.isna(precio_str) or not isinstance(precio_str, str):
        return None
    s = precio_str.strip()
    if s.startswith("USD"):
        v = s.replace("USD ", "").replace(",", "")
        try:
            return int(float(v) * tipo_cambio_usd_mxn)
        except ValueError:
            return None
    if s.startswith("MN"):
        v = s.replace("MN ", "").replace(",", "")
        try:
            return int(float(v))
        except ValueError:
            return None
    return None


def _extraer_int(regex: str, text: str) -> pd.Series:
    m = pd.Series([None])
    try:
        v = pd.Series([text]).str.extract(regex, flags=re.IGNORECASE)
        return v[0]
    except Exception:
        return m


def _mapear_colonia_cp(df: pd.DataFrame, cp_csv_path: str) -> pd.DataFrame:
    df_cp = pd.read_csv(cp_csv_path)
    # Filtramos por tipos deseados
    df_cp = df_cp[df_cp["d_tipo_asenta"].isin(["Colonia", "Barrio", "Pueblo"])].copy()
    df_cp["d_asenta_std"] = df_cp["d_asenta"].apply(_estandarizar_texto)
    df_cp["d_codigo"] = df_cp["d_codigo"].astype(str).apply(_estandarizar_texto)

    # Búsqueda simple por substring (como en el cuaderno)
    def buscar(direccion_std: str):
        for _, row in df_cp.iterrows():
            if row["d_asenta_std"] and row["d_asenta_std"] in direccion_std:
                return row["d_asenta_std"], row["d_codigo"]
        return None, None

    df[["colonia", "cp"]] = df["direccion"].apply(lambda x: pd.Series(buscar(x)))
    return df


def transform_inmuebles24(
    raw_base_dir: str,
    cp_csv_path: str,
    processed_dir: str,
    ds: str | None = None,
    enable_geocode: bool = False,  # deshabilitado por defecto
) -> str:
    # ds: 'YYYY-MM-DD'
    if not ds:
        ds = datetime.now().strftime("%Y-%m-%d")

    input_dir = os.path.join(raw_base_dir, f"por_municipio_{ds}")
    output_path = os.path.join(processed_dir, f"inmuebles24_departamentos_{ds}.parquet")
    os.makedirs(processed_dir, exist_ok=True)

    archivos_csv = glob.glob(os.path.join(input_dir, "*.csv"))
    if not archivos_csv:
        print(f"No se encontraron CSV en {input_dir}")
        return output_path

    # Leer y unir
    dfs = []
    for p in archivos_csv:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"Error leyendo {p}: {e}")
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if df.empty:
        print("No hay datos para procesar.")
        return output_path

    # Deduplicado
    df = df.drop_duplicates().copy()

    # Precio
    df["Precio"] = df["Precio"].astype(str).str.strip()
    df["precio_mxn"] = df["Precio"].apply(_limpiar_y_convertir_precio)

    # Correcciones puntuales detectadas en el cuaderno (si aparecen)
    fixes = {
        "MN 3,849,092": 1 / 100.0,
        "MN 19,250,000": 1 / 1000.0,
    }
    for k, factor in fixes.items():
        m = df["Precio"] == k
        if m.any():
            try:
                df.loc[m, "precio_mxn"] = (
                    df.loc[m, "Precio"].str.replace("MN ", "").str.replace(",", "").astype(float) * factor
                )
            except Exception:
                pass

    # Filtrar locales comerciales
    df["es_local_comercial"] = df["Título"].str.contains("local comercial", case=False, na=False).astype(int)
    df = df[df.es_local_comercial == 0].drop(columns=["es_local_comercial"])

    # Limpiar leyenda del título (texto largo)
    leyenda = "*Aviso de privacidad sugerencias, quejas, aclaraciones, consultar nuestra página. El precio no incluye el mobiliario, electrodomésticos, artículos de decoración y/o arte que pudieran aparecer en las fotografías. El precio de venta no incluye gastos e impuestos de escrituración, o gastos derivados por algún tipo de credito."
    df["Título"] = df["Título"].astype(str).str.replace(leyenda, "", regex=False).str.strip()

    # Extracción básica de características
    # lote m2
    df["lote_m2"] = (
        df["Características"].astype(str).str.extract(r"(\d+)\s*m²\s*lote", flags=re.IGNORECASE)[0].astype("Int64")
    )
    # recamaras
    df["recamaras"] = (
        df["Características"].astype(str).str.extract(r"(\d+)\s*rec\.?", flags=re.IGNORECASE)[0].astype("Int64")
    )
    # baños
    df["baños"] = (
        df["Características"].astype(str).str.extract(r"(\d+)\s*baño\s?", flags=re.IGNORECASE)[0].astype("Int64")
    )
    # estacionamiento
    df["estacionamiento"] = (
        df["Características"].astype(str).str.extract(r"(\d+)\s*estac\.?", flags=re.IGNORECASE)[0].astype("Int64")
    )

    # Complementos desde Título
    df["estacionamiento_txt"] = df["Título"].str.contains("estacionamiento", case=False, na=False).astype(int)
    df["estacionamiento"] = np.where(
        df["estacionamiento"].notna(),
        df["estacionamiento"],
        np.where((df["estacionamiento"].isna()) & (df["estacionamiento_txt"] == 1), 1, 0),
    )

    df["recamaras_txt"] = df["Título"].str.contains("recamaras", case=False, na=False).astype(int)
    df["recamaras"] = np.where(
        df["recamaras"].notna(),
        df["recamaras"],
        np.where((df["recamaras"].isna()) & (df["recamaras_txt"] == 1), 1, 0),
    )

    df["baños_txt"] = df["Título"].str.contains("baño", case=False, na=False).astype(int)
    df["baños"] = np.where(
        df["baños"].notna(),
        df["baños"],
        np.where((df["baños"].isna()) & (df["baños_txt"] == 1), 1, 0),
    )

    # Flags adicionales
    df["en_renta"] = df["Título"].str.contains(r" renta ", case=False, na=False).astype(int)
    df["en_venta"] = df["Título"].str.contains(r" venta |se vende|preventa|venta departamento", case=False, na=False).astype(int)
    df["es_amueblado"] = df["Título"].str.contains("amueblado", case=False, na=False).astype(int)
    df["es_penthouse"] = df["Título"].str.contains("penthouse| ph |pentgarden|departamento de lujo", case=False, na=False).astype(int)

    # Quedarnos con rentas y precios razonables
    df = df[((df["en_venta"] == 0) & (df["precio_mxn"] > 1) & (df["precio_mxn"] < 1_000_000))].copy()

    # Estandarizar dirección y municipio
    df["direccion"] = df["Dirección"].apply(_estandarizar_texto)
    df["municipio"] = df["Municipio de Origen"].apply(_estandarizar_texto)

    # Normalizaciones puntuales
    df["direccion"] = df["direccion"].str.replace("santa maria maninalco", "santa maria malinalco")
    df["direccion"] = df["direccion"].str.replace("country club", "churubusco country club")

    # Mapear colonia y CP usando el CSV
    df = _mapear_colonia_cp(df, cp_csv_path)

    # Opcional: geocoding (deshabilitado por defecto en Airflow)
    if enable_geocode:
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="airflow_inmuebles24_transform")
            def completar_colonia_cp(direccion):
                try:
                    loc = geolocator.geocode(direccion)
                    if loc:
                        det = geolocator.reverse((loc.latitude, loc.longitude), language="es")
                        address = det.raw.get("address", {})
                        colonia = address.get("neighbourhood", None)
                        codigo_postal = address.get("postcode", None)
                        return colonia, codigo_postal
                except Exception:
                    return None, None
                return None, None

            faltantes = df[df["colonia"].isna()].index
            for idx in faltantes:
                col, cp = completar_colonia_cp(df.at[idx, "direccion"])
                if col:
                    df.at[idx, "colonia"] = str(col).replace("colonia ", "").replace("Colonia ", "")
                if cp:
                    df.at[idx, "cp"] = cp
        except Exception as e:
            print(f"Geocoding deshabilitado por error: {e}")

    # Columnas finales
    important_columns = [
        "precio_mxn", "lote_m2", "recamaras", "baños",
        "estacionamiento", "es_amueblado", "es_penthouse",
        "direccion", "colonia", "cp", "municipio",
        "Título", "Enlace",
    ]
    df_final = df[df["colonia"].notna()][important_columns].copy()

    # Guardar Parquet
    df_final.to_parquet(output_path, index=False)
    print(f"Parquet generado: {output_path} (filas: {len(df_final)})")
    return output_path
