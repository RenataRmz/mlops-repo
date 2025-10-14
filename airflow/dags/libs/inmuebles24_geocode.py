import os
import time
from datetime import datetime
from typing import Dict, Tuple, Optional

import pandas as pd

def _geocode_unique_addresses(
    addresses: list[str],
    user_agent: str = "airflow_inmuebles24_geocode",
    throttle_seconds: float = 1.0,
    max_retries: int = 2,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeopyError
    except Exception as e:
        raise RuntimeError(
            "geopy no está instalado. Instala con `pip install geopy` en tu imagen/entorno de Airflow."
        ) from e

    geolocator = Nominatim(user_agent=user_agent, timeout=10)
    results: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    for addr in addresses:
        lat, lon = None, None
        for attempt in range(max_retries + 1):
            try:
                if not addr:
                    break
                loc = geolocator.geocode(addr)
                if loc:
                    lat, lon = float(loc.latitude), float(loc.longitude)
                break
            except (GeocoderTimedOut, GeopyError, Exception):
                if attempt < max_retries:
                    time.sleep(throttle_seconds * (attempt + 1))
                    continue
                else:
                    break
            finally:
                # Throttle entre solicitudes
                time.sleep(throttle_seconds)
        results[addr] = (lat, lon)
    return results


def geocode_inmuebles24(
    processed_dir: str,
    ds: str | None = None,
    user_agent: str = "airflow_inmuebles24_geocode",
    throttle_seconds: float = 1.0,
    max_retries: int = 2,
) -> str:
    """
    Lee el parquet de transform_inmuebles24 y genera un parquet enriquecido con latitud/longitud.
    Entrada: {processed_dir}/inmuebles24_departamentos_{ds}.parquet
    Salida:  {processed_dir}/inmuebles24_departamentos_coordenadas_{ds}.parquet
    """
    if not ds:
        ds = datetime.now().strftime("%Y-%m-%d")

    input_path = os.path.join(processed_dir, f"inmuebles24_departamentos_{ds}.parquet")
    output_path = os.path.join(processed_dir, f"inmuebles24_departamentos_coordenadas_{ds}.parquet")

    if not os.path.exists(input_path):
        print(f"No existe el parquet de entrada: {input_path}")
        return output_path

    df = pd.read_parquet(input_path)
    if df.empty:
        print("El parquet de entrada no contiene filas. Nada por geocodificar.")
        df.to_parquet(output_path, index=False)
        return output_path

    # Tomar direcciones únicas válidas
    addrs = (
        df["direccion"].dropna().astype(str).str.strip().replace({"": None}).dropna().unique().tolist()
        if "direccion" in df.columns
        else []
    )
    if not addrs:
        print("No se encontraron direcciones para geocodificar.")
        df.assign(latitud=None, longitud=None).to_parquet(output_path, index=False)
        return output_path

    print(f"Geocodificando {len(addrs)} direcciones únicas con Nominatim...")
    mapping = _geocode_unique_addresses(
        addrs,
        user_agent=user_agent,
        throttle_seconds=throttle_seconds,
        max_retries=max_retries,
    )

    geo_df = pd.DataFrame.from_dict(mapping, orient="index", columns=["latitud", "longitud"]).reset_index()
    geo_df = geo_df.rename(columns={"index": "direccion"})

    # Merge a nivel de dirección
    out = df.merge(geo_df, on="direccion", how="left")
    out.to_parquet(output_path, index=False)
    ok = out["latitud"].notna().sum()
    print(f"Parquet con coordenadas: {output_path} (lat/long pobladas: {ok} de {len(out)})")
    return output_path
