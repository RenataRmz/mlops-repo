from datetime import timedelta
import os
import re
import csv
import time
import unicodedata

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
#from airflow.utils.context import get_current_context
from airflow.exceptions import AirflowSkipException

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException

from libs.inmuebles24_scraper import scrape_inmuebles24_delegaciones
from libs.inmuebles24_transform import transform_inmuebles24
from libs.inmuebles24_geocode import geocode_inmuebles24

# --- Configuración por defecto del DAG ---
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# --- Definición del DAG y tarea ---
with DAG(
    dag_id="webscrapping_inmuebles24_delegacion",
    description="Primer paso: scraping de Inmuebles24 por delegación (CSV por municipio)",
    default_args=default_args,
    start_date=days_ago(1),
    schedule=None,  # Ajusta a '0 3 * * *' si deseas ejecución diaria
    catchup=False,
    tags=["webscraping", "inmuebles24", "delegaciones"],
) as dag:
    scrape_delegaciones = PythonOperator(
        task_id="scrape_delegaciones",
        python_callable=scrape_inmuebles24_delegaciones,
        op_kwargs={
            "csv_colonias_filename": "/Users/renataramirez/Documents/mlops-repo/data/raw/CP/CPdescarga - Distrito_Federal.csv",
            "url_base": "https://www.inmuebles24.com/departamentos-en-renta-en-",
            "output_directory_base": "/Users/renataramirez/Documents/mlops-repo/data/raw/webscrapping",
            "headless": True,
            "ds": "{{ ds }}",
        },
    )

    transform_inmuebles = PythonOperator(
        task_id="transform_inmuebles24",
        python_callable=transform_inmuebles24,
        op_kwargs={
            "raw_base_dir": "/Users/renataramirez/Documents/mlops-repo/data/raw/webscrapping",
            "cp_csv_path": "/Users/renataramirez/Documents/mlops-repo/data/raw/CP/CPdescarga - Distrito_Federal.csv",
            "processed_dir": "/Users/renataramirez/Documents/mlops-repo/data/processed",
            "ds": "{{ ds }}",
            "enable_geocode": False,
        },
    )

    geocode_inmuebles = PythonOperator(
        task_id="geocode_inmuebles24",
        python_callable=geocode_inmuebles24,
        op_kwargs={
            "processed_dir": "/Users/renataramirez/Documents/mlops-repo/data/processed",
            "ds": "{{ ds }}",
            "user_agent": "airflow_inmuebles24_geocode",
            "throttle_seconds": 1.0,
            "max_retries": 2,
        },
    )

    scrape_delegaciones >> transform_inmuebles >> geocode_inmuebles
