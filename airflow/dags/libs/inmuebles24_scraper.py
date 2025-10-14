import os
import re
import csv
import time
import unicodedata
from datetime import datetime

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException


def normalizar_nombre(nombre: str) -> str:
    nombre = nombre.lower()
    nombre = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode("utf-8")
    nombre = re.sub(r"[\s\W]+", "-", nombre)
    return nombre


def leer_municipios_csv(filepath: str):
    municipios = []
    try:
        with open(filepath, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if "D_mnpio" in row and row["D_mnpio"]:
                    municipios.append(row["D_mnpio"].strip())
    except FileNotFoundError:
        print(f"Error: El archivo '{filepath}' no se encontró.")
        return None
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo CSV: {e}")
        return None
    return sorted(list(set(municipios)))


def scrape_current_page(driver, nombre_municipio: str):
    inmuebles_en_pagina = []
    try:
        listings = driver.find_elements(By.CSS_SELECTOR, ".postingCardLayout-module__posting-card-layout")
        for listing in listings:
            inmueble = {}
            try:
                title_container = listing.find_element(By.CSS_SELECTOR, ".postingCard-module__posting-description")
                inmueble["Título"] = title_container.text.strip()
                title_element = title_container.find_element(By.TAG_NAME, "a")
                inmueble["Enlace"] = title_element.get_attribute("href")
            except Exception:
                inmueble["Título"] = "No disponible"
                inmueble["Enlace"] = "No disponible"

            try:
                price_element = listing.find_element(By.CSS_SELECTOR, 'div[data-qa="POSTING_CARD_PRICE"]')
                inmueble["Precio"] = price_element.text.strip()
            except Exception:
                inmueble["Precio"] = "No disponible"

            try:
                address_element = listing.find_element(By.CSS_SELECTOR, ".postingLocations-module__location-block")
                inmueble["Dirección"] = address_element.text.strip()
            except Exception:
                inmueble["Dirección"] = "No disponible"

            try:
                features = listing.find_elements(
                    By.CSS_SELECTOR, "span.postingMainFeatures-module__posting-main-features-listing"
                )
                features_list = [feature.text.strip() for feature in features]
                inmueble["Características"] = ", ".join(features_list)
            except Exception:
                inmueble["Características"] = "No disponible"

            inmueble["Municipio de Origen"] = nombre_municipio
            inmuebles_en_pagina.append(inmueble)
    except Exception as e:
        print(f"Error al extraer datos de la página: {e}")
    return inmuebles_en_pagina


def scrape_inmuebles24_delegaciones(
    csv_colonias_filename: str,
    url_base: str,
    output_directory_base: str,
    headless: bool = True,
    ds: str | None = None,
):
    # ds: 'YYYY-MM-DD' (si no se provee, usar fecha actual)
    if not ds:
        ds = datetime.now().strftime("%Y-%m-%d")
    output_directory = os.path.join(output_directory_base, f"por_municipio_{ds}")

    print("Leyendo la lista de municipios desde el archivo...")
    lista_municipios = leer_municipios_csv(csv_colonias_filename)
    if not lista_municipios:
        print("No se pudo obtener la lista de municipios. Terminando la ejecución.")
        return

    print(f"Se encontraron {len(lista_municipios)} municipios únicos para buscar.")

    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")

    try:
        print("\nIniciando el navegador web sin detección...")
        driver = uc.Chrome(options=options, timeout=240)
    except WebDriverException as e:
        print(f"Error al iniciar el navegador: {e}")
        raise

    municipios_sin_resultados = []

    try:
        os.makedirs(output_directory, exist_ok=True)

        for index, nombre_municipio in enumerate(lista_municipios):
            normalized_name = normalizar_nombre(nombre_municipio)
            municipio_filename = os.path.join(output_directory, f"{normalized_name}.csv")

            if os.path.exists(municipio_filename):
                print(
                    f"Archivo para '{nombre_municipio}' ya existe. Saltando este municipio. "
                    f"(van {index + 1} de {len(lista_municipios)}, faltan {len(lista_municipios) - (index + 1)})"
                )
                continue

            print(
                f"\n--- Buscando en municipio: {nombre_municipio} "
                f"(van {index + 1} de {len(lista_municipios)}, faltan {len(lista_municipios) - (index + 1)}) ---"
            )

            inmuebles_por_municipio = []
            url_to_visit = f"{url_base}{normalized_name}.html"

            while True:
                print(f"Buscando en la URL: {url_to_visit}")
                try:
                    driver.get(url_to_visit)
                    wait = WebDriverWait(driver, 15)

                    wait.until(
                        EC.any_of(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "div.postingsList-module__postings-container")),
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".emptyResults-module__empty-results")),
                        )
                    )

                    if len(driver.find_elements(By.CSS_SELECTOR, ".emptyResults-module__empty-results")) > 0:
                        print("La página no tiene resultados. Finalizando la paginación.")
                        break

                    current_page_data = scrape_current_page(driver, nombre_municipio)
                    if not current_page_data:
                        print("No se encontraron propiedades en la página. Finalizando la paginación.")
                        break

                    inmuebles_por_municipio.extend(current_page_data)
                    print(f"Página scrapeada con éxito. Total de propiedades hasta ahora: {len(inmuebles_por_municipio)}")

                    try:
                        next_button = driver.find_element(By.CSS_SELECTOR, 'a[data-qa="PAGING_NEXT"]')
                        url_to_visit = next_button.get_attribute("href")
                        time.sleep(2)
                    except NoSuchElementException:
                        print("Botón 'Siguiente' no encontrado. Fin de la paginación.")
                        break

                except TimeoutException:
                    print("Se superó el tiempo de espera. Probablemente no existen más páginas.")
                    break
                except Exception as e:
                    print(f"Ocurrió un error inesperado al procesar la URL '{url_to_visit}': {e}")
                    break

            if inmuebles_por_municipio:
                with open(municipio_filename, "w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = ["Título", "Enlace", "Precio", "Dirección", "Características", "Municipio de Origen"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(inmuebles_por_municipio)
                print(f"Datos de {len(inmuebles_por_municipio)} propiedades guardados en '{municipio_filename}'.")
            else:
                print(f"No se encontraron propiedades en ninguna página para el municipio '{nombre_municipio}'.")
                municipios_sin_resultados.append(nombre_municipio)

        if municipios_sin_resultados:
            print("\nLos siguientes municipios no arrojaron resultados:")
            for m in sorted(municipios_sin_resultados):
                print(f"- {m}")
        else:
            print("\n¡Todos los municipios procesados tuvieron al menos una propiedad!")
    finally:
        print("\nCerrando el navegador.")
        try:
            driver.quit()
        except Exception:
            pass
