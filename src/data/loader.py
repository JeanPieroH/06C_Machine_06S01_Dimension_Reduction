# -*- coding: utf-8 -*-
"""
Módulo para la carga y procesamiento inicial del dataset Fashion MNIST.

Funciones:
- ejecutar_pipeline_carga: Orquesta la descarga, guardado y procesamiento básico.
- guardar_conjunto_procesado: Guarda conjuntos de datos procesados en disco.
- cargar_conjunto_procesado: Carga un conjunto de datos previamente procesado.
"""
import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
import pkg_resources

# Semilla para reproducibilidad
SEMILLA = 42
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)

def _obtener_versiones_paquetes():
    """Obtiene las versiones de los paquetes clave."""
    paquetes = ['numpy', 'tensorflow', 'scikit-learn']
    versiones = {}
    for paquete in paquetes:
        try:
            versiones[paquete] = pkg_resources.get_distribution(paquete).version
        except pkg_resources.DistributionNotFound:
            versiones[paquete] = "No encontrado"
    return versiones

def guardar_conjunto_procesado(x_train, x_test, y_train, y_test, dir_salida, nombre_normalizador, meta=None):
    """
    Guarda los arrays de datos procesados en un subdirectorio específico.

    Args:
        x_train (np.ndarray): Datos de entrenamiento.
        x_test (np.ndarray): Datos de prueba.
        y_train (np.ndarray): Etiquetas de entrenamiento.
        y_test (np.ndarray): Etiquetas de prueba.
        dir_salida (str): Directorio base para los datos procesados.
        nombre_normalizador (str): Nombre del normalizador usado para crear el subdirectorio.
        meta (dict, optional): Metadatos adicionales para guardar. Defaults to None.

    Returns:
        str: Ruta al directorio donde se guardaron los datos.
    """
    ruta_version = os.path.join(dir_salida, 'normalizations', nombre_normalizador)
    os.makedirs(ruta_version, exist_ok=True)

    np.save(os.path.join(ruta_version, 'X_train.npy'), x_train)
    np.save(os.path.join(ruta_version, 'X_test.npy'), x_test)
    np.save(os.path.join(ruta_version, 'y_train.npy'), y_train)
    np.save(os.path.join(ruta_version, 'y_test.npy'), y_test)

    if meta:
        with open(os.path.join(ruta_version, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=4)

    print(f"Datos guardados en: {ruta_version}")
    return ruta_version

def ejecutar_pipeline_carga(descargar=True, dir_crudos="data/raw/", dir_procesados="data/processed/", sobrescribir=False):
    """
    Ejecuta el pipeline completo de carga de datos para Fashion MNIST.

    Descarga los datos, los guarda en formato raw, crea metadatos y
    genera una versión procesada inicial (normalización MinMax [0,1]).

    Args:
        descargar (bool): Si es True, descarga los datos.
        dir_crudos (str): Directorio para guardar los datos crudos.
        dir_procesados (str): Directorio para guardar los datos procesados.
        sobrescribir (bool): Si es True, sobrescribe los archivos existentes.

    Returns:
        tuple: Tupla con (rutas_crudo, ruta_procesado_baseline, metadatos).
    """
    os.makedirs(dir_crudos, exist_ok=True)
    os.makedirs(dir_procesados, exist_ok=True)

    rutas_crudos = {
        "X_train": os.path.join(dir_crudos, 'X_train_raw.npy'),
        "y_train": os.path.join(dir_crudos, 'y_train_raw.npy'),
        "X_test": os.path.join(dir_crudos, 'X_test_raw.npy'),
        "y_test": os.path.join(dir_crudos, 'y_test_raw.npy'),
        "metadata": os.path.join(dir_crudos, 'metadata.json')
    }

    if not sobrescribir and all(os.path.exists(p) for p in rutas_crudos.values()):
        print("Los archivos crudos ya existen. Saltando descarga y guardado.")
        with open(rutas_crudos["metadata"], 'r') as f:
            metadatos = json.load(f)

        ruta_baseline = os.path.join(dir_procesados, 'normalizations', 'minmax')
        if os.path.exists(ruta_baseline):
             print(f"La versión procesada 'minmax' ya existe en {ruta_baseline}")
        else:
            print("Generando versión procesada 'minmax' desde archivos raw...")
            x_train_crudo = np.load(rutas_crudos["X_train"])
            x_test_crudo = np.load(rutas_crudos["X_test"])
            y_train_crudo = np.load(rutas_crudos["y_train"])
            y_test_crudo = np.load(rutas_crudos["y_test"])

            x_train_norm = x_train_crudo.astype('float32') / 255.0
            x_test_norm = x_test_crudo.astype('float32') / 255.0

            ruta_baseline = guardar_conjunto_procesado(
                x_train_norm, x_test_norm, y_train_crudo, y_test_crudo,
                dir_procesados, "minmax", meta=metadatos
            )

        return rutas_crudos, ruta_baseline, metadatos

    if not descargar:
        print("La descarga está desactivada y los archivos no existen. No se puede continuar.")
        return None, None, None

    print("Descargando el dataset Fashion MNIST...")
    (x_train_crudo, y_train_crudo), (x_test_crudo, y_test_crudo) = tf.keras.datasets.fashion_mnist.load_data()
    print("Descarga completa.")

    np.save(rutas_crudos["X_train"], x_train_crudo)
    np.save(rutas_crudos["y_train"], y_train_crudo)
    np.save(rutas_crudos["X_test"], x_test_crudo)
    np.save(rutas_crudos["y_test"], y_test_crudo)
    print(f"Datos crudos guardados en: {dir_crudos}")

    metadatos = {
        "nombre_dataset": "Fashion MNIST",
        "fecha_descarga_utc": datetime.utcnow().isoformat(),
        "semilla_reproducibilidad": SEMILLA,
        "fuente": "TensorFlow/Keras Datasets",
        "versiones_paquetes": _obtener_versiones_paquetes(),
        "archivos_crudos": {
            "X_train": {"forma": x_train_crudo.shape, "tipo_dato": str(x_train_crudo.dtype), "tamano_bytes": x_train_crudo.nbytes},
            "y_train": {"forma": y_train_crudo.shape, "tipo_dato": str(y_train_crudo.dtype), "tamano_bytes": y_train_crudo.nbytes},
            "X_test": {"forma": x_test_crudo.shape, "tipo_dato": str(x_test_crudo.dtype), "tamano_bytes": x_test_crudo.nbytes},
            "y_test": {"forma": y_test_crudo.shape, "tipo_dato": str(y_test_crudo.dtype), "tamano_bytes": y_test_crudo.nbytes},
        }
    }
    with open(rutas_crudos["metadata"], 'w') as f:
        json.dump(metadatos, f, indent=4)
    print(f"Metadatos guardados en: {rutas_crudos['metadata']}")

    print("Generando versión procesada baseline (minmax)...")
    x_train_norm = x_train_crudo.astype('float32') / 255.0
    x_test_norm = x_test_crudo.astype('float32') / 255.0

    ruta_baseline = guardar_conjunto_procesado(
        x_train_norm, x_test_norm, y_train_crudo, y_test_crudo,
        dir_procesados, "minmax", meta=metadatos
    )

    return rutas_crudos, ruta_baseline, metadatos

def cargar_conjunto_procesado(dir_procesados="data/processed/", nombre_normalizador="minmax"):
    """
    Carga un conjunto de datos procesado desde el disco.

    Args:
        dir_procesados (str): Directorio base donde se encuentran los datos procesados.
        nombre_normalizador (str): Nombre de la versión de normalización a cargar.

    Returns:
        tuple: Tupla con (X_train, X_test, y_train, y_test, metadata).
               Retorna (None, None, None, None, None) si no se encuentra.
    """
    ruta_datos = os.path.join(dir_procesados, 'normalizations', nombre_normalizador)
    if not os.path.exists(ruta_datos):
        print(f"Error: El directorio de datos procesados '{ruta_datos}' no existe.")
        return None, None, None, None, None

    try:
        x_train = np.load(os.path.join(ruta_datos, 'X_train.npy'))
        x_test = np.load(os.path.join(ruta_datos, 'X_test.npy'))
        y_train = np.load(os.path.join(ruta_datos, 'y_train.npy'))
        y_test = np.load(os.path.join(ruta_datos, 'y_test.npy'))

        ruta_metadata = os.path.join(ruta_datos, 'metadata.json')
        if os.path.exists(ruta_metadata):
            with open(ruta_metadata, 'r') as f:
                metadatos = json.load(f)
        else:
            metadatos = None

        print(f"Datos cargados exitosamente desde '{ruta_datos}'.")
        return x_train, x_test, y_train, y_test, metadatos

    except FileNotFoundError as e:
        print(f"Error al cargar los archivos desde '{ruta_datos}': {e}")
        return None, None, None, None, None

if __name__ == '__main__':
    """
    Punto de entrada para ejecutar el script directamente.
    Esto descargará los datos y creará la estructura de archivos inicial.
    """
    print("Ejecutando el pipeline de carga de datos como un script independiente...")
    rutas_crudos, ruta_baseline, metadatos = ejecutar_pipeline_carga(sobrescribir=True)

    if rutas_crudos:
        print("\n--- Resumen de la Ejecución ---")
        print(f"Rutas de datos crudos generadas:")
        for clave, ruta in rutas_crudos.items():
            print(f"  - {clave}: {ruta}")

        print(f"\nRuta de la versión procesada (baseline): {ruta_baseline}")

        print("\nMetadatos generados:")
        print(json.dumps(metadatos, indent=2, ensure_ascii=False))
        print("\nPipeline de carga completado exitosamente.")
    else:
        print("\nEl pipeline de carga no se pudo completar.")