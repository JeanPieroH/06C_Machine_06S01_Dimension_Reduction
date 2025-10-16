# -*- coding: utf-8 -*-
"""
Módulo para la carga, procesamiento y guardado de datos.

Funciones:
- generar_datos_simulados: Crea datos de ejemplo para train y test.
- guardar_archivos_raw: Guarda los datos originales en formato .npy.
- guardar_arrays_procesados: Normaliza y guarda los datos procesados.
- ejecutar_pipeline_carga: Orquesta el proceso completo de ingestión de datos.
"""

import numpy as np
import pandas as pd
import os

def generar_datos_simulados():
    """
    Genera y devuelve datos simulados para entrenamiento y prueba.
    - X: Imágenes simuladas (valores de 0 a 255).
    - y: Etiquetas simuladas (enteros de 0 a 9).
    """
    print("Generando datos simulados...")
    # Datos de entrenamiento
    train_X = np.random.randint(0, 256, size=(100, 28, 28), dtype=np.uint8)
    train_y = np.random.randint(0, 10, size=(100,), dtype=np.uint8)

    # Datos de prueba
    test_X = np.random.randint(0, 256, size=(20, 28, 28), dtype=np.uint8)
    test_y = np.random.randint(0, 10, size=(20,), dtype=np.uint8)

    print("Datos simulados generados con éxito.")
    return train_X, train_y, test_X, test_y

def guardar_archivos_raw(train_X, train_y, test_X, test_y, out_dir="data/raw/"):
    """
    Guarda los conjuntos de datos raw en la ruta especificada.

    Args:
        train_X (np.ndarray): Datos de entrenamiento X.
        train_y (np.ndarray): Etiquetas de entrenamiento y.
        test_X (np.ndarray): Datos de prueba X.
        test_y (np.ndarray): Etiquetas de prueba y.
        out_dir (str): Directorio de salida para los archivos raw.
    """
    print(f"Guardando archivos raw en '{out_dir}'...")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "train_X.npy"), train_X)
    np.save(os.path.join(out_dir, "train_Y.npy"), train_y)
    np.save(os.path.join(out_dir, "test_X.npy"), test_X)
    np.save(os.path.join(out_dir, "test_Y.npy"), test_y)

    print("Archivos raw guardados con éxito.")

def guardar_arrays_procesados(X_train, X_test, y_train, y_test, out_dir="data/processed/"):
    """
    Normaliza los datos X y guarda los arrays procesados.

    Args:
        X_train (np.ndarray): Datos de entrenamiento X.
        X_test (np.ndarray): Datos de prueba X.
        y_train (np.ndarray): Etiquetas de entrenamiento y.
        y_test (np.ndarray): Etiquetas de prueba y.
        out_dir (str): Directorio de salida para los archivos procesados.
    """
    print(f"Procesando y guardando arrays en '{out_dir}'...")
    os.makedirs(out_dir, exist_ok=True)

    # Normalización de las imágenes
    X_train_proc = X_train.astype('float32') / 255.0
    X_test_proc = X_test.astype('float32') / 255.0

    # Guardado de arrays NumPy
    np.save(os.path.join(out_dir, "X_train.npy"), X_train_proc)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test_proc)

    # Guardado de etiquetas como CSV
    pd.DataFrame(y_train, columns=['label']).to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test, columns=['label']).to_csv(os.path.join(out_dir, "y_test.csv"), index=False)

    print("Arrays procesados guardados con éxito.")

def ejecutar_pipeline_carga(raw_out="data/raw/", processed_out="data/processed/", overwrite=False):
    """
    Orquesta la ejecución del pipeline de carga de datos.

    Args:
        raw_out (str): Directorio de salida para datos raw.
        processed_out (str): Directorio de salida para datos procesados.
        overwrite (bool): Si es True, sobrescribe los archivos existentes.
    """
    print("Iniciando pipeline de carga de datos...")

    # Verificar si los archivos ya existen
    archivos_existen = os.path.exists(os.path.join(raw_out, "train_X.npy"))
    if archivos_existen and not overwrite:
        print(f"Los archivos ya existen en '{raw_out}'. Use overwrite=True para regenerarlos.")
        return

    # 1. Generar datos simulados
    train_X, train_y, test_X, test_y = generar_datos_simulados()

    # 2. Guardar archivos raw
    guardar_archivos_raw(train_X, train_y, test_X, test_y, out_dir=raw_out)

    # 3. Crear y guardar versiones procesadas
    guardar_arrays_procesados(train_X, test_X, train_y, test_y, out_dir=processed_out)

    print("Pipeline de carga de datos finalizado con éxito.")

if __name__ == '__main__':
    # Ejemplo de uso desde la línea de comandos
    # Se asume que se ejecuta desde la raíz del proyecto, por lo que las rutas relativas son correctas.
    ejecutar_pipeline_carga(
        raw_out="project_name/data/raw/",
        processed_out="project_name/data/processed/",
        overwrite=True
    )