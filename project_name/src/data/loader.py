# -*- coding: utf-8 -*-
"""
Módulo para la carga, procesamiento y guardado de datos.

Funciones:
- leer_etiquetas_idx: Lee un archivo de etiquetas en formato IDX.
- leer_imagenes_idx: Lee un archivo de imágenes en formato IDX.
- normalizar_y_guardar: Normaliza los datos y los guarda en formato npy/csv.
"""

import numpy as np
import pandas as pd
import os
import struct
import gdown

def descargar_datos(id_archivo, nombre_archivo):
    """
    Descarga un archivo desde Google Drive.
    """
    url = f"https://drive.google.com/uc?id={id_archivo}"
    gdown.download(url, nombre_archivo, quiet=False)
    return nombre_archivo

def leer_etiquetas_idx(ruta_archivo):
    """
    Lee un archivo de etiquetas en formato IDX y devuelve un DataFrame de pandas.
    """
    with open(ruta_archivo, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return pd.DataFrame(labels, columns=["label"])

def leer_imagenes_idx(ruta_archivo):
    """
    Lee un archivo de imágenes en formato IDX y devuelve un array de NumPy.
    """
    with open(ruta_archivo, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
    return images

def normalizar_y_guardar(X_train, X_test, y_train, y_test, out_dir="data/processed/"):
    """
    Normaliza los datos X y guarda los arrays procesados.

    Args:
        X_train (np.ndarray): Datos de entrenamiento X.
        X_test (np.ndarray): Datos de prueba X.
        y_train (pd.DataFrame): Etiquetas de entrenamiento y.
        y_test (pd.DataFrame): Etiquetas de prueba y.
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
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)

    print("Arrays procesados guardados con éxito.")

if __name__ == '__main__':
    # IDs de los archivos en Google Drive
    file_ids = {
        'train_X': '1enziBIpqiv_t95KQcifsclNH2BdR8lAd',
        'test_X': '1Jeax6tnQ6Nmr2PTNXdQqzKnN0YqtrLe4',
        'train_Y': '1MZtn2iA5cgiYT1i3O0ECuR01oD0kGHh7',
        'test_Y': '1K5pxwk2s3RDYsYuwv8RftJTXZ-RGR7K4'
    }

    # Directorio para los datos raw
    raw_dir = 'data/raw/'
    os.makedirs(raw_dir, exist_ok=True)

    # Descargar los archivos
    ruta_train_x = descargar_datos(file_ids['train_X'], os.path.join(raw_dir, 'train-images-idx3-ubyte'))
    ruta_test_x = descargar_datos(file_ids['test_X'], os.path.join(raw_dir, 't10k-images-idx3-ubyte'))
    ruta_train_y = descargar_datos(file_ids['train_Y'], os.path.join(raw_dir, 'train-labels-idx1-ubyte'))
    ruta_test_y = descargar_datos(file_ids['test_Y'], os.path.join(raw_dir, 't10k-labels-idx1-ubyte'))

    # Cargar los datos raw
    train_X = leer_imagenes_idx(ruta_train_x)
    train_y = leer_etiquetas_idx(ruta_train_y)
    test_X = leer_imagenes_idx(ruta_test_x)
    test_y = leer_etiquetas_idx(ruta_test_y)

    # Normalizar y guardar los datos procesados
    normalizar_y_guardar(train_X, test_X, train_y, test_y, out_dir="data/processed/")