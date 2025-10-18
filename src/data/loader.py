"""
Módulo de carga de datos para el proyecto de reducción de dimensionalidad.
Maneja la descarga, lectura y guardado de datos crudos y procesados.
"""

import os
import json
from datetime import datetime
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import gdown
import numpy as np
import struct
import matplotlib
import platform
import sys
from pathlib import Path

def descargar_datos(id_archivo, nombre_archivo):
    """Descarga un archivo desde Google Drive usando su ID."""
    url = f"https://drive.google.com/uc?id={id_archivo}"
    gdown.download(url, nombre_archivo, quiet=False)
    return nombre_archivo

def leer_etiquetas(ruta_archivo):
    """Lee archivo de etiquetas en formato MNIST y retorna DataFrame con etiquetas y nombres de clase."""
    nombres_clases = {
        0: "Camiseta/Top",
        1: "Pantalón",
        2: "Suéter",
        3: "Vestido",
        4: "Abrigo",
        5: "Sandalia",
        6: "Camisa",
        7: "Zapatilla",
        8: "Bolso",
        9: "Botín"
    }

    with open(ruta_archivo, 'rb') as f:
        magico, num_etiquetas = struct.unpack(">II", f.read(8))
        etiquetas = np.frombuffer(f.read(), dtype=np.uint8)
    df = pd.DataFrame(etiquetas, columns=["etiqueta"])
    df["nombre_clase"] = df["etiqueta"].map(nombres_clases)
    return df


def extraer_caracteristicas_imagenes(ruta_archivo):
    """Extrae características de imágenes del formato MNIST."""
    with open(ruta_archivo, 'rb') as f:
        magico, num_imagenes, filas, cols = struct.unpack(">IIII", f.read(16))
        print("Número de imágenes:", num_imagenes)
        print("Dimensiones de cada imagen:", filas, "x", cols)
        datos_imagen = np.frombuffer(f.read(), dtype=np.uint8)
        imagenes = datos_imagen.reshape(num_imagenes, filas, cols)
        X = imagenes.reshape(num_imagenes, filas * cols).astype(np.float32)
        print("Forma de la matriz final:", X.shape) 
    return X

def mostrar_imagen(X, nro_imagen):
    """Muestra una imagen específica del conjunto de datos."""
    if nro_imagen < 0 or nro_imagen >= X.shape[0]:
        raise IndexError(f"El índice {nro_imagen} está fuera de rango. Debe estar entre 0 y {X.shape[0]-1}")

    img = X[nro_imagen].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f"Imagen #{nro_imagen}")
    plt.axis('off')
    plt.show()



def obtener_metadatos():
    """Obtiene metadatos del entorno y paquetes."""
    return {
        "marca_tiempo": datetime.utcnow().isoformat() + "Z",
        "plataforma": platform.platform(),
        "version_python": sys.version,
        "versiones_paquetes": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__
        }
    }

def guardar_arrays_procesados(X_train, X_test, y_train, y_test, dir_salida="data/processed/", nombre_normalizador="minmax", meta=None):
    """Guarda arrays ya normalizados/transformados y registra el normalizador en meta."""
    os.makedirs(dir_salida, exist_ok=True)
    
    # Guardar arrays
    np.save(os.path.join(dir_salida, "X_train.npy"), X_train)
    np.save(os.path.join(dir_salida, "X_test.npy"), X_test)
    
    # Convertir y guardar etiquetas como CSV
    if isinstance(y_train, pd.DataFrame):
        y_train.to_csv(os.path.join(dir_salida, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(dir_salida, "y_test.csv"), index=False)
    else:
        pd.DataFrame(y_train, columns=["etiqueta"]).to_csv(os.path.join(dir_salida, "y_train.csv"), index=False)
        pd.DataFrame(y_test, columns=["etiqueta"]).to_csv(os.path.join(dir_salida, "y_test.csv"), index=False)
    
    # Actualizar y guardar metadatos
    if meta is None:
        meta = {}
    meta.update({
        "normalizador": nombre_normalizador,
        "formas": {
            "X_train": X_train.shape,
            "X_test": X_test.shape
        },
        "marca_tiempo": datetime.utcnow().isoformat() + "Z"
    })
    
    with open(os.path.join(dir_salida, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    return meta

def cargar_procesados(dir_salida="data/processed/"):
    """Carga arrays procesados y metadatos."""
    X_train = np.load(os.path.join(dir_salida, "X_train.npy"))
    X_test = np.load(os.path.join(dir_salida, "X_test.npy"))
    y_train = pd.read_csv(os.path.join(dir_salida, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(dir_salida, "y_test.csv"))
    
    try:
        with open(os.path.join(dir_salida, "metadata.json"), 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = None
        
    return X_train, X_test, y_train, y_test, meta

def ejecutar_pipeline_carga(descarga=True, dir_crudo="data/raw/", dir_procesado="data/processed/", sobreescribir=False, semilla=42):
    """Pipeline principal que maneja descarga, lectura y guardado de datos.
    
    Args:
        descarga: Si True, descarga los archivos. Si False, asume que existen.
        dir_crudo: Directorio para datos crudos.
        dir_procesado: Directorio para datos procesados.
        sobreescribir: Si True, sobrescribe archivos existentes.
        semilla: Semilla para reproducibilidad.
    
    Returns:
        dict con metadatos y rutas.
    """
    os.makedirs(dir_crudo, exist_ok=True)
    os.makedirs(dir_procesado, exist_ok=True)
    
    # IDs de archivos en Drive
    ids_archivos = {
        "train_X": "1enziBIpqiv_t95KQcifsclNH2BdR8lAd",
        "test_X": "1Jeax6tnQ6Nmr2PTNXdQqzKnN0YqtrLe4",
        "train_Y": "1MZtn2iA5cgiYT1i3O0ECuR01oD0kGHh7",
        "test_Y": "1K5pxwk2s3RDYsYuwv8RftJTXZ-RGR7K4"
    }
    
    archivos = {}
    meta = obtener_metadatos()
    meta["semilla"] = semilla
    meta["ids_archivos"] = ids_archivos
    
    # Descarga/carga de archivos
    if descarga:
        for nombre, id_archivo in ids_archivos.items():
            ruta = os.path.join(dir_crudo, nombre)
            if sobreescribir or not os.path.exists(ruta):
                archivos[nombre] = descargar_datos(id_archivo, ruta)
            else:
                archivos[nombre] = ruta
    else:
        for nombre in ids_archivos:
            ruta = os.path.join(dir_crudo, nombre)
            if not os.path.exists(ruta):
                raise FileNotFoundError(f"Archivo {ruta} no encontrado. Use descarga=True para descargar.")
            archivos[nombre] = ruta
    
    # Lectura de datos
    X_train = extraer_caracteristicas_imagenes(archivos["train_X"])
    X_test = extraer_caracteristicas_imagenes(archivos["test_X"])
    y_train = leer_etiquetas(archivos["train_Y"])
    y_test = leer_etiquetas(archivos["test_Y"])
    
    meta["formas"] = {
        "X_train": X_train.shape,
        "X_test": X_test.shape
    }
    
    # Guardar metadatos de datos crudos
    with open(os.path.join(dir_crudo, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    # Crear versión procesada base (/255)
    dir_base = os.path.join(dir_procesado, "baseline")
    if sobreescribir or not os.path.exists(dir_base):
        X_train_base = X_train / 255.0
        X_test_base = X_test / 255.0
        guardar_arrays_procesados(
            X_train_base, X_test_base, y_train, y_test,
            dir_salida=dir_base,
            nombre_normalizador="division_255",
            meta=meta
        )
    
    return {
        "rutas": archivos,
        "meta": meta,
        "dir_crudo": dir_crudo,
        "dir_procesado": dir_procesado
    }








