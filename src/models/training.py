# -*- coding: utf-8 -*-
"""
Módulo para el entrenamiento y gestión de modelos de clasificación.
"""
import time
import joblib
import os
import json

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def obtener_clasificador(nombre, semilla_aleatoria=42, **kwargs):
    """
    Factory que devuelve un objeto clasificador de scikit-learn.

    Args:
        nombre (str): Nombre del clasificador ('svm', 'logistic', 'rf', 'knn').
        semilla_aleatoria (int): Semilla para métodos estocásticos.
        **kwargs: Argumentos adicionales para el constructor del clasificador.

    Returns:
        object: Una instancia de un clasificador de scikit-learn.
    """
    # Parámetros por defecto para asegurar reproducibilidad y buen rendimiento base
    params_comunes = {'random_state': semilla_aleatoria}

    clasificadores = {
        'svm': SVC(probability=True, **params_comunes),
        'logistic': LogisticRegression(max_iter=1000, **params_comunes),
        'rf': RandomForestClassifier(**params_comunes, n_jobs=-1),
        'knn': KNeighborsClassifier(n_jobs=-1)
    }

    if nombre.lower() not in clasificadores:
        raise ValueError(f"Clasificador '{nombre}' no reconocido. Opciones: {list(clasificadores.keys())}")

    clf = clasificadores[nombre.lower()]

    # Aplicar parámetros personalizados del usuario
    # kwargs puede sobreescribir los defaults si es necesario
    params_finales = clf.get_params()
    params_finales.update(kwargs)
    clf.set_params(**params_finales)

    return clf

def entrenar_modelo(clf, x_train, y_train):
    """
    Entrena un clasificador y mide el tiempo de entrenamiento.

    Args:
        clf (object): Clasificador de scikit-learn.
        x_train (np.ndarray): Datos de entrenamiento.
        y_train (np.ndarray): Etiquetas de entrenamiento.

    Returns:
        tuple: (clasificador_ajustado, tiempo_entrenamiento_s)
    """
    print(f"Entrenando {type(clf).__name__}...")
    tiempo_inicio = time.time()
    clf.fit(x_train, y_train)
    tiempo_fin = time.time()
    tiempo_entrenamiento = tiempo_fin - tiempo_inicio
    print(f"Entrenamiento completado en {tiempo_entrenamiento:.4f} segundos.")
    return clf, tiempo_entrenamiento

def guardar_modelo(ruta_base, clf, meta):
    """
    Guarda un modelo entrenado y sus metadatos.

    Args:
        ruta_base (str): Ruta base para guardar el modelo sin extensión.
        clf (object): Modelo de scikit-learn ajustado.
        meta (dict): Diccionario con metadatos a guardar en un archivo JSON.
    """
    dir_salida = os.path.dirname(ruta_base)
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)

    # Guardar modelo
    ruta_modelo = f"{ruta_base}.joblib"
    joblib.dump(clf, ruta_modelo)

    # Guardar metadatos
    meta['ruta_modelo'] = ruta_modelo
    meta['tamano_modelo_bytes'] = os.path.getsize(ruta_modelo)

    ruta_meta = f"{ruta_base}_meta.json"
    with open(ruta_meta, 'w') as f:
        json.dump(meta, f, indent=4)

    print(f"Modelo guardado en: {ruta_modelo}")
    print(f"Metadatos del modelo guardados en: {ruta_meta}")