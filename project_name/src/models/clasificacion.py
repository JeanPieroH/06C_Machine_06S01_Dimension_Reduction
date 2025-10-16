# -*- coding: utf-8 -*-
"""
Módulo para el entrenamiento y evaluación de modelos de clasificación.

Funciones:
- obtener_clasificador: Devuelve una instancia de un clasificador de sklearn.
- entrenar_modelo: Entrena un clasificador y mide el tiempo de ejecución.
- evaluar_modelo: Calcula métricas de rendimiento y la matriz de confusión.
- guardar_modelo: Serializa un modelo entrenado usando joblib.
"""

import os
import time
import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def obtener_clasificador(nombre, random_state=42, **params):
    """
    Obtiene una instancia de un clasificador de scikit-learn.

    Args:
        nombre (str): Nombre del clasificador ('svm', 'logistic', 'rf', 'knn').
        random_state (int): Semilla aleatoria para reproducibilidad.
        **params: Parámetros adicionales para el constructor del clasificador.

    Returns:
        Un estimador de scikit-learn.
    """
    if nombre == 'svm':
        return SVC(random_state=random_state, **params)
    elif nombre == 'logistic':
        return LogisticRegression(random_state=random_state, max_iter=1000, **params)
    elif nombre == 'rf':
        return RandomForestClassifier(random_state=random_state, **params)
    elif nombre == 'knn':
        # KNN no tiene random_state
        return KNeighborsClassifier(**params)
    else:
        raise ValueError(f"Clasificador '{nombre}' no soportado.")

def entrenar_modelo(clasificador, X_train, y_train):
    """
    Entrena un modelo de clasificación y devuelve el modelo ajustado y el tiempo de entrenamiento.

    Args:
        clasificador: El objeto clasificador de sklearn.
        X_train (np.ndarray): Datos de entrenamiento.
        y_train (np.ndarray): Etiquetas de entrenamiento.

    Returns:
        tuple: (clasificador_ajustado, tiempo_de_entrenamiento)
    """
    print(f"Entrenando clasificador {type(clasificador).__name__}...")
    tiempo_inicio = time.time()
    clasificador.fit(X_train, y_train)
    tiempo_fin = time.time()

    tiempo_entrenamiento = round(tiempo_fin - tiempo_inicio, 4)
    print(f"Entrenamiento completado en {tiempo_entrenamiento:.4f} segundos.")

    return clasificador, tiempo_entrenamiento

def evaluar_modelo(clasificador, X_test, y_test):
    """
    Evalúa un modelo y devuelve un diccionario de métricas y la matriz de confusión.

    Args:
        clasificador: El clasificador ajustado.
        X_test (np.ndarray): Datos de prueba.
        y_test (np.ndarray): Etiquetas de prueba.

    Returns:
        tuple: (diccionario_de_metricas, matriz_de_confusion)
    """
    print("Evaluando modelo...")
    y_pred = clasificador.predict(X_test)

    metricas = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision_macro': round(precision_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'recall_macro': round(recall_score(y_test, y_pred, average='macro', zero_division=0), 4),
        'f1_macro': round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4)
    }

    matriz_confusion = confusion_matrix(y_test, y_pred)

    print(f"Evaluación completada. F1-score (macro): {metricas['f1_macro']:.4f}")
    return metricas, matriz_confusion

def guardar_modelo(ruta_archivo, clasificador):
    """
    Guarda un modelo entrenado en un archivo usando joblib.

    Args:
        ruta_archivo (str): Ruta donde se guardará el modelo.
        clasificador: El objeto clasificador ajustado.
    """
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    joblib.dump(clasificador, ruta_archivo)
    print(f"Modelo guardado en: {ruta_archivo}")