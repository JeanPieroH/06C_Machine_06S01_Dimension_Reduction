"""
Módulo de entrenamiento de modelos para clasificación.
"""
import os
import json
import time
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def obtener_clasificador(nombre, **params):
    """Devuelve un clasificador con parámetros estándar y semilla fija.
    
    Args:
        nombre: Tipo de clasificador ('svm', 'logistic', 'rf', 'knn')
        **params: Parámetros específicos del clasificador
    """
    clasificadores = {
        'svm': lambda p: SVC(kernel='rbf', random_state=42, **p),
        'logistic': lambda p: LogisticRegression(random_state=42, max_iter=1000, **p),
        'rf': lambda p: RandomForestClassifier(n_estimators=100, random_state=42, **p),
        'knn': lambda p: KNeighborsClassifier(**p)
    }
    
    if nombre not in clasificadores:
        raise ValueError(f"Clasificador '{nombre}' no soportado. Opciones: {list(clasificadores.keys())}")
    
    return clasificadores[nombre](params)

def entrenar_modelo(clf, X_train, y_train):
    """Entrena un clasificador y mide el tiempo de entrenamiento.
    
    Args:
        clf: Clasificador scikit-learn
        X_train: Datos de entrenamiento
        y_train: Etiquetas
        
    Returns:
        tuple: (clasificador_entrenado, tiempo_entrenamiento)
    """
    inicio = time.time()
    clf.fit(X_train, y_train)
    tiempo = time.time() - inicio
    return clf, tiempo

def guardar_modelo(ruta, clf, meta):
    """Guarda un modelo entrenado y sus metadatos.
    
    Args:
        ruta: Ruta base para guardar el modelo
        clf: Clasificador entrenado
        meta: Diccionario con metadatos (n_components, method, métricas, etc.)
    """
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    
    # Guardar modelo
    joblib.dump(clf, f"{ruta}.joblib")
    
    # Guardar metadatos
    with open(f"{ruta}_meta.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)