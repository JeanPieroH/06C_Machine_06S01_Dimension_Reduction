# -*- coding: utf-8 -*-
"""
Módulo para la evaluación de modelos de clasificación.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

def evaluar_modelo(clf, x_test, y_test, nombres_clases=None):
    """
    Evalúa un clasificador en el conjunto de prueba y devuelve un conjunto completo de métricas.

    Args:
        clf (object): Clasificador de scikit-learn ajustado.
        x_test (np.ndarray): Datos de prueba.
        y_test (np.ndarray): Etiquetas verdaderas de prueba.
        nombres_clases (list, optional): Nombres de las clases para el reporte.

    Returns:
        tuple: (
            dict_metricas: Diccionario con las principales métricas.
            matriz_confusion: Matriz de confusión de numpy.
            reporte_clasificacion: Diccionario del reporte de clasificación de sklearn.
            y_pred: Predicciones del modelo.
            y_score: Puntuaciones de probabilidad del modelo.
        )
    """
    # Realizar predicciones
    y_pred = clf.predict(x_test)
    y_score = clf.predict_proba(x_test) if hasattr(clf, "predict_proba") else None

    # Calcular métricas principales
    dict_metricas = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
    }

    # Matriz de confusión
    matriz_confusion = confusion_matrix(y_test, y_pred)

    # Reporte de clasificación completo
    reporte_clasificacion = classification_report(
        y_test, y_pred,
        target_names=nombres_clases,
        output_dict=True,
        zero_division=0
    )

    return dict_metricas, matriz_confusion, reporte_clasificacion, y_pred, y_score


def estimar_cv_anidado(pipeline_constructor, X, y, cv=5):
    """
    Placeholder para la estimación de rendimiento con validación cruzada anidada.
    Esta es una técnica robusta pero computacionalmente costosa.
    """
    print("Función 'estimar_cv_anidado' no implementada.")
    # Lógica futura: Bucle externo para la división de prueba, bucle interno para la búsqueda de hiperparámetros.
    return {"f1_macro_mean": np.nan, "f1_macro_std": np.nan}


def evaluacion_bootstrap(pipeline_constructor, X, y, B=30):
    """
    Placeholder para la evaluación de la distribución de una métrica usando bootstrap.
    """
    print("Función 'evaluacion_bootstrap' no implementada.")
    # Lógica futura: Muestrear con reemplazo B veces, entrenar y evaluar en cada muestra.
    return np.array([np.nan] * B)