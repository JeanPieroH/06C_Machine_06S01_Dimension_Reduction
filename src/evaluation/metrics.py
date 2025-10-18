"""
Módulo de evaluación y métricas para clasificación multiclase.
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.base import clone
from ..utils.helpers import guardar_figura

def evaluate_model(clf, X_test, y_test, nombres_clases):
    """Evalúa un modelo entrenado y retorna métricas detalladas.
    
    Args:
        clf: Clasificador entrenado
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas
        nombres_clases: Lista de nombres de clases
        
    Returns:
        tuple: (dict_metricas, df_confusion, df_metricas_clase)
    """
    # Predicciones
    y_pred = clf.predict(X_test)
    
    # Métricas globales
    metricas = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_micro': f1_score(y_test, y_pred, average='micro')
    }
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    df_confusion = pd.DataFrame(
        cm, 
        index=nombres_clases,
        columns=nombres_clases
    )
    
    # Métricas por clase
    report = classification_report(y_test, y_pred, target_names=nombres_clases, output_dict=True)
    df_metricas_clase = pd.DataFrame.from_dict(report).transpose()
    df_metricas_clase = df_metricas_clase.drop(['accuracy', 'macro avg', 'weighted avg'])
    df_metricas_clase.index.name = 'class'
    df_metricas_clase = df_metricas_clase.reset_index()
    
    return metricas, df_confusion, df_metricas_clase

def plot_confusion_matrix(cm, nombres_clases, normalize=True, prefix=''):
    """Genera y guarda visualización de matriz de confusión.
    
    Args:
        cm: Matriz de confusión
        nombres_clases: Lista de nombres de clases
        normalize: Si normalizar valores (default: True)
        prefix: Prefijo para nombre archivo
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=nombres_clases,
                yticklabels=nombres_clases)
    plt.title('Matriz de Confusión Normalizada' if normalize else 'Matriz de Confusión')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    
    # Guardar
    nombre = f"{prefix}_confusion_matrix_{'norm' if normalize else 'raw'}"
    ruta_base = os.path.join('outputs', 'figures', nombre)
    guardar_figura(plt.gcf(), ruta_base)
    plt.close()

def nested_cv_estimate(constructor_pipeline, X, y, cv_externo=5, cv_interno=5):
    """Estima rendimiento usando validación cruzada anidada.
    
    Args:
        constructor_pipeline: Función que retorna un nuevo pipeline
        X: Datos
        y: Etiquetas
        cv_externo: Número de folds externos
        cv_interno: Número de folds internos
    
    Returns:
        dict: Métricas con media y desviación estándar
    """
    cv_externo = StratifiedKFold(n_splits=cv_externo, shuffle=True, random_state=42)
    metricas = []
    
    for fold, (train_idx, test_idx) in enumerate(cv_externo.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Entrenar y evaluar
        pipeline = constructor_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calcular métricas
        fold_metrics = {
            'fold': fold,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro')
        }
        metricas.append(fold_metrics)
    
    # Calcular estadísticas
    df_metricas = pd.DataFrame(metricas)
    resultados = {
        metric: {
            'media': df_metricas[metric].mean(),
            'std': df_metricas[metric].std()
        }
        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    }
    
    return resultados

def bootstrap_eval(constructor_pipeline, X, y, B=30):
    """Genera distribución de métricas usando bootstrap.
    
    Args:
        constructor_pipeline: Función que retorna un nuevo pipeline
        X: Datos
        y: Etiquetas
        B: Número de muestras bootstrap
    
    Returns:
        pd.DataFrame: Métricas por cada muestra bootstrap
    """
    n_samples = len(X)
    resultados = []
    
    for b in range(B):
        # Muestreo bootstrap
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot, y_boot = X[indices], y[indices]
        
        # Train-test split en muestra bootstrap
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=b)
        train_idx, test_idx = next(cv.split(X_boot, y_boot))
        
        X_train, X_test = X_boot[train_idx], X_boot[test_idx]
        y_train, y_test = y_boot[train_idx], y_boot[test_idx]
        
        # Entrenar y evaluar
        pipeline = constructor_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Métricas
        metricas = {
            'muestra': b,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro')
        }
        resultados.append(metricas)
    
    return pd.DataFrame(resultados)
