# -*- coding: utf-8 -*-
"""
Módulo para funciones de selección de modelos y análisis de resultados.
"""
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def calcular_frente_pareto(df, objetivos):
    """
    Calcula el frente de Pareto para un conjunto de soluciones.

    Args:
        df (pd.DataFrame): DataFrame con las soluciones y sus métricas.
        objetivos (dict): Diccionario donde las claves son los nombres de las columnas
                          y los valores son 'max' o 'min' para la optimización.
                          Ej: {'f1_macro': 'max', 'tiempo_fit_s': 'min'}

    Returns:
        pd.DataFrame: Un subconjunto del DataFrame original que representa el frente de Pareto.
    """
    indices_pareto = []

    for i in range(len(df)):
        es_dominado = False
        for j in range(len(df)):
            if i == j:
                continue

            # Comprobar si la solución j domina a la solución i
            domina = True
            for obj, direccion in objetivos.items():
                if direccion == 'max':
                    if df.iloc[j][obj] < df.iloc[i][obj]:
                        domina = False
                        break
                elif direccion == 'min':
                    if df.iloc[j][obj] > df.iloc[i][obj]:
                        domina = False
                        break

            # Si alguna otra solución la domina estrictamente en al menos un objetivo
            # y no es peor en ninguno
            if domina:
                es_estrictamente_mejor = any(
                    (df.iloc[j][obj] > df.iloc[i][obj] if d == 'max' else df.iloc[j][obj] < df.iloc[i][obj])
                    for obj, d in objetivos.items()
                )
                if es_estrictamente_mejor:
                    es_dominado = True
                    break

        if not es_dominado:
            indices_pareto.append(i)

    return df.iloc[indices_pareto].sort_values(list(objetivos.keys())[0], ascending=False)


def prueba_permutacion_f1(y_true, y_pred_a, y_pred_b, metrica_func, n_permutaciones=1000):
    """
    Realiza una prueba de permutación para comparar la métrica F1 de dos modelos.

    Hipótesis nula (H0): Los dos modelos tienen el mismo rendimiento (la diferencia en F1 es casual).

    Args:
        y_true (np.array): Etiquetas verdaderas.
        y_pred_a (np.array): Predicciones del modelo A.
        y_pred_b (np.array): Predicciones del modelo B.
        metrica_func (function): Función de métrica a usar (ej. f1_score con average='macro').
        n_permutaciones (int): Número de permutaciones a realizar.

    Returns:
        tuple: (diferencia_observada, p_valor)
    """
    diferencia_observada = metrica_func(y_true, y_pred_a) - metrica_func(y_true, y_pred_b)

    y_pred_combinadas = np.vstack((y_pred_a, y_pred_b)).T
    count = 0

    for _ in range(n_permutaciones):
        # Permutar las predicciones para cada muestra
        indices_permutados = np.random.permutation(y_pred_combinadas.shape[1])
        y_pred_a_perm = y_pred_combinadas[:, indices_permutados][:, 0]
        y_pred_b_perm = y_pred_combinadas[:, indices_permutados][:, 1]

        diferencia_perm = metrica_func(y_true, y_pred_a_perm) - metrica_func(y_true, y_pred_b_perm)

        if abs(diferencia_perm) >= abs(diferencia_observada):
            count += 1

    p_valor = count / n_permutaciones

    return diferencia_observada, p_valor


def bootstrap_ci(metric_values, alpha=0.05):
    """
    Calcula el intervalo de confianza de una métrica usando percentiles de bootstrap.

    Args:
        metric_values (np.array): Array de valores de la métrica obtenidos de las muestras bootstrap.
        alpha (float): Nivel de significancia.

    Returns:
        tuple: (limite_inferior, limite_superior)
    """
    percentil_inferior = (alpha / 2.0) * 100
    percentil_superior = (1.0 - alpha / 2.0) * 100
    limite_inferior = np.percentile(metric_values, percentil_inferior)
    limite_superior = np.percentile(metric_values, percentil_superior)
    return limite_inferior, limite_superior


def seleccionar_n_minimo_con_epsilon(df, metrica='f1_macro', epsilon=0.98):
    """
    Selecciona los modelos cuyo rendimiento está dentro de un umbral del mejor
    y luego elige el de menor dimensionalidad para cada método.

    Args:
        df (pd.DataFrame): DataFrame con los resultados.
        metrica (str): Métrica a optimizar.
        epsilon (float): Porcentaje del máximo rendimiento a considerar.

    Returns:
        pd.DataFrame: DataFrame con los candidatos seleccionados.
    """
    f1_max = df[metrica].max()
    umbral = epsilon * f1_max

    candidatos_eps = df[df[metrica] >= umbral].copy()

    # Para cada método, encontrar el que tiene el menor 'n_componentes'
    idx_min_n = candidatos_eps.groupby('metodo_reduccion')['n_componentes'].idxmin()

    seleccion_final = candidatos_eps.loc[idx_min_n]

    return seleccion_final.sort_values(metrica, ascending=False)