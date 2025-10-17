# -*- coding: utf-8 -*-
"""
Módulo para funciones de Análisis Exploratorio de Datos (EDA) cuantitativo.

Funciones:
- calcular_varianza_acumulada_pca: Calcula la varianza acumulada de PCA.
- calcular_informacion_mutua_componentes: Calcula la información mutua entre componentes y etiquetas.
- estimar_dimension_intrinseca_twonn: Estima la dimensión intrínseca con TwoNN.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors

def calcular_varianza_acumulada_pca(X, max_componentes=None):
    """
    Calcula la varianza acumulada para componentes de PCA.

    Args:
        X (np.ndarray): Datos de entrada (n_muestras, n_caracteristicas).
        max_componentes (int, optional): Número máximo de componentes a evaluar.
                                      Si es None, se usa min(n_muestras, n_caracteristicas).

    Returns:
        tuple: (pd.Series con la varianza acumulada,
                dict con n_componentes para 90% y 95% de varianza).
    """
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    pca = PCA(n_components=max_componentes, random_state=42)
    pca.fit(X)

    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
    serie_varianza_acumulada = pd.Series(varianza_acumulada, index=range(1, len(varianza_acumulada) + 1), name='varianza_acumulada')

    n_para_90 = np.argmax(varianza_acumulada >= 0.90) + 1
    n_para_95 = np.argmax(varianza_acumulada >= 0.95) + 1

    umbrales = {
        'n_componentes_90_varianza': n_para_90,
        'n_componentes_95_varianza': n_para_95
    }

    print(f"PCA: Se necesitan {n_para_90} componentes para 90% de varianza.")
    print(f"PCA: Se necesitan {n_para_95} componentes para 95% de varianza.")

    return serie_varianza_acumulada, umbrales

def calcular_informacion_mutua_componentes(X, y, n_componentes=100):
    """
    Calcula la información mutua (MI) entre las etiquetas y los componentes principales de PCA.

    Args:
        X (np.ndarray): Datos de entrada (n_muestras, n_caracteristicas).
        y (np.ndarray): Etiquetas (n_muestras,).
        n_componentes (int): Número de componentes de PCA a considerar.

    Returns:
        pd.Series: Serie con la información mutua para cada componente.
    """
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    n_componentes = min(n_componentes, X.shape[1])

    pca = PCA(n_components=n_componentes, random_state=42)
    X_pca = pca.fit_transform(X)

    puntajes_mi = mutual_info_classif(X_pca, y, random_state=42)
    serie_mi = pd.Series(puntajes_mi, index=[f'PC_{i+1}' for i in range(n_componentes)], name='informacion_mutua')

    print(f"Información mutua calculada para {n_componentes} componentes.")

    return serie_mi.sort_values(ascending=False)

def estimar_dimension_intrinseca_twonn(X, k=5):
    """
    Estima la dimensión intrínseca usando el estimador TwoNN.

    Args:
        X (np.ndarray): Datos de entrada (n_muestras, n_caracteristicas).
        k (int): Número de vecinos a considerar.

    Returns:
        float: Estimación de la dimensión intrínseca.
    """
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    vecinos = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(X)
    distancias, _ = vecinos.kneighbors(X)
    distancias = distancias[:, 1:]

    dist_r1 = distancias[:, 0]
    dist_rk = distancias[:, -1]

    mascara = dist_r1 > 0
    if not np.any(mascara):
        print("Advertencia: Todos los puntos son duplicados. No se puede estimar la dimensión.")
        return np.nan

    ratios = dist_rk[mascara] / dist_r1[mascara]

    ratios_ordenados = np.sort(ratios)
    cdf = np.arange(1, len(ratios_ordenados) + 1) / len(ratios_ordenados)

    # Para evitar inestabilidad numérica, ajustamos solo en la parte central de la distribución
    indices_validos = (cdf > 0.01) & (cdf < 0.99)
    if not np.any(indices_validos):
        print("Advertencia: No hay suficientes puntos para un ajuste estable de la dimensión.")
        return np.nan

    log_ratios = np.log(ratios_ordenados[indices_validos])
    # La teoría sugiere que (1-F(x)) ~ x^(-d), por lo que log(1-F(x)) ~ -d*log(x)
    log_cdf_inverso = np.log(1 - cdf[indices_validos])

    try:
        # A es la matriz de diseño, y resolvemos para la pendiente 'm'
        A = np.vstack([log_ratios, np.ones(len(log_ratios))]).T
        # La pendiente (m) es nuestra estimación de la dimensión (-d)
        m, _ = np.linalg.lstsq(A, -log_cdf_inverso, rcond=None)[0]
        estimacion_dim = m
    except np.linalg.LinAlgError:
        estimacion_dim = np.nan

    print(f"Estimación de dimensión intrínseca (TwoNN, k={k}): {estimacion_dim:.2f}")

    return estimacion_dim