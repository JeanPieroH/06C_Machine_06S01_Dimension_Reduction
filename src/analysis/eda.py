# -*- coding: utf-8 -*-
"""
Módulo con funciones de análisis exploratorio de datos (EDA) cuantitativo.
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors

def save_figure(fig, path_base, dpi=300):
    """
    Guarda una figura de matplotlib en formatos PNG y SVG.

    Args:
        fig (matplotlib.figure.Figure): La figura a guardar.
        path_base (str): Ruta base para el archivo sin extensión (incluyendo directorio).
        dpi (int): Resolución para el archivo PNG.
    """
    # Asegurarse de que el directorio de salida exista
    os.makedirs(os.path.dirname(path_base), exist_ok=True)

    # Guardar como PNG
    png_path = f"{path_base}.png"
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"Figura guardada en: {png_path}")

    # Guardar como SVG
    svg_path = f"{path_base}.svg"
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    print(f"Figura guardada en: {svg_path}")

def compute_pca_cumvar(X, max_components=200):
    """
    Calcula la varianza acumulada de PCA y los componentes para 90% y 95%.

    Args:
        X (np.ndarray): Matriz de datos (n_samples, n_features).
        max_components (int): Número máximo de componentes a calcular.

    Returns:
        tuple: (
            np.ndarray: Varianza explicada acumulada por componente.
            int: Número de componentes para alcanzar el 90% de varianza.
            int: Número de componentes para alcanzar el 95% de varianza.
        )
    """
    n_features = X.shape[1]
    n_components = min(max_components, n_features)

    pca = PCA(n_components=n_components)
    pca.fit(X)

    cum_var = np.cumsum(pca.explained_variance_ratio_)

    try:
        # +1 porque los índices son base 0
        n_at_90 = np.where(cum_var >= 0.90)[0][0] + 1
    except IndexError:
        n_at_90 = -1 # No alcanzado

    try:
        # +1 porque los índices son base 0
        n_at_95 = np.where(cum_var >= 0.95)[0][0] + 1
    except IndexError:
        n_at_95 = -1 # No alcanzado

    return cum_var, n_at_90, n_at_95

def compute_mutual_info_components(X, y, n_components=100):
    """
    Calcula la información mutua entre las etiquetas y los componentes principales de X.

    Args:
        X (np.ndarray): Matriz de datos (n_samples, n_features).
        y (np.ndarray): Etiquetas (n_samples,).
        n_components (int): Número de componentes PCA a utilizar.

    Returns:
        np.ndarray: Array con el valor de información mutua para cada componente.
    """
    n_features = X.shape[1]
    n_components = min(n_components, n_features)

    # Usar un PCA para proyectar los datos
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # Calcular información mutua en el espacio reducido
    mi_scores = mutual_info_classif(X_pca, y, discrete_features=False, random_state=42)

    return mi_scores

def _estimate_intrinsic_dim_twonn(X_sample):
    """Helper para estimar la dimensión intrínseca con el método TwoNN."""
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(-1, 1)

    # Encontrar los dos vecinos más cercanos (k=3 para incluir el punto mismo)
    nn = NearestNeighbors(n_neighbors=3, algorithm='auto', n_jobs=-1)
    nn.fit(X_sample)
    distances, _ = nn.kneighbors(X_sample)

    # d1 es la distancia al primer vecino, d2 al segundo.
    d1 = distances[:, 1]
    d2 = distances[:, 2]

    # Evitar división por cero o ratios inválidos
    valid_mask = (d1 > 0) & (d2 > 0)
    d1 = d1[valid_mask]
    d2 = d2[valid_mask]

    if len(d1) == 0:
        return np.nan

    # Ratio de distancias, debe ser > 1
    mu = d2 / d1
    mu = mu[mu > 1]

    if len(mu) == 0:
        return np.nan

    # CDF empírica de mu
    n = len(mu)
    mu_sorted = np.sort(mu)
    F_mu = np.arange(1, n + 1) / n

    # Ajuste lineal de log(1 - F(mu)) vs log(mu)
    x = np.log(mu_sorted)
    y = np.log(1 - F_mu)

    # Filtrar valores infinitos en y (cuando F_mu es 1)
    valid_indices = np.isfinite(y)
    if np.sum(valid_indices) < 2:
        return np.nan # No hay suficientes puntos para el ajuste

    x_fit = x[valid_indices]
    y_fit = y[valid_indices]

    # La pendiente es -d, por lo que d = -pendiente
    d_mle, _ = np.polyfit(x_fit, y_fit, 1)

    return -d_mle

def _calculate_participation_ratio(X_sample):
    """Helper para calcular el participation ratio a partir de los datos."""
    X_centered = X_sample - np.mean(X_sample, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)

    sum_eig = np.sum(eigenvalues)
    if sum_eig == 0:
        return 0.0

    # Fórmula del Participation Ratio
    pr = (sum_eig ** 2) / np.sum(eigenvalues ** 2)
    return pr

def estimate_intrinsic_dim(X_sample):
    """
    Estima la dimensión intrínseca utilizando dos métodos:
    - Participation Ratio (PR)
    - Maximum Likelihood Estimation (MLE) via Two-Nearest Neighbors (TwoNN)

    Args:
        X_sample (np.ndarray): Muestra de los datos (n_samples, n_features).

    Returns:
        tuple: (
            float: Estimación de la dimensión por Participation Ratio.
            float: Estimación de la dimensión por MLE (TwoNN).
        )
    """
    pr = _calculate_participation_ratio(X_sample)
    id_mle = _estimate_intrinsic_dim_twonn(X_sample)

    return pr, id_mle

if __name__ == '__main__':
    print("Ejecutando pruebas para las funciones de EDA...")
    from sklearn.datasets import make_classification

    # Crear datos de prueba sintéticos
    X, y = make_classification(
        n_samples=1000, n_features=100, n_informative=20,
        n_redundant=30, n_classes=5, random_state=42
    )

    # 1. Probar save_figure
    print("\n--- Probando save_figure ---")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Figura de prueba")
    temp_dir = "outputs/figures_temp"
    os.makedirs(temp_dir, exist_ok=True)
    save_figure(fig, os.path.join(temp_dir, "test_plot"))
    plt.close(fig)

    # 2. Probar compute_pca_cumvar
    print("\n--- Probando compute_pca_cumvar ---")
    cum_var, n90, n95 = compute_pca_cumvar(X, max_components=100)
    print(f"Longitud de varianza acumulada: {len(cum_var)}")
    print(f"Varianza final alcanzada: {cum_var[-1]:.2f}")
    print(f"Componentes para 90% de varianza: {n90}")
    print(f"Componentes para 95% de varianza: {n95}")

    # 3. Probar compute_mutual_info_components
    print("\n--- Probando compute_mutual_info_components ---")
    mi_scores = compute_mutual_info_components(X, y, n_components=50)
    print(f"Puntuaciones de MI obtenidas para {len(mi_scores)} componentes.")
    print(f"Primeras 5 puntuaciones: {mi_scores[:5]}")

    # 4. Probar estimate_intrinsic_dim
    print("\n--- Probando estimate_intrinsic_dim ---")
    X_sample = X[:500, :]
    pr, id_mle = estimate_intrinsic_dim(X_sample)
    print(f"Dimensión intrínseca (Participation Ratio): {pr:.2f}")
    print(f"Dimensión intrínseca (TwoNN MLE): {id_mle:.2f}")

    # Limpiar archivos de prueba
    shutil.rmtree(temp_dir)
    print("\nPruebas completadas y directorio temporal eliminado.")