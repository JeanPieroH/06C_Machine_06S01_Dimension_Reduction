# -*- coding: utf-8 -*-
"""
Módulo con funciones de visualización para el Análisis Exploratorio de Datos (EDA).

Funciones:
- graficar_distribucion_clases: Genera un gráfico de barras de la distribución de clases.
- mostrar_rejilla_imagenes: Muestra una grilla de imágenes de ejemplo por clase.
- graficar_histograma_pixeles: Compara histogramas de intensidad de píxeles.
- calcular_imagenes_promedio_por_clase: Calcula la imagen promedio para cada clase.
- graficar_imagenes_promedio: Grafica las imágenes promedio.
- analizar_varianza_pca: Realiza un análisis de PCA y grafica la varianza acumulada.
- detectar_outliers_por_intensidad: Identifica imágenes con intensidad promedio fuera de un rango.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def graficar_distribucion_clases(y, out_path):
    """
    Crea y guarda un gráfico de barras con la distribución de clases.

    Args:
        y (pd.Series or np.ndarray): Etiquetas de las clases.
        out_path (str): Ruta para guardar el gráfico.
    """
    print(f"Generando gráfico de distribución de clases en '{out_path}'...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title('Distribución de Clases en el Conjunto de Datos')
    plt.xlabel('Clase')
    plt.ylabel('Frecuencia')
    plt.grid(axis='y', linestyle='--')

    # Guardar la figura
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print("Gráfico guardado con éxito.")

def mostrar_rejilla_imagenes(X, y, classes=None, n_per_class=5, out_path=None):
    """
    Muestra una rejilla con imágenes de ejemplo para varias clases.

    Args:
        X (np.ndarray): Array de imágenes (n_samples, height, width).
        y (np.ndarray): Etiquetas correspondientes.
        classes (list, optional): Lista de clases a mostrar. Si es None, se eligen al azar.
        n_per_class (int): Número de imágenes por clase.
        out_path (str, optional): Ruta para guardar la figura. Si es None, no se guarda.
    """
    print(f"Generando rejilla de imágenes de ejemplo...")
    if classes is None:
        classes = np.random.choice(np.unique(y), size=3, replace=False)

    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, n_per_class, figsize=(n_per_class * 2, num_classes * 2))
    fig.suptitle('Muestra de Imágenes por Clase', fontsize=16)

    for i, cls in enumerate(classes):
        idxs = np.where(y == cls)[0]
        sample_idxs = np.random.choice(idxs, n_per_class, replace=False)
        for j, idx in enumerate(sample_idxs):
            ax = axes[i, j]
            ax.imshow(X[idx], cmap='gray')
            ax.set_title(f'Clase: {cls}')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        print(f"Rejilla de imágenes guardada en '{out_path}'.")

    plt.close()

def graficar_histograma_pixeles(X_train, X_test, out_path):
    """
    Crea y guarda histogramas comparando la intensidad de píxeles para train y test.

    Args:
        X_train (np.ndarray): Datos de entrenamiento (normalizados).
        X_test (np.ndarray): Datos de prueba (normalizados).
        out_path (str): Ruta para guardar el gráfico.
    """
    print(f"Generando histograma de intensidad de píxeles en '{out_path}'...")
    plt.figure(figsize=(12, 6))

    plt.hist(X_train.ravel(), bins=50, color='blue', alpha=0.7, label='Entrenamiento', density=True)
    plt.hist(X_test.ravel(), bins=50, color='red', alpha=0.7, label='Prueba', density=True)

    plt.title('Distribución de Intensidad de Píxeles (Normalizada)')
    plt.xlabel('Intensidad de Píxel')
    plt.ylabel('Densidad')
    plt.legend()
    plt.grid(True, linestyle='--')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print("Histograma guardado con éxito.")

def calcular_imagenes_promedio_por_clase(X, y):
    """
    Calcula la imagen promedio para cada clase.

    Args:
        X (np.ndarray): Array de imágenes.
        y (np.ndarray): Etiquetas.

    Returns:
        tuple: Tupla con (imágenes promedio, etiquetas de clases).
    """
    print("Calculando imágenes promedio por clase...")
    unique_classes = np.unique(y)
    mean_images = []
    for cls in unique_classes:
        mean_image = np.mean(X[y == cls], axis=0)
        mean_images.append(mean_image)
    return np.array(mean_images), unique_classes

def graficar_imagenes_promedio(mean_images, labels, out_path):
    """
    Grafica las imágenes promedio por clase.

    Args:
        mean_images (np.ndarray): Array con las imágenes promedio.
        labels (np.ndarray): Etiquetas correspondientes.
        out_path (str): Ruta para guardar la figura.
    """
    print(f"Graficando imágenes promedio en '{out_path}'...")
    num_classes = len(labels)
    cols = min(num_classes, 5)
    rows = (num_classes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, (img, label) in enumerate(zip(mean_images, labels)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Promedio Clase: {label}')
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print("Imágenes promedio guardadas.")

def analizar_varianza_pca(X, n_components=50, out_path=None):
    """
    Realiza un análisis de PCA y grafica la varianza explicada acumulada.

    Args:
        X (np.ndarray): Datos de entrada (n_samples, n_features).
        n_components (int): Número de componentes para el PCA.
        out_path (str, optional): Ruta para guardar el gráfico.
    """
    print(f"Realizando análisis de PCA con {n_components} componentes...")
    # Aplanar las imágenes
    X_flat = X.reshape(X.shape[0], -1)

    pca = PCA(n_components=n_components)
    pca.fit(X_flat)

    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Análisis de Varianza Explicada por PCA')
    plt.grid(True, linestyle='--')

    # Línea para el 95% de varianza
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Varianza')
    plt.legend()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        print(f"Gráfico de PCA guardado en '{out_path}'.")

    plt.close()
    return pca

def detectar_outliers_por_intensidad(X, low_threshold=0.1, high_threshold=0.9):
    """
    Detecta imágenes cuya intensidad promedio está fuera de los umbrales.
    Se asume que X está normalizado entre 0 y 1.

    Args:
        X (np.ndarray): Array de imágenes normalizadas.
        low_threshold (float): Umbral inferior para la intensidad promedio.
        high_threshold (float): Umbral superior para la intensidad promedio.

    Returns:
        np.ndarray: Índices de los outliers detectados.
    """
    print("Buscando outliers por intensidad de píxeles...")
    mean_intensities = np.mean(X, axis=(1, 2))
    outliers = np.where((mean_intensities < low_threshold) | (mean_intensities > high_threshold))[0]

    if len(outliers) > 0:
        print(f"Se encontraron {len(outliers)} posibles outliers.")
    else:
        print("No se encontraron outliers por intensidad.")

    return outliers

def graficar_panel_embeddings(embeddings_dict, y_true, out_path):
    """
    Crea un panel con gráficos de dispersión para diferentes embeddings 2D.

    Args:
        embeddings_dict (dict): Diccionario donde las claves son los nombres de los
                                métodos y los valores son los arrays de embeddings (n_samples, 2).
        y_true (np.ndarray): Etiquetas verdaderas para colorear los puntos.
        out_path (str): Ruta para guardar la figura del panel.
    """
    print(f"Generando panel de visualización de embeddings en '{out_path}'...")

    num_plots = len(embeddings_dict)
    # Ajustar el número de columnas para que no sea demasiado ancho
    cols = min(num_plots, 3)
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    axes = axes.flatten()

    class_labels = np.unique(y_true)

    for i, (nombre_metodo, embedding) in enumerate(embeddings_dict.items()):
        ax = axes[i]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=y_true, cmap=plt.get_cmap('viridis', len(class_labels)), s=10, alpha=0.7)
        ax.set_title(f"Visualización 2D con {nombre_metodo.upper()}")
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.grid(True, linestyle='--', alpha=0.5)

        # Crear una leyenda discreta
        legend_handles = scatter.legend_elements(num=len(class_labels))
        ax.legend(handles=legend_handles[0], labels=list(class_labels), title="Clases")

    # Ocultar ejes no utilizados
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print("Panel de embeddings guardado con éxito.")