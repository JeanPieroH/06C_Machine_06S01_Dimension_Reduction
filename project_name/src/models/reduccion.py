# -*- coding: utf-8 -*-
"""
Módulo para aplicar técnicas de reducción de dimensionalidad.
"""

import os
import time
import json
import numpy as np

from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from umap import UMAP

def _inicializar_reductor(metodo, n_componentes, random_state=42, **kwargs):
    """Inicializa una instancia de un modelo reductor."""
    if metodo == 'pca':
        return PCA(n_components=n_componentes, random_state=random_state, **kwargs)
    if metodo == 'nmf':
        return NMF(n_components=n_componentes, random_state=random_state, init='random', **kwargs)
    if metodo == 'umap':
        return UMAP(n_components=n_componentes, random_state=random_state, **kwargs)
    if metodo == 'isomap':
        return Isomap(n_components=n_componentes, **kwargs)
    if metodo == 'spectral':
        return SpectralEmbedding(n_components=n_componentes, random_state=random_state, **kwargs)
    if metodo == 'tsne':
        return TSNE(n_components=n_componentes, random_state=random_state, **kwargs)
    raise ValueError(f"Método de reducción '{metodo}' no soportado.")

def guardar_embedding(ruta_archivo, datos, metadatos):
    """Guarda un embedding y sus metadatos."""
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    np.save(ruta_archivo, datos)
    ruta_meta = ruta_archivo.replace('.npy', '_meta.json')
    with open(ruta_meta, 'w', encoding='utf-8') as f:
        json.dump(metadatos, f, ensure_ascii=False, indent=4)
    print(f"Embedding guardado en: {ruta_archivo}")

def ejecutar_reduccion(X_train, X_test, metodo, n_componentes, ruta_base_embeddings, **kwargs):
    """
    Ajusta un reductor en X_train, transforma ambos conjuntos y guarda los resultados.
    Maneja correctamente los métodos que no soportan transform para datos no vistos.
    """
    print(f"\n--- Procesando: {metodo.upper()} con n_components={n_componentes} ---")
    reductor = _inicializar_reductor(metodo, n_componentes, **kwargs)

    # Métodos que no tienen un `transform` fiable para datos fuera de la muestra
    metodos_sin_transform_fiable = ['isomap', 'spectral', 'tsne']

    # --- Procesamiento de Entrenamiento ---
    print(f"Ajustando y transformando datos de entrenamiento con {metodo.upper()}...")
    tiempo_inicio_fit_transform = time.time()
    X_train_reducido = reductor.fit_transform(X_train)
    tiempo_fin_fit_transform = time.time()

    meta_train = {
        'metodo': metodo,
        'n_componentes': n_componentes,
        'set': 'train',
        'tiempo_fit_transform_seg': round(tiempo_fin_fit_transform - tiempo_inicio_fit_transform, 4),
        'parametros': kwargs
    }
    ruta_train = os.path.join(ruta_base_embeddings, f"{metodo}_{n_componentes}_train.npy")
    guardar_embedding(ruta_train, X_train_reducido, meta_train)

    # --- Procesamiento de Prueba ---
    if metodo not in metodos_sin_transform_fiable:
        print(f"Transformando datos de prueba con {metodo.upper()}...")
        tiempo_inicio_transform = time.time()
        X_test_reducido = reductor.transform(X_test)
        tiempo_fin_transform = time.time()

        meta_test = {
            'metodo': metodo,
            'n_componentes': n_componentes,
            'set': 'test',
            'tiempo_transform_seg': round(tiempo_fin_transform - tiempo_inicio_transform, 4),
            'parametros': kwargs
        }
        ruta_test = os.path.join(ruta_base_embeddings, f"{metodo}_{n_componentes}_test.npy")
        guardar_embedding(ruta_test, X_test_reducido, meta_test)
    else:
        print(f"El método {metodo.upper()} no soporta 'transform' para datos no vistos. No se genera embedding de prueba.")

def ajustar_transformar_visualizacion(X_sample, metodo, n_componentes, **kwargs):
    """Aplica fit_transform para visualización (e.g., t-SNE)."""
    print(f"Ejecutando fit_transform con '{metodo}' para visualización (n_components=2)...")
    reductor = _inicializar_reductor(metodo, n_componentes, **kwargs)

    tiempo_inicio = time.time()
    embedding = reductor.fit_transform(X_sample)
    tiempo_fin = time.time()

    metadatos = {
        'metodo': metodo,
        'n_componentes': n_componentes,
        'tiempo_fit_transform_seg': round(tiempo_fin - tiempo_inicio, 4),
        'parametros': kwargs,
        'es_solo_visualizacion': True
    }
    print(f"Visualización generada en {metadatos['tiempo_fit_transform_seg']:.4f} segundos.")
    return embedding, metadatos