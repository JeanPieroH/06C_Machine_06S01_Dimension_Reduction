# -*- coding: utf-8 -*-
"""
Módulo para la aplicación de técnicas de reducción de dimensionalidad.
"""
import os
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, NMF, SparsePCA
from sklearn.manifold import Isomap, TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from umap import UMAP

import tensorflow as tf
from tensorflow.keras import layers, models

# --- Función Factory para Reductores Clásicos ---

def obtener_reductor(metodo, n_componentes, semilla_aleatoria=42, **kwargs):
    """
    Factory que devuelve un objeto reductor de dimensionalidad de scikit-learn.

    Args:
        metodo (str): Nombre del método ('pca', 'nmf', 'umap', 'isomap', 'sparsepca', 'lda').
        n_componentes (int): Número de dimensiones a reducir.
        semilla_aleatoria (int): Semilla para métodos estocásticos.
        **kwargs: Argumentos adicionales para el constructor del reductor.

    Returns:
        object: Una instancia de un reductor de dimensionalidad.
    """
    metodos_disponibles = {
        'pca': PCA,
        'nmf': NMF,
        'umap': UMAP,
        'isomap': Isomap,
        'sparsepca': SparsePCA,
        'lda': LDA,
    }

    if metodo.lower() not in metodos_disponibles:
        raise ValueError(f"Método de reducción '{metodo}' no reconocido. Opciones: {list(metodos_disponibles.keys())}")

    clase_reductor = metodos_disponibles[metodo.lower()]

    # Parámetros comunes
    params = {'n_components': n_componentes}

    # Parámetros específicos del método
    if metodo in ['pca', 'nmf', 'sparsepca', 'umap']:
        params['random_state'] = semilla_aleatoria
    if metodo == 'isomap':
        params['n_jobs'] = -1

    # Sobrescribir con kwargs si es necesario
    params.update(kwargs)

    # LDA no usa 'random_state' en su constructor principal
    if metodo == 'lda':
        params.pop('random_state', None)

    return clase_reductor(**params)

# --- Implementación de Autoencoder ---

def construir_autoencoder(shape_entrada, dim_latente, capas_encoder=[128, 64], activacion='relu'):
    """
    Construye un modelo de autoencoder simple.

    Args:
        shape_entrada (tuple): Forma de los datos de entrada (ej. (784,)).
        dim_latente (int): Dimensión del cuello de botella (embedding).
        capas_encoder (list): Lista de neuronas en las capas ocultas del encoder.
        activacion (str): Función de activación para las capas ocultas.

    Returns:
        tuple: (encoder, decoder, autoencoder) modelos de Keras.
    """
    # Encoder
    entrada = layers.Input(shape=shape_entrada)
    x = entrada
    for neuronas in capas_encoder:
        x = layers.Dense(neuronas, activation=activacion)(x)
    cuello_botella = layers.Dense(dim_latente, activation=activacion, name='cuello_de_botella')(x)
    encoder = models.Model(entrada, cuello_botella, name='encoder')

    # Decoder
    entrada_decoder = layers.Input(shape=(dim_latente,))
    x = entrada_decoder
    for neuronas in reversed(capas_encoder):
        x = layers.Dense(neuronas, activation=activacion)(x)
    salida = layers.Dense(shape_entrada[0], activation='sigmoid')(x) # Sigmoid para salida en [0,1]
    decoder = models.Model(entrada_decoder, salida, name='decoder')

    # Autoencoder completo
    autoencoder = models.Model(entrada, decoder(encoder(entrada)), name='autoencoder')

    return encoder, decoder, autoencoder

# --- Funciones de Orquestación ---

def guardar_embedding(prefijo_ruta, x_train_red, x_test_red, meta):
    """Guarda los embeddings y metadatos."""
    os.makedirs(os.path.dirname(prefijo_ruta), exist_ok=True)
    np.save(f"{prefijo_ruta}__train.npy", x_train_red)
    np.save(f"{prefijo_ruta}__test.npy", x_test_red)
    with open(f"{prefijo_ruta}__meta.json", 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"Embeddings guardados con prefijo: {prefijo_ruta}")

def guardar_objeto_reductor(prefijo_ruta, reductor, meta):
    """Guarda el objeto reductor ajustado."""
    os.makedirs(os.path.dirname(prefijo_ruta), exist_ok=True)
    joblib.dump(reductor, f"{prefijo_ruta}.joblib")
    with open(f"{prefijo_ruta}__meta.json", 'w') as f:
        json.dump(meta, f, indent=4)
    print(f"Objeto reductor guardado en: {prefijo_ruta}.joblib")


def ajustar_transformar_y_guardar(datos_norm, metodo, n_componentes, config_exp):
    """
    Orquesta el pipeline completo de reducción para un método y n_componentes dados.

    Args:
        datos_norm (dict): Diccionario con 'X_train', 'X_test', 'y_train'.
        metodo (str): Método de reducción a aplicar.
        n_componentes (int): Dimensión del embedding.
        config_exp (dict): Configuración del experimento (nombre_norm, semilla, etc.).

    Returns:
        dict: Metadatos con información sobre la ejecución.
    """
    x_train, x_test = datos_norm['X_train'], datos_norm['X_test']
    nombre_norm = config_exp['nombre_normalizador']
    semilla = config_exp['semilla']

    # --- Ajuste y Transformación ---
    tiempo_inicio_fit = time.time()

    if metodo == 'autoencoder':
        shape_entrada = (x_train.shape[1],)
        encoder, _, autoencoder = construir_autoencoder(shape_entrada, n_componentes)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Usar EarlyStopping para evitar sobreajuste
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

        autoencoder.fit(x_train, x_train,
                        epochs=50,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        callbacks=callbacks,
                        verbose=0)

        reductor = encoder # El "reductor" es el encoder
        tiempo_fit = time.time() - tiempo_inicio_fit

        tiempo_inicio_transform = time.time()
        x_train_red = reductor.predict(x_train)
        x_test_red = reductor.predict(x_test)
        tiempo_transform = time.time() - tiempo_inicio_transform

        transform_soportado = True

    else: # Métodos de Scikit-learn
        reductor = obtener_reductor(metodo, n_componentes, semilla_aleatoria=semilla)

        # LDA es supervisado
        if metodo == 'lda':
            x_train_red = reductor.fit_transform(x_train, datos_norm['y_train'])
        else:
            x_train_red = reductor.fit_transform(x_train)

        tiempo_fit = time.time() - tiempo_inicio_fit

        # Medir tiempo de transformación
        tiempo_inicio_transform = time.time()
        # No todos los métodos soportan `transform` en nuevos datos (ej. Isomap pre-calculado)
        if hasattr(reductor, 'transform'):
            x_test_red = reductor.transform(x_test)
            transform_soportado = True
        else:
            # Si no hay transform, se reajusta en el test (no ideal, pero necesario para comparar)
            x_test_red = reductor.fit_transform(x_test)
            transform_soportado = False
        tiempo_transform = time.time() - tiempo_inicio_transform

    # --- Guardado de Artefactos ---

    meta_run = {
        'normalizador': nombre_norm,
        'metodo_reduccion': metodo,
        'n_componentes': n_componentes,
        'tiempo_fit_s': round(tiempo_fit, 4),
        'tiempo_transform_s': round(tiempo_transform, 4),
        'transform_soportado': transform_soportado,
        'memoria_estimada_mb': x_train_red.nbytes / (1024 * 1024)
    }

    # Guardar embeddings
    prefijo_embedding = f"data/processed/embeddings/{nombre_norm}__{metodo}__{n_componentes}"
    guardar_embedding(prefijo_embedding, x_train_red, x_test_red, meta_run)

    # Guardar objeto reductor
    if transform_soportado:
        prefijo_reductor = f"outputs/models/reductores/reductor__{nombre_norm}__{metodo}__{n_componentes}"
        if metodo == 'autoencoder':
            os.makedirs(os.path.dirname(prefijo_reductor), exist_ok=True)
            reductor.save(f"{prefijo_reductor}.keras")
        else:
            guardar_objeto_reductor(prefijo_reductor, reductor, meta_run)

    return meta_run