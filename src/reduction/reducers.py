"""
Implementación de reductores de dimensionalidad y funciones auxiliares.
"""
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF, SparsePCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
import joblib
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

def obtener_reductor(metodo, **params):
    """Obtiene un objeto reductor según el método especificado."""
    # Corregir n_componentes a n_components para UMAP
    if metodo == 'umap' and 'n_componentes' in params:
        params['n_components'] = params.pop('n_componentes')
    
    reductores = {
        'pca': lambda p: PCA(**p),
        'nmf': lambda p: NMF(**p),
        'umap': lambda p: umap.UMAP(**p),
        'isomap': lambda p: Isomap(**p),
        'spectral': lambda p: SpectralEmbedding(**p),
        'sparsepca': lambda p: SparsePCA(**p),
        'lda': lambda p: LinearDiscriminantAnalysis(**p),
        'tsne': lambda p: TSNE(**p)
    }
    
    if metodo not in reductores:
        raise ValueError(f"Método {metodo} no soportado")
    
    return reductores[metodo](params)

def ajustar_reductor(X_train, metodo, n_componentes, semilla=42, **kwargs):
    """Ajusta un reductor a los datos de entrenamiento."""
    inicio = datetime.now()
    
    params = {
        'n_components': n_componentes,
        'random_state': semilla,
        **kwargs
    }
    
    reductor = obtener_reductor(metodo, **params)
    reductor.fit(X_train)
    
    tiempo_ajuste = (datetime.now() - inicio).total_seconds()
    
    meta = {
        'metodo': metodo,
        'n_componentes': n_componentes,
        'tiempo_ajuste': tiempo_ajuste,
        'params': params,
        'forma_entrada': X_train.shape,
        'marca_tiempo': datetime.utcnow().isoformat() + "Z"
    }
    
    return reductor, meta

def transformar_reductor(reductor_obj, X):
    """Transforma datos usando un reductor ajustado."""
    inicio = datetime.now()
    X_red = reductor_obj.transform(X)
    tiempo_transform = (datetime.now() - inicio).total_seconds()
    return X_red, tiempo_transform

def ajustar_transformar_viz(X_muestra, metodo='tsne', n_componentes=2, **kwargs):
    """Ajusta y transforma datos para visualización 2D."""
    if metodo == 'tsne':
        reductor = TSNE(n_components=n_componentes, **kwargs)
    else:
        reductor = obtener_reductor(metodo, n_components=n_componentes, **kwargs)
    
    return reductor.fit_transform(X_muestra)

def verificar_estabilidad(constructor_reductor, X_train, n_componentes, 
                         corridas=5, fraccion_muestra=0.8, semilla=42):
    """Verifica la estabilidad del método de reducción."""
    np.random.seed(semilla)
    n_muestras = int(len(X_train) * fraccion_muestra)
    embeddings = []
    
    for i in range(corridas):
        idx = np.random.choice(len(X_train), n_muestras, replace=False)
        X_muestra = X_train[idx]
        
        reductor = constructor_reductor(n_componentes=n_componentes, random_state=semilla+i)
        embedding = reductor.fit_transform(X_muestra)
        embeddings.append(embedding)
    
    # Calcular correlaciones entre embeddings
    correlaciones = []
    for i in range(corridas):
        for j in range(i+1, corridas):
            corr = np.corrcoef(embeddings[i].flatten(), embeddings[j].flatten())[0,1]
            correlaciones.append(corr)
    
    return np.mean(correlaciones)

def guardar_embedding(prefijo, X_train_red, X_test_red, meta):
    """Guarda los embeddings y metadatos."""
    os.makedirs(os.path.dirname(prefijo), exist_ok=True)
    
    np.save(f"{prefijo}__train.npy", X_train_red)
    if X_test_red is not None:
        np.save(f"{prefijo}__test.npy", X_test_red)
    
    with open(f"{prefijo}__meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

def guardar_reductor(prefijo, reductor_obj, meta):
    """Guarda el objeto reductor y sus metadatos."""
    os.makedirs(os.path.dirname(prefijo), exist_ok=True)
    joblib.dump(reductor_obj, f"{prefijo}.joblib")
    
    with open(f"{prefijo}__meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

def obtener_autoencoder(forma_entrada, dim_cuello, params):
    """Construye un autoencoder con arquitectura configurable."""
    entrada = Input(shape=(forma_entrada,))
    
    # Encoder
    x = entrada
    for unidades in params.get('capas_encoder', [128, 64]):
        x = Dense(unidades, activation='relu')(x)
    
    # Cuello de botella
    cuello = Dense(dim_cuello, activation='relu', name='embedding')(x)
    
    # Decoder
    x = cuello
    for unidades in reversed(params.get('capas_encoder', [128, 64])):
        x = Dense(unidades, activation='relu')(x)
    
    salida = Dense(forma_entrada, activation='sigmoid')(x)
    
    # Modelos
    autoencoder = Model(entrada, salida)
    encoder = Model(entrada, cuello)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder