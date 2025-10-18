"""
Módulo de normalización de datos.
Implementa varios normalizadores con API scikit-learn (fit/transform).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import cv2

class NormalizacionPorImagen(BaseEstimator, TransformerMixin):
    """Normaliza cada imagen independientemente centrando y escalando."""
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Asume X como (n_muestras, n_pixeles)
        X_norm = X.copy().astype(np.float32)
        medias = X_norm.mean(axis=1, keepdims=True)
        desv_std = X_norm.std(axis=1, keepdims=True) + self.epsilon
        X_norm = (X_norm - medias) / desv_std
        return X_norm

class BlanqueadoPCA(BaseEstimator, TransformerMixin):
    """Aplica PCA y normalización de varianza (whitening)."""
    
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.componentes_ = None
        self.media_ = None
        self.valores_singulares_ = None
    
    def fit(self, X, y=None):
        # Centrar los datos
        self.media_ = X.mean(axis=0, keepdims=True)
        X_centrado = X - self.media_
        
        # PCA usando SVD
        U, S, Vt = np.linalg.svd(X_centrado, full_matrices=False)
        self.componentes_ = Vt
        self.valores_singulares_ = S
        return self
    
    def transform(self, X):
        # Centrar usando la media del entrenamiento
        X_centrado = X - self.media_
        
        # Proyectar en componentes principales y normalizar varianza
        X_transformed = np.dot(X_centrado, self.componentes_.T)
        X_whitened = X_transformed / (self.valores_singulares_ + self.epsilon)
        return X_whitened

class NormalizadorCLAHE(BaseEstimator, TransformerMixin):
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    
    def __init__(self, forma_img=(28,28), clip_limit=2.0, grid_size=(8,8)):
        self.forma_img = forma_img
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Procesar cada imagen individualmente
        X_norm = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[0]):
            # Reshape a imagen 2D y convertir a uint8
            img = X[i].reshape(self.forma_img)
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Aplicar CLAHE
            img_clahe = self.clahe.apply(img_uint8)
            
            # Normalizar a [0,1] y aplanar
            X_norm[i] = img_clahe.flatten() / 255.0
        return X_norm

class NormaL2(BaseEstimator, TransformerMixin):
    """Normaliza cada vector a norma unitaria L2."""
    
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Calcular norma L2 por fila
        normas = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        # Normalizar, evitando división por cero
        X_norm = X / (normas + self.epsilon)
        return X_norm

def obtener_normalizador(nombre, forma_img=(28,28)):
    """Obtiene un normalizador por nombre.
    
    Args:
        nombre: Identificador del normalizador a usar
        forma_img: Forma de las imágenes para normalizadores que la requieran
    
    Returns:
        Objeto normalizador con API fit/transform
    """
    normalizadores = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
        'robust': RobustScaler(),
        'per_image': NormalizacionPorImagen(),
        'l2': NormaL2(),
        'pca_whiten': BlanqueadoPCA(),
        'clahe': NormalizadorCLAHE(forma_img=forma_img)
    }
    
    if nombre not in normalizadores:
        raise ValueError(f"Normalizador '{nombre}' no reconocido. Opciones válidas: {list(normalizadores.keys())}")
    
    return normalizadores[nombre]