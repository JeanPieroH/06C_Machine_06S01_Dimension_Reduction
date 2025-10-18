"""
Funciones auxiliares para análisis y evaluación.
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time

def guardar_figura(fig, ruta_base):
    """Guarda figura matplotlib `fig` en {ruta_base}.png y {ruta_base}.svg

    Retorna tupla de rutas guardadas.
    """
    directorio = os.path.dirname(ruta_base)
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    ruta_png = f"{ruta_base}.png"
    ruta_svg = f"{ruta_base}.svg"
    fig.savefig(ruta_png, bbox_inches='tight')
    fig.savefig(ruta_svg, bbox_inches='tight')
    return ruta_png, ruta_svg

def vision_general_dataset(X_train, X_test, y_train, y_test, csv_salida=None):
    """Genera resumen del dataset con shapes, dtypes y estadísticas básicas."""
    resumen = {
        "X_train": {
            "shape": X_train.shape,
            "dtype": str(X_train.dtype),
            "min": float(X_train.min()),
            "max": float(X_train.max()),
            "mean": float(X_train.mean()),
            "std": float(X_train.std())
        },
        "X_test": {
            "shape": X_test.shape,
            "dtype": str(X_test.dtype),
            "min": float(X_test.min()),
            "max": float(X_test.max()),
            "mean": float(X_test.mean()),
            "std": float(X_test.std())
        }
    }
    
    if csv_salida:
        df = pd.DataFrame([{
            "dataset": nombre,
            **{f"{k}_{metrica}": v 
               for k, datos in resumen.items()
               for metrica, v in datos.items()}
        } for nombre in ["fashion_mnist"]])
        os.makedirs(os.path.dirname(csv_salida), exist_ok=True)
        df.to_csv(csv_salida, index=False)
    
    return resumen

def calcular_pca_varianza_acumulada(X, max_componentes=200):
    """Calcula varianza explicada acumulada de PCA."""
    n_caracteristicas = X.shape[1]
    n_comp = min(max_componentes, n_caracteristicas)
    pca = PCA(n_components=n_comp, svd_solver='full', whiten=False)
    pca.fit(X)
    var_acum = np.cumsum(pca.explained_variance_ratio_)
    return var_acum

def calcular_informacion_mutua_componentes(X, y, n_componentes=100, semilla=0):
    """Calcula información mutua entre cada característica original y la clase y.

    Retorna índices de las n_componentes principales características y puntajes MI como DataFrame.
    """
    # mutual_info_classif trabaja con arrays 2D
    puntajes = mutual_info_classif(X, y, random_state=semilla)
    idx = np.argsort(puntajes)[::-1]
    top = idx[:n_componentes]
    df = pd.DataFrame({"caracteristica": np.arange(X.shape[1])[top], "mi": puntajes[top]})
    return df

def contar_clases(y, csv_salida=None):
    """Retorna DataFrame con conteo de clases y opcionalmente guarda en CSV."""
    if isinstance(y, (pd.DataFrame, pd.Series)):
        valores = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    else:
        valores = pd.Series(y.ravel())
    df = valores.value_counts().sort_index().reset_index()
    df.columns = ['etiqueta', 'conteo']
    if csv_salida:
        os.makedirs(os.path.dirname(csv_salida), exist_ok=True)
        df.to_csv(csv_salida, index=False)
    return df

def cuadricula_muestras_eda(X, y, n_columnas=10, n_filas=5, forma_img=(28,28), generador=None, ruta_guardado=None, nombres_clases=None):
    """Grafica una cuadrícula de imágenes de muestra estratificadas por clase si es posible.

    X: array numpy (n_muestras, n_caracteristicas)
    y: etiquetas (array-like o DataFrame con columna de etiquetas)
    nombres_clases: diccionario opcional que mapea etiquetas numéricas a nombres de clase
    """
    if generador is None:
        generador = np.random.default_rng(0)
    n = n_columnas * n_filas
    idx = generador.choice(X.shape[0], size=min(n, X.shape[0]), replace=False)
    fig, axes = plt.subplots(n_filas, n_columnas, figsize=(n_columnas, n_filas))
    axes = axes.flatten()
    for ax, i in zip(axes, idx):
        img = X[i].reshape(forma_img)
        ax.imshow(img, cmap='gray')
        etiqueta = None
        try:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                etiqueta = y.iloc[i,0] if isinstance(y, pd.DataFrame) else y.iloc[i]
            else:
                etiqueta = y[i]
            if nombres_clases is not None:
                etiqueta = nombres_clases.get(etiqueta, str(etiqueta))
        except Exception:
            etiqueta = ''
        ax.set_title(str(etiqueta), fontsize=6)
        ax.axis('off')
    for ax in axes[len(idx):]:
        ax.axis('off')
    fig.tight_layout()
    if ruta_guardado:
        guardar_figura(fig, ruta_guardado)
    return fig

def histograma_pixeles_eda(X, bins=50, ruta_guardado=None):
    """Grafica histograma de intensidades de píxeles (aplanado sobre el conjunto de datos)."""
    valores = X.ravel()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(valores, bins=bins)
    ax.set_title('Pixel Intensity Distribution')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    if ruta_guardado:
        guardar_figura(fig, ruta_guardado)
    return fig

def imagenes_promedio_eda(X, y, forma_img=(28,28), directorio_salida=None, nombres_clases=None):
    """Calcula imagen promedio por clase y guarda figuras.
    
    nombres_clases: diccionario opcional que mapea etiquetas numéricas a nombres de clase
    """
    if isinstance(y, (pd.DataFrame)):
        etiquetas = y.iloc[:,0].values
    elif isinstance(y, (pd.Series)):
        etiquetas = y.values
    else:
        etiquetas = np.array(y).ravel()
    
    unicas = np.unique(etiquetas)
    rutas = {}
    
    for etiq in unicas:
        idx = np.where(etiquetas==etiq)[0]
        img_promedio = X[idx].mean(axis=0).reshape(forma_img)
        fig, ax = plt.subplots(figsize=(3,3))
        ax.imshow(img_promedio, cmap='gray')
        titulo = nombres_clases.get(etiq, f'Class {etiq}') if nombres_clases is not None else f'Class {etiq}'
        ax.set_title(f'Average {titulo}')
        ax.axis('off')
        if directorio_salida:
            os.makedirs(directorio_salida, exist_ok=True)
            nombre_clase = nombres_clases.get(etiq, str(etiq)) if nombres_clases is not None else str(etiq)
            # Sanitizar el nombre del archivo
            nombre_archivo = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in nombre_clase.lower())
            base = os.path.join(directorio_salida, f'class_average_{nombre_archivo}')
            guardar_figura(fig, base)
            rutas[etiq] = (f'{base}.png', f'{base}.svg')
        else:
            rutas[etiq] = None
    return rutas

def pca_n_en_umbral(X, umbrales=(0.9, 0.95), max_componentes=200):
    """Calcula número de componentes necesarios para alcanzar umbral de varianza explicada."""
    var_acum = calcular_pca_varianza_acumulada(X, max_componentes=max_componentes)
    resultados = {}
    for u in umbrales:
        # primer índice donde var_acum >= u
        idx = int(np.searchsorted(var_acum, u)) + 1
        resultados[f'n_en_{int(u*100)}'] = idx
    resultados['var_acum'] = var_acum
    return resultados

def estimar_dimension_intrinseca(X, max_componentes=200):
    """Calcula estimaciones de dimensión intrínseca:
    - ratio de participación de valores propios PCA
    - n_en_90/95 de varianza explicada PCA
    """
    n_caracteristicas = X.shape[1]
    n_comp = min(max_componentes, n_caracteristicas)
    pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=0)
    pca.fit(X)
    vp = pca.explained_variance_
    # ratio de participación
    rp = (vp.sum()**2) / (np.sum(vp**2) + 1e-12)
    conteos_pca = pca_n_en_umbral(X, umbrales=(0.9,0.95), max_componentes=max_componentes)
    return {
        "ratio_participacion": float(rp),
        "n_en_90": int(conteos_pca['n_en_90']),
        "n_en_95": int(conteos_pca['n_en_95'])
    }

def evaluar_normalizadores(X_train, X_test, y_train, y_test, lista_normalizadores=None, 
                         submuestra=5000, semilla=0, csv_salida=None, fig_salida=None):
    """Evaluación rápida de normalizadores en submuestra estratificada.
    
    Usa pipeline PCA(10)+RegresionLogistica para comparar rendimiento.
    Retorna DataFrame con normalizador, f1_macro, precisión, tiempo_seg, notas.
    """
    if lista_normalizadores is None:
        lista_normalizadores = ['minmax','standard','robust','per_image','l2','pca_whiten','clahe']

    # Aplanar y
    if isinstance(y_train, (pd.DataFrame, pd.Series)):
        valores_y = y_train.iloc[:,0].values
    else:
        valores_y = np.array(y_train).ravel()

    # submuestra estratificada
    n_muestras = min(submuestra, X_train.shape[0])
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_muestras, random_state=semilla)
    for idx_train, _ in sss.split(X_train, valores_y):
        idx = idx_train
        break
    X_sub = X_train[idx]
    y_sub = valores_y[idx]

    resultados = []
    for nombre in lista_normalizadores:
        t0 = time.time()
        try:
            from src.data.normalizers import obtener_normalizador
            norm = obtener_normalizador(nombre, forma_img=(28,28))
            # ajustar/transformar en submuestra
            X_tr = norm.fit_transform(X_sub)
            X_te = norm.transform(X_test[:min(2000, X_test.shape[0])]) if hasattr(norm, 'transform') else norm.fit_transform(X_test[:min(2000, X_test.shape[0])])
            
            # pipeline rápido: PCA 10 + regresión logística
            pca = PCA(n_components=min(10, X_tr.shape[1]), svd_solver='randomized', random_state=0)
            pca.fit(X_tr)
            X_tr_p = pca.transform(X_tr)
            X_te_p = pca.transform(X_te)
            clf = LogisticRegression(max_iter=200, solver='liblinear', multi_class='ovr')
            clf.fit(X_tr_p, y_sub)
            y_pred = clf.predict(X_te_p)
            
            # y_test puede ser DataFrame
            if isinstance(y_test, (pd.DataFrame, pd.Series)):
                valores_test = y_test.iloc[:,0].values[:len(y_pred)]
            else:
                valores_test = np.array(y_test).ravel()[:len(y_pred)]
            
            f1 = float(f1_score(valores_test, y_pred, average='macro'))
            precision = float(accuracy_score(valores_test, y_pred))
            notas = ''
        except Exception as e:
            f1 = None
            precision = None
            notas = str(e)
        t = time.time() - t0
        resultados.append({
            'normalizador': nombre,
            'f1_macro': f1,
            'precision': precision,
            'tiempo_seg': t,
            'notas': notas
        })

    df = pd.DataFrame(resultados).sort_values(by='f1_macro', ascending=False)
    if csv_salida:
        os.makedirs(os.path.dirname(csv_salida), exist_ok=True)
        df.to_csv(csv_salida, index=False)

    if fig_salida:
        fig, ax = plt.subplots(figsize=(8,4))
        df_grafico = df.copy()
        df_grafico['f1_grafico'] = df_grafico['f1_macro'].fillna(0)
        ax.bar(df_grafico['normalizador'], df_grafico['f1_grafico'])
        ax.set_ylabel('F1_macro')
        ax.set_title('Comparación de normalizadores')
        plt.xticks(rotation=45)
        guardar_figura(fig, fig_salida)

    return df

def elegir_normalizadores_y_guardar(df_resultados, top_k=2, 
                                  meta_salida='data/processed/normalizadores_elegidos.json',
                                  preferir_no_negativo='minmax'):
    """Elige los top-k normalizadores por f1_macro y guarda metadatos."""
    # eliminar filas con f1 None
    df = df_resultados.copy()
    df = df[df['f1_macro'].notnull()]
    df = df.sort_values(by='f1_macro', ascending=False)
    elegidos = df['normalizador'].tolist()[:top_k]
    
    # asegurar presencia de preferir_no_negativo
    if preferir_no_negativo not in elegidos and preferir_no_negativo in df['normalizador'].values:
        if len(elegidos) < top_k:
            elegidos.append(preferir_no_negativo)
        else:
            elegidos[-1] = preferir_no_negativo

    meta = {
        'elegidos': elegidos,
        'marca_tiempo': datetime.utcnow().isoformat() + 'Z',
        'ruta_df_resultados': None
    }
    os.makedirs(os.path.dirname(meta_salida), exist_ok=True)
    with open(meta_salida, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta

def calcular_candidatos_dimension(X, info_pca=None, df_mi=None, est_intrinseca=None,
                               csv_salida='outputs/tables/candidatos_dimension.csv'):
    """Combina múltiples métodos para proponer dimensiones candidatas."""
    n_caracteristicas = X.shape[1]
    candidatos = set()
    
    if info_pca is not None:
        for k in ('n_en_90','n_en_95'):
            v = info_pca.get(k)
            if v is not None:
                candidatos.add(int(max(2, min(n_caracteristicas, int(v)))))

    if est_intrinseca is not None:
        rp = est_intrinseca.get('ratio_participacion')
        if rp is not None:
            candidatos.add(int(max(2, min(n_caracteristicas, round(rp)))))
        for k in ('n_en_90','n_en_95'):
            v = est_intrinseca.get(k)
            if v is not None:
                candidatos.add(int(max(2, min(n_caracteristicas, int(v)))))

    if df_mi is not None and not df_mi.empty:
        mi_ordenado = df_mi.sort_values('mi', ascending=False)
        valores_mi = mi_ordenado['mi'].values
        if valores_mi.sum() > 0:
            acum = np.cumsum(valores_mi)
            fraccion = acum / acum[-1]
            # primer índice donde fraccion >= 0.9
            idx90 = int(np.searchsorted(fraccion, 0.9)) + 1
            candidatos.add(int(max(2, min(n_caracteristicas, idx90))))

            # también probar un límite superior fijo
            candidatos.add(int(min(50, n_caracteristicas)))

    # agregar otra opción pequeña
    pequeno = max(5, max(candidatos)//2 if candidatos else n_caracteristicas//2)
    if pequeno <= n_caracteristicas:
        candidatos.add(pequeno)

    # ordenar y filtrar valores válidos
    lista_candidatos = sorted([int(x) for x in candidatos if x >= 2 and x <= n_caracteristicas])
    
    # guardar resultados
    os.makedirs(os.path.dirname(csv_salida), exist_ok=True)
    pd.DataFrame({'dimension_candidata': lista_candidatos}).to_csv(csv_salida, index=False)
    
    return lista_candidatos

def guardar_json(obj, ruta):
    """Guarda objeto como JSON con formato agradable."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)