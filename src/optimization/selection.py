"""
Módulo de selección y optimización multi-criterio.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ..utils.helpers import guardar_figura

def compute_pareto_front(df, x='f1_macro', y='n_components', z='train_time_s'):
    """Calcula frente de Pareto para optimización multi-objetivo.
    
    Args:
        df: DataFrame con resultados
        x, y, z: Nombres de columnas para objetivos
        
    Returns:
        pd.DataFrame: Solo las filas que forman el frente de Pareto
    """
    def domina(row1, row2):
        """Determina si row1 domina a row2 (minimización)."""
        return all(row1 <= row2) and any(row1 < row2)
    
    # Normalizar valores para comparación justa
    scaler = StandardScaler()
    valores = df[[x, y, z]].values
    valores_norm = scaler.fit_transform(valores)
    
    # Encontrar puntos no dominados
    n_puntos = len(df)
    es_pareto = np.ones(n_puntos, dtype=bool)
    
    for i in range(n_puntos):
        for j in range(n_puntos):
            if i != j and domina(valores_norm[j], valores_norm[i]):
                es_pareto[i] = False
                break
    
    return df[es_pareto]

def find_knee_point(x, y):
    """Encuentra el punto de inflexión en una curva.
    
    Args:
        x, y: Arrays de coordenadas
        
    Returns:
        int: Índice del punto de inflexión
    """
    # Normalizar datos
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    # Calcular distancias a la línea diagonal
    coords = np.column_stack((x_norm, y_norm))
    punto_inicio = coords[0]
    punto_final = coords[-1]
    
    # Vector de la línea diagonal
    vec = punto_final - punto_inicio
    vec_norm = vec / np.linalg.norm(vec)
    
    # Distancias perpendiculares
    vec_perp = np.array([-vec_norm[1], vec_norm[0]])
    distancias = np.abs(np.dot(coords - punto_inicio, vec_perp))
    
    return int(np.argmax(distancias))

def bootstrap_ci(valores_metrica, alpha=0.05):
    """Calcula intervalo de confianza bootstrap.
    
    Args:
        valores_metrica: Array de valores de la métrica
        alpha: Nivel de significancia (default: 0.05)
        
    Returns:
        tuple: (límite_inferior, límite_superior)
    """
    percentil_inferior = alpha/2 * 100
    percentil_superior = (1 - alpha/2) * 100
    return np.percentile(valores_metrica, [percentil_inferior, percentil_superior])

def select_minimal_n_with_eps(df, eps=0.98, metrica='f1_macro'):
    """Selecciona modelos con menor dimensionalidad que mantienen rendimiento.
    
    Args:
        df: DataFrame con resultados
        eps: Umbral relativo al máximo (default: 0.98)
        metrica: Nombre de la métrica a optimizar
        
    Returns:
        pd.DataFrame: Modelos seleccionados
    """
    max_valor = df[metrica].max()
    umbral = eps * max_valor
    
    # Filtrar por umbral
    df_filtrado = df[df[metrica] >= umbral].copy()
    
    # Agrupar por método y normalización
    grupos = df_filtrado.groupby(['normalizer', 'method'])
    
    seleccionados = []
    for nombre, grupo in grupos:
        # Tomar el de menor dimensionalidad
        idx_min = grupo['n_components'].idxmin()
        seleccionados.append(df_filtrado.loc[idx_min])
    
    return pd.DataFrame(seleccionados)

def compare_class_performance(df_base, df_candidato, nombres_clases):
    """Compara rendimiento por clase entre dos modelos.
    
    Args:
        df_base: DataFrame con métricas del modelo base
        df_candidato: DataFrame con métricas del modelo candidato
        nombres_clases: Lista de nombres de clases
        
    Returns:
        pd.DataFrame: Comparación por clase
    """
    metricas = ['precision', 'recall', 'f1-score']
    comparacion = []
    
    for clase in nombres_clases:
        fila = {'clase': clase}
        
        for metrica in metricas:
            valor_base = df_base.loc[df_base['class'] == clase, metrica].iloc[0]
            valor_cand = df_candidato.loc[df_candidato['class'] == clase, metrica].iloc[0]
            
            fila[f'{metrica}_base'] = valor_base
            fila[f'{metrica}_candidato'] = valor_cand
            fila[f'{metrica}_diff'] = valor_cand - valor_base
            
        comparacion.append(fila)
    
    return pd.DataFrame(comparacion)

def plot_pareto_3d(df, x='f1_macro', y='n_components', z='train_time_s', 
                  highlight_selected=None):
    """Visualiza frente de Pareto en 3D.
    
    Args:
        df: DataFrame con resultados
        x,y,z: Nombres de columnas para ejes
        highlight_selected: Lista de índices para resaltar
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar todos los puntos
    scatter = ax.scatter(df[x], df[y], df[z], c='blue', alpha=0.6)
    
    # Resaltar seleccionados
    if highlight_selected is not None:
        df_sel = df.iloc[highlight_selected]
        ax.scatter(df_sel[x], df_sel[y], df_sel[z], 
                  c='red', s=100, label='Seleccionados')
    
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.title('Frente de Pareto 3D')
    
    if highlight_selected is not None:
        plt.legend()
    
    # Guardar
    ruta_base = 'outputs/figures/pareto_front_3d'
    guardar_figura(plt.gcf(), ruta_base)
    plt.close()
