# -*- coding: utf-8 -*-
"""
Módulo con funciones de utilidad para el proyecto.
"""
import os
import matplotlib.pyplot as plt

def guardar_figura(fig, ruta_base, formatos=('png', 'svg'), dpi=300):
    """
    Guarda una figura de Matplotlib en los formatos especificados.

    Crea el directorio de salida si no existe.

    Args:
        fig (matplotlib.figure.Figure): La figura a guardar.
        ruta_base (str): La ruta base para el archivo sin extensión
                         (e.g., 'outputs/figures/mi_grafico').
        formatos (tuple): Una tupla de formatos de archivo (e.g., ('png', 'svg')).
        dpi (int): Puntos por pulgada para los formatos rasterizados como PNG.
    """
    dir_salida = os.path.dirname(ruta_base)
    if not os.path.exists(dir_salida):
        os.makedirs(dir_salida)
        print(f"Directorio creado: {dir_salida}")

    for fmt in formatos:
        ruta_salida = f"{ruta_base}.{fmt}"
        try:
            fig.savefig(ruta_salida, dpi=dpi, bbox_inches='tight')
            print(f"Figura guardada en: {ruta_salida}")
        except Exception as e:
            print(f"No se pudo guardar la figura en formato {fmt}: {e}")

    # Cerrar la figura para liberar memoria
    plt.close(fig)