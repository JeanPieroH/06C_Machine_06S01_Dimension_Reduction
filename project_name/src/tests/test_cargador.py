# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el módulo cargador.py.
"""

import unittest
import os
import shutil
import numpy as np
import pandas as pd
import sys

# Añadir la ruta raíz del proyecto al sys.path para permitir la importación de 'src'
# Se asume que la prueba se ejecuta desde la raíz del proyecto o que la estructura es reconocida.
ruta_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ruta_proyecto not in sys.path:
    sys.path.insert(0, ruta_proyecto)

from src.data import cargador

class TestCargador(unittest.TestCase):

    def setUp(self):
        """Configura un entorno de prueba limpio antes de cada test."""
        self.directorio_prueba = "directorio_temporal_de_prueba"
        self.directorio_raw = os.path.join(self.directorio_prueba, "raw")
        self.directorio_procesado = os.path.join(self.directorio_prueba, "processed")
        # No es necesario crear los directorios aquí, la función debe hacerlo.

    def tearDown(self):
        """Limpia el entorno de prueba después de cada test."""
        if os.path.exists(self.directorio_prueba):
            shutil.rmtree(self.directorio_prueba)

    def test_pipeline_completo_crea_archivos(self):
        """
        Verifica que ejecutar el pipeline completo crea todos los archivos esperados
        en los directorios correctos.
        """
        # Ejecutar la función a probar
        cargador.ejecutar_pipeline_carga(
            raw_out=self.directorio_raw,
            processed_out=self.directorio_procesado,
            overwrite=True
        )

        # Verificar que los archivos raw existen
        self.assertTrue(os.path.exists(os.path.join(self.directorio_raw, "train_X.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.directorio_raw, "train_Y.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.directorio_raw, "test_X.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.directorio_raw, "test_Y.npy")))

        # Verificar que los archivos procesados existen
        self.assertTrue(os.path.exists(os.path.join(self.directorio_procesado, "X_train.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.directorio_procesado, "X_test.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.directorio_procesado, "y_train.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.directorio_procesado, "y_test.csv")))

    def test_normalizacion_de_datos(self):
        """
        Verifica que los datos de imágenes procesados estén normalizados (entre 0 y 1).
        """
        cargador.ejecutar_pipeline_carga(
            raw_out=self.directorio_raw,
            processed_out=self.directorio_procesado,
            overwrite=True
        )

        x_train_procesado = np.load(os.path.join(self.directorio_procesado, "X_train.npy"))

        # Comprobar que los valores están en el rango [0, 1]
        self.assertLessEqual(x_train_procesado.max(), 1.0)
        self.assertGreaterEqual(x_train_procesado.min(), 0.0)

    def test_consistencia_de_dimensiones(self):
        """
        Verifica que el número de imágenes y etiquetas coincida después del procesamiento.
        """
        cargador.ejecutar_pipeline_carga(
            raw_out=self.directorio_raw,
            processed_out=self.directorio_procesado,
            overwrite=True
        )

        x_train_procesado = np.load(os.path.join(self.directorio_procesado, "X_train.npy"))
        y_train_df = pd.read_csv(os.path.join(self.directorio_procesado, "y_train.csv"))

        self.assertEqual(x_train_procesado.shape[0], len(y_train_df))

if __name__ == '__main__':
    # Esto permite ejecutar las pruebas directamente desde el script
    unittest.main(verbosity=2)