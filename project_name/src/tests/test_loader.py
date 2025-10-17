# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el módulo loader.py.
"""

import unittest
import os
import shutil
import numpy as np
import pandas as pd
import struct
from src.data import loader
from unittest.mock import patch

class TestLoader(unittest.TestCase):

    @patch('src.data.loader.gdown.download')
    def test_descargar_datos(self, mock_download):
        """Prueba que la función de descarga se llama con los argumentos correctos."""
        loader.descargar_datos('test_id', 'test_file')
        mock_download.assert_called_once_with('https://drive.google.com/uc?id=test_id', 'test_file', quiet=False)

    def setUp(self):
        """Configura un entorno de prueba limpio antes de cada test."""
        self.directorio_prueba = "directorio_temporal_de_prueba"
        os.makedirs(self.directorio_prueba, exist_ok=True)

        # Crear archivos IDX de prueba
        self.ruta_imagenes_prueba = os.path.join(self.directorio_prueba, "images.idx")
        self.ruta_etiquetas_prueba = os.path.join(self.directorio_prueba, "labels.idx")

        # Escribir datos de imagen de prueba (2 imágenes, 2x2 píxeles)
        with open(self.ruta_imagenes_prueba, 'wb') as f:
            f.write(struct.pack('>IIII', 2051, 2, 2, 2)) # magic, num_images, rows, cols
            f.write(np.array([1, 2, 3, 4, 255, 254, 253, 252], dtype=np.uint8).tobytes())

        # Escribir datos de etiqueta de prueba (2 etiquetas)
        with open(self.ruta_etiquetas_prueba, 'wb') as f:
            f.write(struct.pack('>II', 2049, 2)) # magic, num_labels
            f.write(np.array([5, 9], dtype=np.uint8).tobytes())

    def tearDown(self):
        """Limpia el entorno de prueba después de cada test."""
        if os.path.exists(self.directorio_prueba):
            shutil.rmtree(self.directorio_prueba)

    def test_leer_imagenes_idx(self):
        """Prueba que las imágenes IDX se lean correctamente."""
        imagenes = loader.leer_imagenes_idx(self.ruta_imagenes_prueba)
        self.assertEqual(imagenes.shape, (2, 2, 2))
        self.assertEqual(imagenes[1, 1, 1], 252)

    def test_leer_etiquetas_idx(self):
        """Prueba que las etiquetas IDX se lean correctamente."""
        etiquetas = loader.leer_etiquetas_idx(self.ruta_etiquetas_prueba)
        self.assertEqual(len(etiquetas), 2)
        self.assertEqual(etiquetas['label'][1], 9)

    def test_normalizar_y_guardar(self):
        """Prueba la normalización y el guardado de datos."""
        X_train = np.random.randint(0, 256, size=(10, 28, 28), dtype=np.uint8)
        y_train = pd.DataFrame({'label': np.random.randint(0, 10, size=10)})
        X_test = np.random.randint(0, 256, size=(5, 28, 28), dtype=np.uint8)
        y_test = pd.DataFrame({'label': np.random.randint(0, 10, size=5)})

        out_dir = os.path.join(self.directorio_prueba, "processed")
        loader.normalizar_y_guardar(X_train, X_test, y_train, y_test, out_dir=out_dir)

        # Verificar que los archivos se crearon
        self.assertTrue(os.path.exists(os.path.join(out_dir, "X_train.npy")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "y_train.csv")))

        # Verificar normalización
        X_train_cargado = np.load(os.path.join(out_dir, "X_train.npy"))
        self.assertLessEqual(X_train_cargado.max(), 1.0)
        self.assertGreaterEqual(X_train_cargado.min(), 0.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)