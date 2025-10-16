# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el módulo reduccion.py (versión corregida).
"""

import unittest
import os
import shutil
import numpy as np
import sys

# Añadir la ruta raíz del proyecto para importar módulos locales
ruta_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ruta_proyecto not in sys.path:
    sys.path.insert(0, ruta_proyecto)

from src.models import reduccion

class TestReduccionCorregido(unittest.TestCase):

    def setUp(self):
        """Configura un entorno de prueba limpio."""
        self.directorio_prueba = "directorio_temporal_test_reduccion"
        os.makedirs(self.directorio_prueba, exist_ok=True)
        # Datos de prueba
        self.X_train = np.random.rand(50, 10)
        self.X_test = np.random.rand(10, 10)

    def tearDown(self):
        """Limpia el entorno de prueba después de cada test."""
        if os.path.exists(self.directorio_prueba):
            shutil.rmtree(self.directorio_prueba)

    def test_ejecutar_reduccion_con_transform_fiable(self):
        """
        Verifica que para PCA se generan embeddings de train y test.
        """
        metodo = 'pca'
        n_componentes = 5
        reduccion.ejecutar_reduccion(
            self.X_train, self.X_test, metodo, n_componentes, self.directorio_prueba
        )

        # Verificar que existen los archivos de train y test
        ruta_train = os.path.join(self.directorio_prueba, f"{metodo}_{n_componentes}_train.npy")
        ruta_test = os.path.join(self.directorio_prueba, f"{metodo}_{n_componentes}_test.npy")

        self.assertTrue(os.path.exists(ruta_train))
        self.assertTrue(os.path.exists(ruta_test))

        # Verificar dimensiones
        embedding_train = np.load(ruta_train)
        self.assertEqual(embedding_train.shape, (self.X_train.shape[0], n_componentes))

        embedding_test = np.load(ruta_test)
        self.assertEqual(embedding_test.shape, (self.X_test.shape[0], n_componentes))

    def test_ejecutar_reduccion_sin_transform_fiable(self):
        """
        Verifica que para Isomap SOLO se genera el embedding de train.
        """
        metodo = 'isomap'
        n_componentes = 2
        reduccion.ejecutar_reduccion(
            self.X_train, self.X_test, metodo, n_componentes, self.directorio_prueba, n_neighbors=5
        )

        # Verificar que existe el archivo de train
        ruta_train = os.path.join(self.directorio_prueba, f"{metodo}_{n_componentes}_train.npy")
        self.assertTrue(os.path.exists(ruta_train))

        # Verificar que NO existe el archivo de test
        ruta_test = os.path.join(self.directorio_prueba, f"{metodo}_{n_componentes}_test.npy")
        self.assertFalse(os.path.exists(ruta_test))

    def test_ajustar_transformar_visualizacion_sigue_funcionando(self):
        """
        Verifica que la función de visualización para t-SNE sigue operativa.
        """
        n_componentes = 2
        embedding, metadatos = reduccion.ajustar_transformar_visualizacion(
            self.X_train, 'tsne', n_componentes, perplexity=5
        )

        self.assertEqual(embedding.shape, (self.X_train.shape[0], n_componentes))
        self.assertTrue(metadatos['es_solo_visualizacion'])


if __name__ == '__main__':
    unittest.main(verbosity=2)