# -*- coding: utf-8 -*-
"""
Pruebas unitarias para el módulo clasificacion.py.
"""

import unittest
import os
import shutil
import numpy as np
import joblib
import sys

# Añadir la ruta raíz del proyecto para importar módulos locales
ruta_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ruta_proyecto not in sys.path:
    sys.path.insert(0, ruta_proyecto)

from src.models import clasificacion
from sklearn.ensemble import RandomForestClassifier

class TestClasificacion(unittest.TestCase):

    def setUp(self):
        """Configura un entorno de prueba limpio."""
        self.directorio_prueba = "directorio_temporal_test_clasificacion"
        os.makedirs(self.directorio_prueba, exist_ok=True)
        # Datos de prueba simples y consistentes
        self.X_train = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [8, 8], [8, 9], [9, 9], [9, 10]])
        self.y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.X_test = np.array([[1, 0], [9, 8]])
        self.y_test = np.array([0, 1])

    def tearDown(self):
        """Limpia el entorno de prueba después de cada test."""
        if os.path.exists(self.directorio_prueba):
            shutil.rmtree(self.directorio_prueba)

    def test_obtener_clasificador(self):
        """Verifica que se obtiene una instancia correcta del clasificador."""
        clf = clasificacion.obtener_clasificador('rf', random_state=123)
        self.assertIsInstance(clf, RandomForestClassifier)
        self.assertEqual(clf.random_state, 123)

        # Probar que un clasificador no soportado lanza un error
        with self.assertRaises(ValueError):
            clasificacion.obtener_clasificador('clasificador_invalido')

    def test_entrenar_modelo(self):
        """Verifica que el modelo se entrena y devuelve el tiempo correcto."""
        clf = clasificacion.obtener_clasificador('svm')
        clf_ajustado, tiempo = clasificacion.entrenar_modelo(clf, self.X_train, self.y_train)

        self.assertIsNotNone(clf_ajustado)
        self.assertGreater(tiempo, 0)
        # Verificar que el modelo está "ajustado" (tiene atributos post-entrenamiento)
        self.assertTrue(hasattr(clf_ajustado, 'support_vectors_'))

    def test_evaluar_modelo(self):
        """Verifica que la evaluación produce métricas y una matriz de confusión."""
        clf = clasificacion.obtener_clasificador('knn', n_neighbors=1)
        clf.fit(self.X_train, self.y_train)

        metricas, matriz_conf = clasificacion.evaluar_modelo(clf, self.X_test, self.y_test)

        # En este caso simple, la predicción debería ser perfecta
        self.assertEqual(metricas['accuracy'], 1.0)
        self.assertEqual(metricas['f1_macro'], 1.0)

        matriz_esperada = np.array([[1, 0], [0, 1]])
        np.testing.assert_array_equal(matriz_conf, matriz_esperada)

    def test_guardar_y_cargar_modelo(self):
        """Verifica que un modelo se guarda y se puede cargar correctamente."""
        ruta_modelo = os.path.join(self.directorio_prueba, "test_modelo.joblib")
        clf_original = clasificacion.obtener_clasificador('logistic')
        clf_original.fit(self.X_train, self.y_train)

        clasificacion.guardar_modelo(ruta_modelo, clf_original)

        self.assertTrue(os.path.exists(ruta_modelo))

        # Cargar el modelo y verificar que funciona
        clf_cargado = joblib.load(ruta_modelo)
        y_pred = clf_cargado.predict(self.X_test)

        np.testing.assert_array_equal(y_pred, self.y_test)

if __name__ == '__main__':
    unittest.main(verbosity=2)