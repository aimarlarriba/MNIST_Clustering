# Práctica de Minería de Datos: Clustering y Reducción de Dimensionalidad sobre MNIST
![Python](https://img.shields.io/badge/python-3.14%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Este repositorio contiene una implementación completa en Python utilizando Jupyter Notebook para la asignatura de **Minería de Datos**. El proyecto aborda un problema clásico de aprendizaje automático no supervisado: agrupar y clasificar sin etiquetas previas las imágenes de dígitos escritos a mano del dataset **MNIST**.

---

## 📋 Descripción del Proyecto

El objetivo principal de esta práctica es aplicar, evaluar y comparar diferentes técnicas de reducción de dimensionalidad y modelos de agrupamiento (*clustering*), resolviendo el desafío inherente de alinear los clústeres descubiertos con las clases reales de los dígitos mediante optimización combinatoria.

---

## 🛠️ Pipeline y Metodología

El desarrollo del notebook sigue una estructura metodológica rigurosa dividida en las siguientes fases:

1. **Importación de Librerías y Entorno**
   * Configuración de las librerías estándar para manipulación numérica, análisis de datos, visualización y modelado (`NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `Scikit-Learn`, `SciPy`).

2. **Carga y Preparación de Datos**
   * Descarga automatizada del dataset `mnist_784` desde OpenML.
   * Normalización de los píxeles al rango [0, 1] y conversión de tipos de datos para optimizar el rendimiento computacional.
   * División estratificada de los datos reservando un conjunto de prueba independiente de 10,000 muestras para evaluar la generalización.

3. **Análisis Exploratorio de Datos (EDA)**
   * Verificación del balance de clases en el conjunto de entrenamiento.
   * Visualización gráfica de muestras representativas de cada dígito (del 0 al 9) para inspeccionar la variabilidad visual de los datos.

4. **Reducción de Dimensionalidad con PCA**
   * Evaluación experimental de múltiples dimensiones (desde 10 hasta las 784 originales) para encontrar el espacio latente óptimo que maximice el rendimiento del clustering sin perder información estructural relevante.

5. **Clustering No Supervisado (K-Means y GMM)**
   * Aplicación de **K-Means** probando diferentes valores de K (número de clústeres) y analizando métricas de calidad de agrupamiento.
   * Implementación de **Gaussian Mixture Models (GMM)** para modelar la densidad probabilística de los datos latentes.

6. **Alineación y Evaluación de Rendimiento**
   * Uso del **Algoritmo Húngaro** (`linear_sum_assignment`) para resolver el problema de correspondencia óptima entre los clústeres generados y las clases reales de los dígitos.
   * Cálculo del *Class-to-Cluster Accuracy* tanto en entrenamiento como en validación/prueba.
   * Análisis de métricas de validación interna:
     * **Coeficiente de Silueta**
     * **Índice de Calinski-Harabasz**
     * **Inercia del modelo**

7. **Visualización Avanzada e Inferencia**
   * Generación de matrices de confusión y heatmaps detallados para analizar las confusiones entre dígitos similares (por ejemplo, el 4 y el 9, o el 3 y el 5).
   * Representación gráfica bidimensional en 2D de las proyecciones espaciales.
   * Desarrollo de una función de inferencia *out-of-sample* orientada a evitar el *data leakage* al proyectar nuevas instancias.

---

## 🤖 Uso de Inteligencia Artificial

Durante el desarrollo de esta práctica, se han empleado herramientas de Inteligencia Artificial como apoyo en las siguientes tareas:

* **Estructuración y Refactoring:** Asistencia en la modularización del código dentro del notebook y estructuración limpia de los bloques de experimentación.
* **Depuración y Optimización:** Ayuda en la resolución de problemas relacionados con la aplicación del algoritmo húngaro para el mapeo de clústeres y la prevención de fugas de datos (*data leakage*) durante la inferencia en test.
* **Documentación:** Apoyo en la redacción técnica de este documento README para reflejar con claridad el rigor metodológico del proyecto.

---

## ⚙️ Requisitos e Instalación

Para ejecutar este notebook correctamente, asegúrate de tener instalado Python y las siguientes dependencias:

```bash
pip install numpy matplotlib seaborn pandas scikit-learn scipy
```

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
