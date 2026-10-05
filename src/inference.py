"""
Módulo de Inferencia y Mapeo Hungarian para Clustering No Supervisado sobre MNIST.
Permite proyectar nuevas instancias de imágenes sobre el espacio latente PCA y predecir
la clase real mediante la correspondencia biunívoca óptima resuelta con el Algoritmo Húngaro.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from typing import Dict, Tuple, Optional


def align_clusters_hungarian(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> Dict[int, int]:
    """
    Calcula el emparejamiento óptimo entre clústeres no supervisados y etiquetas reales
    minimizando el coste de desalineación mediante el Algoritmo Húngaro (Kuhn-Munkres).
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(max(10, k)))[:10, :k]
    # Invertir la matriz de contingencia para convertir maximización de coincidencias en minimización de coste
    cost_matrix = cm.max() - cm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return {cluster: true_class for true_class, cluster in zip(row_ind, col_ind)}


def predict_new_instance(
    image_raw: np.ndarray,
    pca_model,
    kmeans_model,
    mapping_dict: Dict[int, int]
) -> Tuple[int, int]:
    """
    Proyecta una nueva imagen sobre el espacio PCA existente y predice su clase dominante.
    Garantiza ausencia total de data leakage al reutilizar los transformadores pre-ajustados.
    
    Retorna: (cluster_id, predicted_digit)
    """
    if image_raw.ndim == 1:
        image_raw = image_raw.reshape(1, -1)
    
    # Proyección en el espacio latente
    x_latent = pca_model.transform(image_raw)
    
    # Asignación de clúster más cercano
    cluster_id = int(kmeans_model.predict(x_latent)[0])
    
    # Decodificación mediante el mapeo húngaro
    predicted_digit = mapping_dict.get(cluster_id, -1)
    return cluster_id, predicted_digit
