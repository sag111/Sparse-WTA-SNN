from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np


def clusterize(arrays, method='kmeans', params={}):
    # Преобразование изображений в плоский формат
    flattened_arrays = arrays.reshape(arrays.shape[0], -1)
    
    # Выбор метода кластеризации
    if method == 'kmeans':
        model = KMeans(**params)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(**params)
    elif method == 'spectral':
        model = SpectralClustering(**params)
    elif method == 'dbscan':
        model = DBSCAN(**params)
    else:
        raise ValueError("Неподдерживаемый метод кластеризации")
    
    labels = model.fit_predict(flattened_arrays)
    
    # Шумовые примеры в DBSCAN получают метку -1
    if method == 'dbscan':
        core_samples = labels != -1
        if np.sum(core_samples) > 1:  # Проверка на достаточное количество точек
            score = silhouette_score(flattened_arrays[core_samples], labels[core_samples])
        else:
            score = -1  # Недостаточно точек для оценки
    else:
        score = silhouette_score(flattened_arrays, labels)
    
    return labels, score
