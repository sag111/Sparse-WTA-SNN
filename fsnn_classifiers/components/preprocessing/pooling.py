import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.base import BaseEstimator, TransformerMixin

class Pooling(BaseEstimator, TransformerMixin):
    """
    Трансформер для предобработки изображений MNIST с помощью векторизованного пулинга.
    
    Parameters:
        window_sizes (tuple): Размеры окон для пулинга (по умолчанию (8, 6, 4)).
        stride (int): Шаг пулинга (по умолчанию 2).
        pool_type (str): Тип пулинга: 'max', 'avg' или 'min' (по умолчанию 'max').
    """
    
    def __init__(self, window_sizes=(8, 6, 4), stride=2, pool_type='max'):
        self.window_sizes = window_sizes
        self.stride = stride
        self.pool_type = pool_type
        
        # Проверка корректности pool_type
        if pool_type not in ['max', 'avg', 'min']:
            raise ValueError("pool_type должен быть 'max', 'avg' или 'min'")
    
    def _vectorized_pooling(self, image, window_size):
        """
        Векторизованный пулинг для одного изображения.
        """
        h, w = image.shape
        out_h = (h - window_size) // self.stride + 1
        out_w = (w - window_size) // self.stride + 1
        
        # Создание "view" с окнами
        shape = (out_h, out_w, window_size, window_size)
        strides = (self.stride * image.strides[0], self.stride * image.strides[1], 
                  image.strides[0], image.strides[1])
        windows = as_strided(image, shape=shape, strides=strides)
        
        # Применение пулинга
        if self.pool_type == 'max':
            pooled = windows.max(axis=(2, 3))
        elif self.pool_type == 'avg':
            pooled = windows.mean(axis=(2, 3))
        else:  # 'min'
            pooled = windows.min(axis=(2, 3))
        
        return pooled.flatten()
    
    def fit(self, X, y=None):
        """
        Подготовка трансформера (ничего не делает, так как нет параметров для обучения).
        """
        return self
    
    def transform(self, X):
        """
        Преобразование батча изображений MNIST.
        
        Parameters:
            X (numpy.ndarray): Входной массив размером (N, 784).
        
        Returns:
            numpy.ndarray: Преобразованный массив размером (N, 434).
        """
        # Преобразование входных данных в формат (N, 28, 28)
        images_2d = X.reshape(-1, 28, 28)
        N = images_2d.shape[0]
        
        # Список для результатов пулинга
        pooled_results = []
        
        # Применение пулинга для каждого размера окна
        for window_size in self.window_sizes:
            # Инициализация массива для результатов пулинга текущего размера окна
            out_h = (28 - window_size) // self.stride + 1
            out_w = (28 - window_size) // self.stride + 1
            pooled = np.zeros((N, out_h * out_w))
            
            # Обработка каждого изображения
            for i in range(N):
                pooled[i] = self._vectorized_pooling(images_2d[i], window_size)
            
            pooled_results.append(pooled)
        
        # Конкатенация результатов
        result = np.concatenate(pooled_results, axis=1)
        return result

# Пример использования:
if __name__ == "__main__":
    # Создание случайных данных для теста
    X = np.random.rand(10, 784)  # 10 изображений 28x28 в плоском виде
    
    # Инициализация трансформера
    transformer = Pooling(window_sizes=(8, 6, 4), stride=2, pool_type='max')
    
    # Применение трансформера
    X_transformed = transformer.fit_transform(X)
    print(f"Размер преобразованного массива: {X_transformed.shape}")