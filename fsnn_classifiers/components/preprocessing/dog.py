import numpy as np
from scipy.ndimage import convolve
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage


def dog_filter(image, sigma1=0.2, sigma2=0.5, k1=1, k2=1, radius=4):
    blur1 = gaussian_filter(image, sigma=sigma1)
    blur2 = gaussian_filter(image, sigma=sigma2)
    dog = k1*blur1 - k2*blur2
    return dog


# def gaussian_kernel(size, sigma):
#     """Создает гауссово ядро с заданным размером и стандартным отклонением."""
#     kernel = np.fromfunction(
#         lambda x, y: np.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / sigma**2),
#         (size, size)
#     )
#     # Нормируем ядро
#     # kernel /= np.sum(kernel)

#     return kernel


# def dog_filter(array_of_images, sigma1, sigma2, k1, k2, radius):
#     # Создаем center и surround ядра
#     center_kernel = gaussian_kernel(size=2*radius+1, sigma=sigma1)
#     surround_kernel = gaussian_kernel(size=2*radius+1, sigma=sigma2)

#     # Применяем фильтры
#     center_filtered_image = np.array([ndimage.convolve(image, center_kernel, mode='constant') for image in array_of_images])
#     surround_filtered_image = np.array([ndimage.convolve(image, surround_kernel, mode='constant') for image in array_of_images])

#     # Вычитаем фильтры, получая DoG
#     dog_filtered_images = k1 * center_filtered_image - k2 * surround_filtered_image

#     return dog_filtered_images


class DoG(BaseEstimator, TransformerMixin):
    def __init__(self, sigma1=1, sigma2=2, k1=1, k2=1, radius=4):
        '''
        sigma1 = 1, sigma2 = 2 for the on-center filter
        sigma1 = 2, sigma2 = 1 for the off-center filter
        '''
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.k1 = k1 
        self.k2 = k2 
        self.radius = radius  # Half-size of the kernel (radius determines the range of i and j)
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Проверка на возможность преобразования в квадратную матрицу
        side_length = int(np.sqrt(X.shape[-1]))

        if side_length ** 2 != X.shape[-1]:
            raise ValueError("The last dimension of X is not a perfect square, cannot reshape into a square matrix.")

        # Преобразование X в квадратные матрицы
        old_shape = X.shape
        X_reshaped = X.reshape((-1, side_length, side_length))
        
        # Применение свертки
        return dog_filter(X_reshaped, sigma1=self.sigma1, sigma2=self.sigma2, k1=self.k1, k2=self.k2, radius=self.radius).reshape(old_shape)
