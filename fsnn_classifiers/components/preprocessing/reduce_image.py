from skimage.transform import resize as skimage_resize
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np


def resize_images(images, new_size=(32, 32), method='skimage', interpolation='bilinear'):
    """
    Массовое изменение размера изображений с выбором библиотеки.

    :param images: np.ndarray, массив изображений в формате (N, H*W) или (N, H, W)
    :param new_size: tuple, новый размер изображений (H_new, W_new)
    :param method: str, библиотека для изменения размера ('skimage', 'cv2', 'pillow', 'tf')
    :param interpolation: str, метод интерполяции:
        - для skimage: 'nearest', 'bilinear', 'bicubic', 'lanczos'
        - для cv2: 'nearest', 'linear', 'cubic', 'lanczos'
        - для pillow: 'nearest', 'bilinear', 'bicubic', 'lanczos'
        - для tf: 'nearest', 'bilinear', 'bicubic', 'lanczos3'
    :return: np.ndarray, массив измененных изображений
    """
    # Определяем исходный размер
    if images.ndim == 2:  # Преобразование из (N, H*W)
        side_length = int(np.sqrt(images.shape[1]))
        if side_length * side_length != images.shape[1]:
            raise ValueError("Размеры входных данных не соответствуют квадратным изображениям.")
        original_size = (side_length, side_length)
        images = images.reshape(-1, original_size[0], original_size[1])
    elif images.ndim == 3:  # Если уже (N, H, W)
        original_size = images.shape[1:3]
    else:
        raise ValueError("Ожидается массив размерности (N, H*W) или (N, H, W).")

    # Выбор библиотеки и интерполяции
    if method == 'skimage':
        interpolation_methods = {
            'nearest': 0,
            'bilinear': 1,
            'bicubic': 3,
            'lanczos': 5
        }
        order = interpolation_methods.get(interpolation, 1)

        resized_images = np.array([
            skimage_resize(image, new_size, mode='reflect', anti_aliasing=True, order=order)
            for image in images
        ])
    elif method == 'cv2':
        interpolation_methods = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        interp_flag = interpolation_methods.get(interpolation, cv2.INTER_LINEAR)

        resized_images = np.array([
            cv2.resize(image, new_size[::-1], interpolation=interp_flag)
            for image in images
        ])
    elif method == 'pillow':
        interpolation_methods = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        interp_flag = interpolation_methods.get(interpolation, Image.BILINEAR)

        resized_images = np.array([
            np.array(Image.fromarray(image).resize(new_size[::-1], interp_flag))
            for image in images
        ])
    elif method == 'tf':
        interpolation_methods = {
            'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            'bilinear': tf.image.ResizeMethod.BILINEAR,
            'bicubic': tf.image.ResizeMethod.BICUBIC,
            'lanczos3': tf.image.ResizeMethod.LANCZOS3
        }
        method_flag = interpolation_methods.get(interpolation, tf.image.ResizeMethod.BILINEAR)

        resized_images = np.array([
            tf.image.resize(image[np.newaxis, ..., np.newaxis], new_size, method=method_flag).numpy().squeeze()
            for image in images
        ])
    else:
        raise ValueError("Метод должен быть 'skimage', 'cv2', 'pillow' или 'tf'.")

    # Преобразуем обратно в одномерный формат
    resized_images_flat = resized_images.reshape(-1, new_size[0] * new_size[1])

    return resized_images_flat
