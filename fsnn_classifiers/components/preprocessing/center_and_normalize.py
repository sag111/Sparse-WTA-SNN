import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass, shift, zoom

def center_and_normalize(image_1d, target_size=(20, 20)):
    # Convert the 1D image back to 2D
    image = image_1d.reshape((28, 28))

    # Calculate the center of mass
    cy, cx = center_of_mass(image)

    # Calculate the translation needed to center the image
    rows, cols = image.shape
    shift_x = np.round(cols/2.0 - cx).astype(int)
    shift_y = np.round(rows/2.0 - cy).astype(int)

    # Apply the translation using numpy indexing
    centered_image = np.roll(image, shift_y, axis=0)
    centered_image = np.roll(centered_image, shift_x, axis=1)

    # Find the bounding box of the digit after translation
    non_zero_coords = np.argwhere(centered_image > 0)
    top_left = np.min(non_zero_coords, axis=0)
    bottom_right = np.max(non_zero_coords, axis=0)

    # Crop the image to the bounding box
    digit = centered_image[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]

    # Resize the digit to the target size
    digit_resized = zoom(digit, (target_size[0] / digit.shape[0], target_size[1] / digit.shape[1]))

    return digit_resized.flatten()

def center_and_normalize_vec(images, target_size=(20, 20)):
    vec_func = np.vectorize(center_and_normalize, signature='(n)->(m)', excluded=['target_size'])
    return vec_func(images, target_size=target_size)


# def center_and_crop_digit(image_1d, target_size=(20, 20)):
#     image = image_1d.reshape((28, 28))
#     center_mass = np.array(np.unravel_index(np.argmax(image), image.shape))
#     top_left = center_mass - np.array(target_size) // 2
#     bottom_right = top_left + np.array(target_size)
#     cropped_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
#     cropped_image_flattened = cropped_image.flatten()
#     return cropped_image_flattened

# def center_and_crop_vec(images, target_size=(20, 20)):
#     vec_func = np.vectorize(center_and_crop_digit, signature='(n)->(m)')
#     return vec_func(images)