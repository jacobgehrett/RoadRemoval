import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal, ndimage

im = Image.open('Map_0009.tif')
im_array = np.array(im)
im_array = im_array[:, :, 0:3]


# Used many functions from Lab 3!


def to_gray_scale(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    image = 0.299 * r + 0.587 * g + 0.114 * b
    return image


# Convert image to greyscale for easier edge detection
gray_im = to_gray_scale(im_array)


def convolution(image, kernel):
    # Create a result buffer so that we don't affect the original image
    result = np.zeros(image.shape)
    for i in range(image.shape[1] - 2):
        for j in range(image.shape[0] - 2):
            result[j + 1, i + 1] = np.sum(np.multiply(
                image[j + 1 - math.floor(kernel.shape[0] / 2):j + 1 + math.ceil(kernel.shape[0] / 2),
                      i + 1 - math.floor(kernel.shape[1] / 2):i + 1 + math.ceil(kernel.shape[1] / 2)], kernel))
    return result


def edge_detect(image):
    mask = np.matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # image1 = convolution(image, mask)
    image1 = signal.fftconvolve(image, mask)
    mask = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # image2 = convolution(image, mask)
    image2 = signal.fftconvolve(image, mask)
    image = np.sqrt(np.square(image1) + np.square(image2))
    image = image / 8
    # Now we have completed the gradient magnitude, however I will return an image scaled for visibility below
    return image * 4


# Perform edge detection on image
edge_im = edge_detect(gray_im)

fig = plt.figure(figsize=(10, 7))

rows = 2
columns = 2

fig.add_subplot(rows, columns, 1)

median_im = ndimage.median_filter(edge_im, size=3)

plt.imshow(edge_im, cmap="Greys_r", vmin=0, vmax=255)

fig.add_subplot(rows, columns, 2)

plt.imshow(median_im, cmap="Greys_r", vmin=0, vmax=255)

plt.show()
