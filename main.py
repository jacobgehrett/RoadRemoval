import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal, ndimage

THRESHOLD = 50

im = Image.open('Map_0009.tif')
im_array = np.array(im)
im_array = im_array[:, :, 0:3]


# Used many functions from Labs 3 and 2!


def to_gray_scale(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    image = 0.299 * r + 0.587 * g + 0.114 * b
    return image


# Convert image to greyscale for easier edge detection
gray_im = to_gray_scale(im_array)


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

fig = plt.figure(figsize=(10, 10), num="Road Detection Attempt")

rows = 3
columns = 2

fig.add_subplot(rows, columns, 1, title="Raw Image")
plt.imshow(im_array, cmap="Greys_r", vmin=0, vmax=255)

fig.add_subplot(rows, columns, 2, title="Gray Image")
plt.imshow(gray_im, cmap="Greys_r", vmin=0, vmax=255)

fig.add_subplot(rows, columns, 3, title="Edge Detection")
plt.imshow(edge_im, cmap="Greys_r", vmin=0, vmax=255)

# Perform median filtering on image
median_im = ndimage.median_filter(edge_im, size=8)
fig.add_subplot(rows, columns, 4, title="Median Filtering")
plt.imshow(median_im, cmap="Greys_r", vmin=0, vmax=255)


def threshold(image):
    image = np.where(image > THRESHOLD, 255, 0)
    return image


# Perform thresholding on image
false_im = threshold(median_im)
fig.add_subplot(rows, columns, 5, title="False Coloring")
plt.imshow(false_im, cmap="Greys_r", vmin=0, vmax=255)

plt.show()


# At this point, I am realizing that it is very difficult to identify the man-made features from the naturally-formed
# ones, so I am trying a different approach. The script, "mainTwo.py" is my second attempt, in which I am using the
# MatPlotLib GUI to manually select points on the image to blend through. If I can get that working, I will come back to
# this.
