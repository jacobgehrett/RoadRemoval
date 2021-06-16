import copy

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

im = Image.open('Map_0009.tif')  # Open tif file
im_array = np.array(im)
im_array = im_array[:, :, 0:3]

line_array = []
j = 0
start_point = None
end_point = None
count = 0


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.points = []
        self.get_points()

    def get_points(self):
        global im_array
        global count

        point_array = [self.start]

        min_x = np.min([self.start.x, self.end.x])
        max_x = np.max([self.start.x, self.end.x])
        if min_x == self.end.x:
            self.start, self.end = self.end, self.start

        for i in range(int(min_x), int(max_x)):
            count += 1
            point_array.append(Point(i,
                                     self.start.y + (self.end.y - self.start.y) / (self.end.x - self.start.x) * count))

        self.points = point_array


def onclick(event):
    global j
    global line_array
    global start_point
    global end_point
    ix, iy = event.xdata, event.ydata

    if (ix is not None) and (ix > 1) and (iy > 1):  # Make sure click is within plot
        j += 1

        if j % 2 == 1:
            start_point = Point(ix, iy)
        else:
            end_point = Point(ix, iy)
            line_array.append(Line(start_point, end_point))
            axis.plot([start_point.x, end_point.x], [start_point.y, end_point.y])
            plt.draw_all()
            plt.pause(0.001)
            plt.draw_all()
            plt.pause(0.001)


figure, axis = plt.subplots(1, num="Initial Road-Marking Procedure")
axis.imshow(im_array, cmap="Greys_r", vmin=0, vmax=255)
cid = figure.canvas.mpl_connect('button_press_event', onclick)
plt.show()

totalPoints = []
for line in line_array:
    for point in line.points:
        totalPoints.append(point)

mask = np.matrix([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

blurredImage = copy.deepcopy(im_array)


def convolution(image, kernel):
    # Create a result buffer so that you don't affect the original image
    result = np.zeros(image.shape)
    for i in range(image.shape[1] - 2):
        for j in range(image.shape[0] - 2):
            result[j + 1, i + 1] = np.sum(np.multiply(
                image[j + 1 - math.floor(kernel.shape[0] / 2):j + 1 + math.ceil(kernel.shape[0] / 2),
                i + 1 - math.floor(kernel.shape[1] / 2):i + 1 + math.ceil(kernel.shape[1] / 2)], kernel))
    return result


for point in totalPoints:
    blurredImage[int(point.y), int(point.x)] = \
        (im_array[int(point.y) - 1, int(point.x) - 1] + im_array[int(point.y) - 1, int(point.x)] +
         im_array[int(point.y) - 1, int(point.x) + 1] + im_array[int(point.y), int(point.x) - 1] +
         im_array[int(point.y), int(point.x) + 1] + im_array[int(point.y) + 1, int(point.x) - 1] +
         im_array[int(point.y) + 1, int(point.x)] + im_array[int(point.y) + 1, int(point.x)] + 1) / 8

fig = plt.figure(figsize=(10, 10), num="Road-b-Gone")

rows = 1
columns = 2

fig.add_subplot(rows, columns, 1, title="Raw Image")
plt.imshow(im_array, cmap="Greys_r", vmin=0, vmax=255)

fig.add_subplot(rows, columns, 2, title="Road Removed")
plt.imshow(blurredImage, cmap="Greys_r", vmin=0, vmax=255)

plt.show()
