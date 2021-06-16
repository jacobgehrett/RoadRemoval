import copy

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

im = Image.open('Map_0009.tif')  # Open tif file
im_array = np.array(im)
im_array = im_array[:, :, 0:3]

# Initialize variables
line_array = []
j = 0
start_point = None
end_point = None
count = 0


# Class containing point objects' x and y integer values for array indexing
class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


# Lines consist of start and end points, and all points within calculated by point-slope formula
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


# Through Matplotlib GUI, extract start and end points
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


# Open colorized image in GUI
figure, axis = plt.subplots(1, num="Initial Road-Marking Procedure")
axis.imshow(im_array, cmap="Greys_r", vmin=0, vmax=255)
cid = figure.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Append points in line, and surrounding points. Surrounding points to include was decided arbitrarily.
totalPoints = []
for line in line_array:
    for point in line.points:
        totalPoints.append(point)
        totalPoints.append(Point(point.x - 1, point.y - 1))
        totalPoints.append(Point(point.x, point.y - 1))
        totalPoints.append(Point(point.x + 1, point.y - 1))
        totalPoints.append(Point(point.x - 1, point.y))
        totalPoints.append(Point(point.x + 1, point.y))
        totalPoints.append(Point(point.x - 1, point.y + 1))
        totalPoints.append(Point(point.x, point.y + 1))
        totalPoints.append(Point(point.x + 1, point.y + 1))

# Created mask arbitrarily as well
mask = np.matrix([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                  [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                  [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                  [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                  [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


# To perform convolution, image should be gray scaled
def to_gray_scale(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    image = 0.299 * r + 0.587 * g + 0.114 * b
    return image


gray_im = to_gray_scale(im_array)


# Function is a modified version of that used in Lab 3. Iterates through points within the totalPoints array.
def convolution(image, kernel):
    result = copy.deepcopy(gray_im)
    for p in totalPoints:
        result[p.y, p.x] = (np.sum(np.multiply(
            image[p.y + 1 - math.floor(kernel.shape[0] / 2):p.y + 1 + math.ceil(kernel.shape[0] / 2),
                  p.x + 1 - math.floor(kernel.shape[1] / 2):p.x + 1 + math.ceil(kernel.shape[1] / 2)], kernel))) / 96.0
    return result


blurredImage = convolution(gray_im, mask)

# Plot the three images used
fig = plt.figure(figsize=(10, 10), num="Road-b-Gone")

rows = 2
columns = 2

fig.add_subplot(rows, columns, 1, title="Raw Image")
plt.imshow(im_array, cmap="Greys_r", vmin=0, vmax=255)

fig.add_subplot(rows, columns, 2, title="Gray Image")
plt.imshow(gray_im, cmap="Greys_r", vmin=0, vmax=255)

fig.add_subplot(rows, columns, 3, title="Road Removed")
plt.imshow(blurredImage, cmap="Greys_r", vmin=0, vmax=255)

plt.show()

# At this point, several things are happening. First, the lines are generally off a few pixels. Not sure if
# this is because of the GUI or what, but it makes the blurring happen in the wrong areas often. Second, there is too
# much hard-coding going on. I should give the user the ability to specify the thickness of the road in pixels at the
# beginning and then adjust automatically. Finally, it doesn't look that great. I think I should make it so that the
# blurring happens more gradually, like a sort of fade, rather than a hard cut like we still see (even after blurring
# the road). Median filtering also may have worked a little better.
