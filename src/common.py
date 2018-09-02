import os
import cv2


def read_directory_images(path, extension, ):
    """ Read images from a directory based on file extension """

    for filename in os.listdir(path):
        if filename.endswith(extension):
            yield cv2.imread(os.path.join(path, filename))


def inbounds(image, x, y):
    return 0 <= x < image.shape[1] and 0 <= y < image.shape[0]


def cutoff_lower(image, percent):
    """ Remove the lower part of an image """
    y = image.shape[0]
    cutoff = int(y - (percent * y))
    return image[:cutoff, :]


def sliding_window(image, window_size, step_size):
    """ Run a sliding window across an image and extract each window """
    for y in range(0, image.shape[0] - (window_size[1] - step_size[1]), step_size[1]):
        for x in range(0,  image.shape[1] - (window_size[0] - step_size[0]), step_size[0]):
            window = image[y: y+window_size[1], x:x + window_size[0]]
            yield x, y, window


def extract_window(image, center, size):
    """ Extract a sub-matrix from an image around a given point. Center given as (x, y) coordinate """

    x_offset = int((size[0] - 1) / 2)
    y_offset = int((size[1] - 1) / 2)
    (x, y) = center

    min_x, max_x = x - x_offset,  x + x_offset + 1
    min_y, max_y = y - y_offset, y + y_offset + 1

    # Even-sized windows pad one pixel to the right and one below
    if size[0] % 2 == 0:
        max_x += 1
        max_y += 1

    # If it's out of bounds don't return a window
    if not (inbounds(image, min_x, min_y) and inbounds(image, max_x, max_y)):
        return None

    return image[min_y:max_y, min_x:max_x]
