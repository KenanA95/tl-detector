import os
import cv2
import time


def resize_images(images, new_size):
    return [cv2.resize(im, new_size) for im in images]


def read_directory_images(path, extension, n=None):
    """
    Read images from a directory based on file extension
    :param path: Path to the directory to read from
    :param extension: The image extension i.e '.png', '.jpg', etc.
    :param n: Number of files to read
    :return: Set of images from the directory
    """

    # Set n to the total number of files in the directory if no value is given
    n = n if n else len(os.listdir(path))

    for filename in os.listdir(path)[:n]:
        if filename.endswith(extension):
            yield cv2.imread(os.path.join(path, filename))


def validate_image_directory(path, extension):
    """ Clean up empty images in a directory. Happens surprisingly often when importing a dataset """
    for filename in os.listdir(path):
        if filename.endswith(extension):
            file_path = os.path.join(path, filename)
            if cv2.imread(file_path) is None:
                os.unlink(file_path)


def cutoff_lower(image, percent):
    """ Remove the lower part of an image """
    y = image.shape[0]
    cutoff = int(y - (percent * y))
    return image[:cutoff, :]


def sliding_window(image, win_size, step_size):
    """ Run a sliding window across an image and extract each window """
    for y in range(0, image.shape[0] - (win_size[1] - step_size[1]), step_size[1]):
        for x in range(0,  image.shape[1] - (win_size[0] - step_size[0]), step_size[0]):
            window = image[y: y+win_size[1], x:x + win_size[0]]
            yield x, y, window


def inbounds(image, x, y):
    return 0 <= x < image.shape[1] and 0 <= y < image.shape[0]


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


def timeit(func):
    """ Decorator to time how long functions take to run """
    def timed(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(func.__name__ + " Run time: {:2.2f} sec".format(end_time - start_time))
        return result

    return timed


def draw_boxes(image, coordinates, win_size):
    """
    Draw a list of boxes onto an image
    :param image: Image to draw on
    :param coordinates: List of (x, y) center coordinates
    :param win_size: How large the boxes should be
    :return:
    """
    x_offset = int((win_size[0] - 1) / 2)
    y_offset = int((win_size[1] - 1) / 2)

    for (x, y) in coordinates:
        if inbounds(image, x - x_offset, y - y_offset) and inbounds(image, x + x_offset, y + y_offset):
            cv2.rectangle(image, (x - x_offset, y - y_offset), (x + x_offset, y + y_offset), (0, 255, 0), 2)


def unpack_box(box):
    x_min = int(round(box['x_min']))
    x_max = int(round(box['x_max']))
    y_min = int(round(box['y_min']))
    y_max = int(round(box['y_max']))
    return x_min, x_max, y_min, y_max
