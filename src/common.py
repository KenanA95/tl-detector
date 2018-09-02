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
