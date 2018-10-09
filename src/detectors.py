import cv2
import numpy as np
from matplotlib import pyplot as plt


class ColorDetector:
    def __init__(self):
        pass

    def compute_roi(self):
        pass

    def display_roi(self):
        pass

    @classmethod
    def from_config_file(cls, settings):
        return cls()


class SpotlightDetector:
    """ Identify the regions of interest in an image by
            1. Locating spotlights (bright spots) through top-hat morphology
            2. Perform a region growing algorithm (watershed) with the spotlights as the seeds
            3. Selecting the spotlights that do not grow too large """

    def __init__(self, threshold, max_size, kernel_size):
        self.threshold = threshold
        self.max_size = max_size
        self.kernel = np.ones((kernel_size, kernel_size), dtype=int)

    def compute_roi(self, image):
        """ Find the spotlights and perform region growing. Store markers that do not grow too large """

        # Perform top-hat morphology to find the spotlights in the image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tophat_image = cv2.morphologyEx(gray_image, cv2.MORPH_TOPHAT, self.kernel)
        ret, thresh = cv2.threshold(tophat_image, self.threshold, 255, cv2.THRESH_BINARY)

        # Watershed region growing algorithm with the spotlights as the seeds
        dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        ret, markers = cv2.connectedComponents(np.uint8(dist_transform))

        # Make sure the background is not 0
        markers += 1
        watershed_image = cv2.watershed(image, markers)

        # Grab the marker values and how many times they occur
        values, counts = np.unique(watershed_image, return_counts=True)

        # Get the indices of where the segments are under the max size
        segment_indices = np.where(counts <= self.max_size)
        markers = values[segment_indices]

        # Get the median coordinates of the markers (roughly the center)
        for marker in markers:
            y_coordinates, x_coordinates = np.where(watershed_image == marker)
            yield int(np.median(x_coordinates)), int(np.median(y_coordinates))

    def display_roi(self, image, window_size):
        """ Construct windows around the center of the identified ROI and display the image """
        mask = np.zeros(image.shape, dtype=np.uint8)
        x_offset = int((window_size[0] - 1) / 2)
        y_offset = int((window_size[1] - 1) / 2)

        for (x, y) in self.compute_roi(image):
            x_min, x_max = x - x_offset, x + x_offset
            y_min, y_max = y - y_offset, y + y_offset
            mask[y_min:y_max, x_min:x_max] = 1

        display_img = np.zeros(image.shape, dtype=np.uint8)
        display_img[mask == 1] = image[mask == 1]

        plt.imshow(display_img, cmap='gray')
        plt.show()

    @classmethod
    def from_config_file(cls, settings):
        return cls(int(settings['threshold']), int(settings['max_size']), int(settings['kernel_size']))
