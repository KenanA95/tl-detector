import cv2
import numpy as np
from helpers import unpack_box
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import label


"""
    Heat map to track overlapping detections 
    Reference: https://github.com/mithi/vehicle-tracking-2
"""


class HeatMap:
    def __init__(self, shape, memory, threshold):
        self.map = np.zeros(shape, dtype=int)
        self.memory = memory
        self.threshold = threshold
        self.history = []

    def add(self, boxes):
        for box in boxes:
            x_min, x_max, y_min, y_max = unpack_box(box)
            self.map[y_min:y_max, x_min:x_max] += 1

    def remove(self, boxes):
        for box in boxes:
            x_min, x_max, y_min, y_max = unpack_box(box)
            self.map[y_min:y_max, x_min:x_max] -= 1

    def update(self, boxes):
        for box in boxes:
            # If we've reached our memory limit then remove the first box added
            if len(self.history) == self.memory:
                self.remove(self.history[0])
                self.history = self.history[1:]

            self.add([box])
            self.history.append([box])

    def get_labeled(self):
        threshold_map = self.map >= self.threshold
        labeled_map, num_labels = label(threshold_map)
        return labeled_map

    def show(self):
        plt.imshow(self.map, cmap='hot')
        plt.show()

    def draw(self, image, color=(0, 255, 0)):
        threshold_map = self.map >= self.threshold
        labeled_map, num_labels = label(threshold_map)

        for index in range(1, num_labels + 1):
            y_coords, x_coords = (labeled_map == index).nonzero()
            pt1 = x_coords.min(), y_coords.min()
            pt2 = x_coords.max(), y_coords.max()
            cv2.rectangle(image, pt1, pt2, color, thickness=2)
