import cv2
import numpy as np
from helpers import unpack_box


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
