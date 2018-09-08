import cv2
import unittest
from skimage import data
from src.features import *


class TestFeatures(unittest.TestCase):

    def test_hog_descriptor(self):

        hog_descriptor = HogDescriptor(win_size=(64, 128), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8),
                                       orientations=9)
        image = data.astronaut()

        # Make sure it returns a feature vector
        self.assertIsNotNone(hog_descriptor.compute(image))

        # TODO: Add more output validation

    def test_lbp_descriptor(self):
        pass
