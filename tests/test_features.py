import cv2
import unittest
from skimage import data
from src.features import *


class TestFeatures(unittest.TestCase):
    
    def test_hog_multichannel(self):
        image = data.astronaut()
        hog_descriptor = cv2.HOGDescriptor()

        vertical_blocks = (image.shape[1] - hog_descriptor.winSize[0]) / hog_descriptor.blockStride[0]
        horizontal_blocks = (image.shape[0] - hog_descriptor.winSize[1]) / hog_descriptor.blockStride[1]
        blocks_per_image = (vertical_blocks + 1) * (horizontal_blocks + 1)
        feature_length = 3780 * blocks_per_image * 3

        res = hog_multichannel(image, hog_descriptor)
        self.assertEqual(len(res), int(feature_length))
