import unittest
import numpy as np
from src.detectors import SpotlightDetector


class TestDetectors(unittest.TestCase):
    def setUp(self):
        # Create an empty synthetic image
        self.image = np.zeros((500, 500, 3), np.uint8)

        # Place 3 random 'small' spots
        random_locations = [(218, 365), (102, 401), (401, 98)]
        for (x, y) in random_locations:
            self.image[y-5:y+5, x-5:x+5, :] = 100

        # Place 3 random 'large' spots
        random_locations = [(256, 256), (98, 41), (435, 324)]
        for (x, y) in random_locations:
            self.image[y - 25:y + 25, x - 25:x + 25, :] = 100

    def tearDown(self):
        del self.image

    def test_spotlight_detector(self):
        # Have the spotlight detector find the small spots but not the big spots
        kernel = np.ones((15, 15), int)
        detector = SpotlightDetector(25, max_size=100, kernel=kernel)
        roi = list(detector.compute_roi(self.image))

        self.assertEqual(len(roi), 3)
