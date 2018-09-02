import unittest
import random
import numpy as np
from skimage import data
from src.common import *


class TestCommonMethods(unittest.TestCase):

    def test_inbounds(self):
        image = np.zeros((100, 100), int)
        self.assertTrue(inbounds(image, 0, 0))
        self.assertTrue(inbounds(image, 99, 99))
        self.assertFalse(inbounds(image, 100, 100))
        self.assertFalse(inbounds(image, 50, 100))

    def test_cutoff_lower(self):
        percent = random.uniform(0, 1)
        image = cutoff_lower(data.astronaut(), percent)
        self.assertEqual(image.shape[0], int(512 - (512 * percent)))

    def test_sliding_window(self):
        image = data.astronaut()
        window_size = (32, 64)
        step_size = (16, 16)

        # Check that the window size is correct every time
        for i, (x, y, window) in enumerate(sliding_window(image, window_size, step_size)):
            self.assertEqual((64, 32), window.shape[:2])

        total_windows = int((512-16) / step_size[0]) * int((512-48) / step_size[1])
        self.assertEqual(i+1, total_windows)

    def test_extract_window(self):
        image = data.astronaut()

        # Test odd-sized windows
        window = extract_window(image, center=(256, 256), size=(25, 25))
        self.assertEqual(window.shape, (25, 25, 3))

        # Test even-sized windows
        window = extract_window(image, center=(256, 256), size=(30, 30))
        self.assertEqual(window.shape, (30, 30, 3))

        # Test out of bounds windows
        window = extract_window(image, center=(500, 500), size=(30, 30))
        self.assertIsNone(window)

        window = extract_window(image, center=(10, 10), size=(30, 30))
        self.assertIsNone(window)
