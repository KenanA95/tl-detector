import cv2
import unittest
import random
import tempfile
import numpy as np
from skimage import data
from src.helpers import *


class TestHelperMethods(unittest.TestCase):

    def test_resize_images(self):
        images = [data.astronaut(), data.hubble_deep_field(), data.coffee()]
        images = resize_images(images, (256, 256))
        [self.assertEqual(im.shape, (256, 256, 3)) for im in images]

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

    def test_read_directory_images(self):
        # Create a temporary directory
        test_dir = tempfile.mkdtemp()

        # Write a few images into it
        cv2.imwrite(test_dir + '/0.png', data.astronaut())
        cv2.imwrite(test_dir + '/1.png', data.coffee())
        cv2.imwrite(test_dir + '/2.png', data.camera())

        # Make sure all the images were read in and are not empty
        images = read_directory_images(test_dir, '.png', n=3)
        self.assertEqual(len(list(images)), 3)
        [self.assertIsNotNone(im.shape) for im in images]

        # Read in only the first two images
        images = read_directory_images(test_dir, '.png', n=2)
        self.assertEqual(len(list(images)), 2)
        [self.assertIsNotNone(im.shape) for im in images]
