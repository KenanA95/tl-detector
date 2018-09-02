import unittest
import random
from skimage import data
from src.common import *


class TestCommonMethods(unittest.TestCase):

    def test_cutoff_lower(self):
        percent = random.uniform(0, 1)
        image = cutoff_lower(data.astronaut(), percent)
        self.assertEqual(image.shape[0], int(512 - (512 * percent)))

