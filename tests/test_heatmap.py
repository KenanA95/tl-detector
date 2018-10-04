import unittest
from heatmap import *


class TestHeatMap(unittest.TestCase):
    def test_add_boxes(self):
        heatmap = HeatMap(shape=(100, 100), memory=5, threshold=2)
        box_one = {'x_min': 10, 'x_max': 20, 'y_min': 10, 'y_max': 20}
        box_two = {'x_min': 50, 'x_max': 60, 'y_min': 50, 'y_max': 60}
        heatmap.add([box_one, box_two])

        # Check that the values are both present
        np.testing.assert_array_equal(heatmap.map[10:20, 10:20], 1)
        np.testing.assert_array_equal(heatmap.map[50:60, 50:60], 1)

        # Check the values outside the box have not been changed
        np.testing.assert_array_equal(heatmap.map[60:70, 60:70], 0)

    def test_remove_boxes(self):
        heatmap = HeatMap(shape=(100, 100), memory=5, threshold=2)

        # Add a box then remove it to make sure both functions work as expected
        box = {'x_min': 10, 'x_max': 20, 'y_min': 10, 'y_max': 20}
        heatmap.add([box])
        heatmap.remove([box])

        # Check that the values are both present
        np.testing.assert_array_equal(heatmap.map[10:20, 10:20], 0)
