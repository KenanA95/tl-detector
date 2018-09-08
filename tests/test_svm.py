import unittest
from skimage import data
from src.features import *
from src.svm import *


class TestSVM(unittest.TestCase):

    def test_svm_train(self):
        """ Make sure the train function can run """

        descriptor = HogDescriptor(win_size=(64, 128), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8),
                                   orientations=9)

        clf = SVM(descriptor)

        images = [data.astronaut(), data.hubble_deep_field(), data.coffee()]
        images = [cv2.resize(im, (256, 256)) for im in images]
        labels = [0, 0, 0]

        res = clf.train(images, labels)
        self.assertIsNotNone(res)
