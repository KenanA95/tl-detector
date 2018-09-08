import unittest
from skimage import data
from src.features import *
from src.svm import *


class TestSVM(unittest.TestCase):

    def setUp(self):
        self.descriptor = HogDescriptor(win_size=(64, 128), block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8),
                                        orientations=9)
        self.clf = SVM(self.descriptor)

        self.images = [data.astronaut(), data.hubble_deep_field(), data.coffee()]
        self.images = [cv2.resize(im, (256, 256)) for im in self.images]

    def test_svm_train(self):
        """ Make sure the train function can run """

        res = self.clf.train(self.images, labels=[0, 0, 0])
        self.assertIsNotNone(res)

    def test_svm_predict(self):
        pass
