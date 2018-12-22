import cv2
import unittest
from skimage import data
from src.features import *
from src.classifiers import *


class TestSVM(unittest.TestCase):

    def setUp(self):
        self.descriptor = HogDescriptor(block_size=(16, 16), cell_size=(8, 8), orientations=9)
        self.images = [data.astronaut(), data.hubble_deep_field(), data.coffee()]
        self.images = [cv2.resize(im, (256, 256)) for im in self.images]

    def test_svm_train(self):
        clf = SVM(self.descriptor)
        res = clf.train(self.images, labels=[0, 1, 0])
        self.assertIsNotNone(res)

    def test_svm_predict(self):
        clf = SVM(self.descriptor)
        clf.train(self.images, labels=[0, 1, 0])
        res = clf.predict_all(self.images)
        self.assertIsNotNone(res)
        self.assertEqual(len(res), 3)
        [self.assertTrue(pred == 0 or pred == 1) for pred in res]
