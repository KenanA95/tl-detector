from sklearn import linear_model
from src.helpers import sliding_window, extract_window


class SVM:
    def __init__(self, descriptor):
        """
        Linear SVM trained with stochastic gradient descent (SGD)
        :param descriptor: Feature Descriptor Object that converts images into vectors
        """
        self.clf = linear_model.SGDClassifier()
        self.descriptor = descriptor

    def train(self, images, labels):
        features = [self.descriptor.compute(im) for im in images]
        self.clf.fit(features, labels)
        return self.clf

    def predict(self, image):
        fd = self.descriptor.compute(image).reshape(1, -1)
        return self.clf.predict(fd)[0]

    def predict_all(self, images):
        return [self.predict(image) for image in images]

    def run_sliding_window(self, image, win_size, step_size):
        for (x, y, window) in sliding_window(image, win_size, step_size):
            if self.predict(window):
                yield (x, y)

    def run_detector(self, detector, image, win_size):
        for (x, y) in detector.compute_roi(image):
            window = extract_window(image, (x, y), win_size)
            if window is not None and self.predict(window):
                yield (x, y)


class HaarCascade:
    def __init__(self):
        pass

    def train(self, images, labels):
        pass

    def predict(self, image):
        pass

    def predict_all(self, images):
        return [self.predict(image) for image in images]
