from sklearn.svm import SVC
from sklearn import preprocessing
from src.helpers import sliding_window, extract_window


class SVM:
    def __init__(self, descriptor, C=1.0):
        """
        C-Support Vector Classification
        :param descriptor: Feature Descriptor Object that converts images into vectors
        :param C: Penalty parameter C of the error term
        """
        self.C = C
        self.clf = SVC(C)
        self.descriptor = descriptor
        self.scaler = preprocessing.StandardScaler()

    def train(self, images, labels):
        features = [self.descriptor.compute(img) for img in images]
        features = self.scaler.fit(features).transform(features)
        self.clf.fit(features, labels)
        return self.clf

    def predict(self, image):
        fd = self.descriptor.compute(image).reshape(1, -1)
        fd = self.scaler.transform(fd)
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

    def __repr__(self):
        return "SVM with C Penalty: {}".format(self.C)


class Cascade:
    def __init__(self, descriptor):
        pass

    def train(self, images, labels):
        pass

    def predict(self, image):
        pass

    def predict_all(self, images):
        return [self.predict(image) for image in images]
