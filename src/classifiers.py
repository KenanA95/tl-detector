from sklearn import linear_model


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
