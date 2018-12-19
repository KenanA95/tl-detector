import yaml
import numpy as np
from helpers import read_directory_images, resize_images
from features import *
from classifiers import *
from sklearn.externals import joblib
from ast import literal_eval


def load_descriptor(settings):
    return {
        'hog': HogDescriptor.from_config_file(settings['hog']),
        'lbp': LBPDescriptor.from_config_file(settings['lbp']),
        'haar': HaarDescriptor.from_config_file(settings['haar'])
    }.get(settings['train']['descriptor'], 'hog')    # Default to HOG for invalid input


def load_classifier(settings, descriptor):
    return {
        'svm': SVM(descriptor, settings['svm']['C']),
        'cascade': Cascade(descriptor),
    }.get(settings['train']['classifier'], 'svm')   # Default to SVM for invalid input


if __name__ == "__main__":

    with open("config.yaml", "r") as stream:
        settings = yaml.load(stream)

    descriptor = load_descriptor(settings)
    classifier = load_classifier(settings, descriptor)

    print("Descriptor Settings \n" + str(descriptor))
    print("Classifier Settings \n" + str(classifier))
    print("Reading in the images...")

    positive_images = read_directory_images(settings['train']['positive_image_directory'], extension='.png')
    negative_images = read_directory_images(settings['train']['negative_image_directory'], extension='.png')

    training_size = literal_eval(settings['train']['window_size'])
    positive_images = resize_images(list(positive_images), training_size)
    negative_images = resize_images(list(negative_images), training_size)
    images = np.concatenate((positive_images, negative_images))

    # Set up the labels for binary classification
    positive_labels = np.ones(len(positive_images), dtype=int)
    negative_labels = np.zeros(len(negative_images), dtype=int)
    labels = np.concatenate((positive_labels, negative_labels))

    print("Starting training...")
    classifier.train(images, labels)
    joblib.dump(classifier, settings['train']['outfile'])
