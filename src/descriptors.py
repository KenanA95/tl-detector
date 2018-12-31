import cv2
import pickle
import numpy as np
from ast import literal_eval
from matplotlib import pyplot as plt
from dask import delayed
from skimage.feature import hog, local_binary_pattern
from skimage.transform import integral_image
from skimage.feature import haar_like_feature, haar_like_feature_coord, draw_haar_like_feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class HogDescriptor:
    def __init__(self, block_size, cell_size, orientations):
        self.block_size = block_size
        self.cell_size = cell_size
        self.orientations = orientations

    def compute(self, image, multichannel=True, visualize=False):
        return hog(image, self.orientations, pixels_per_cell=self.cell_size, cells_per_block=self.block_size,
                   multichannel=multichannel, visualize=visualize, block_norm='L2-Hys')

    def display(self, image):
        fd, hog_image = self.compute(image, visualize=True)
        plt.imshow(hog_image, cmap='gray')
        plt.show()

    @classmethod
    def from_config_file(cls, settings):
        return cls(literal_eval(settings['block_size']), literal_eval(settings['cell_size']),
                   int(settings['orientations']))

    def __repr__(self):
        return " Block Size: {0} \n Cell Size: {1} \n Orientations: {2}" \
               .format(self.block_size, self.cell_size, self.orientations)


class LBPDescriptor:
    def __init__(self, radius, points, method='default'):
        self.points = points
        self.radius = radius
        self.method = method

    def compute(self, image, visualize=False):
        lbp = local_binary_pattern(image, self.points, self.radius, self.method)

        # Compute and normalize the histogram
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, self.points + 3), range=(0, self.points + 2))
        hist = hist.astype('float') / hist.sum()

        if visualize:
            return hist, lbp

        return hist

    def display(self, image):
        hist, lbp = self.compute(image, visualize=True)
        plt.imshow(lbp, cmap='gray')
        plt.show()

    @classmethod
    def from_config_file(cls, settings):
        return cls(int(settings['points']), int(settings['radius']))

    def __repr__(self):
        return "LBP with {0} radius and {1} points using {2} method".format(self.radius, self.points, self.method)


class HaarDescriptor:
    def __init__(self, selected_feature_file):
        """ More information on selected feature file can be found in docs/ """
        self.selected_feature_file = selected_feature_file
        self.selected_feature_coord, self.selected_feature_type = pickle.load(open(selected_feature_file, 'rb'))

    def compute(self, image):
        ii = integral_image(image)
        return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                                 self.selected_feature_type, self.selected_feature_coord)

    def identify_selected_features(self, images, labels, size=(32, 64)):
        """ """
        # Compute the haar-like features for all of the different feature types
        X = delayed(self.compute(img) for img in images)
        X = np.array(X.compute(scheduler='processes'))
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

        # Train a random forest classifier to find out which features are important for training
        clf = RandomForestClassifier(n_estimators=1000, max_depth=None, max_features=100, n_jobs=-1, random_state=0)
        clf.fit(X_train, y_train)
        significant_features = np.argsort(clf.feature_importances_)[::-1]

        cdf_feature_importances = np.cumsum(clf.feature_importances_[significant_features])
        cdf_feature_importances /= np.max(cdf_feature_importances)
        sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.7)
        sig_feature_percent = round(sig_feature_count / len(cdf_feature_importances) * 100, 1)

        print(('{} features, or {}%, account for {}% of branch points in the random '
               'forest.').format(sig_feature_count, sig_feature_percent, 0.7))

        # Extract all possible features to be able to select the most salient
        feature_coord, feature_type = haar_like_feature_coord(width=size[0], height=size[1])

        # Store the most informative features
        self.selected_feature_coord = feature_coord[significant_features[:sig_feature_count]]
        self.selected_feature_type = feature_type[significant_features[:sig_feature_count]]

    def display(self, significant_features, image):
        """ Plot the most significant haar-like features on top of an image """
        feature_coord, _ = haar_like_feature_coord(width=32, height=64)

        fig, axes = plt.subplots(3, 2)
        for idx, ax in enumerate(axes.ravel()):
            image = draw_haar_like_feature(image, 0, 0, 32, 64, [feature_coord[significant_features[idx]]])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle('The most important features')
        plt.show()

    @classmethod
    def from_config_file(cls, settings):
        return cls(settings['selected_feature_file'])

    def __repr__(self):
        return "Haar Descriptor with {} selected features. Model loaded from: {}"\
            .format(len(self.selected_feature_type), self.selected_feature_file)
