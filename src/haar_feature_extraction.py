import cv2
import numpy as np
from dask import delayed
from matplotlib import pyplot as plt
from skimage.transform import integral_image
from skimage.feature import haar_like_feature, haar_like_feature_coord, draw_haar_like_feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from helpers import read_directory_images, resize_images


@delayed
def extract_feature_image(image, feature_type, feature_coord=None):
    ii = integral_image(image)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1], feature_type, feature_coord=feature_coord)


def feature_extraction(images, labels, feature_types):
    """ """
    X = delayed(extract_feature_image(img, feature_types) for img in images)
    X = np.array(X.compute(scheduler='processes'))
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

    # Extract all possible features to be able to select the most salient.
    feature_coord, feature_type = haar_like_feature_coord(width=32, height=64, feature_type=feature_types)

    # Train a random forest classifier and check performance
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None, max_features=100, n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)

    idx_sorted = np.argsort(clf.feature_importances_)[::-1]

    cdf_feature_importances = np.cumsum(clf.feature_importances_[idx_sorted])
    cdf_feature_importances /= np.max(cdf_feature_importances)
    sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.7)
    sig_feature_percent = round(sig_feature_count /
                                len(cdf_feature_importances) * 100, 1)
    print(('{} features, or {}%, account for 70% of branch points in the random '
           'forest.').format(sig_feature_count, sig_feature_percent))

    # Select the most informative features
    selected_feature_coord = feature_coord[idx_sorted[:sig_feature_count]]
    selected_feature_type = feature_type[idx_sorted[:sig_feature_count]]

    # Sort features in order of importance
    return idx_sorted, selected_feature_coord, selected_feature_type


def plot_most_significant(idx_sorted, feature_types, image):
    """ Plot the 6 most significant haar-like features on top of an image """

    feature_coord, _ = haar_like_feature_coord(width=32, height=64, feature_type=feature_types)

    fig, axes = plt.subplots(3, 2)
    for idx, ax in enumerate(axes.ravel()):
        image = draw_haar_like_feature(image, 0, 0, 32, 64, [feature_coord[idx_sorted[idx]]])
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('The most important features')
    plt.show()


if __name__ == "__main__":
    pos_dir = "C:/Users/kenan/Desktop/repos/tl-data/positives/"
    neg_dir = "C:/Users/kenan/Desktop/repos/tl-data/negatives/"

    positive_images = list(read_directory_images(pos_dir, extension='.png', n=10))
    negative_images = list(read_directory_images(neg_dir, extension='.png', n=10))

    # Resize all to 32x64 and convert to grayscale
    positive_images = resize_images(positive_images, (32, 64))
    negative_images = resize_images(negative_images, (32, 64))

    images = np.concatenate((positive_images, negative_images))
    images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]
    labels = np.array([1] * len(positive_images) + [0] * len(negative_images))

    print("Total positive images: {}".format(len(positive_images)))
    print("Total negative images: {}".format(len(negative_images)))

    feature_types = ['type-2-x', 'type-2-y', 'type-3-x', 'type-3-y']
    idx, feature_coord, feature_type = feature_extraction(images, labels, feature_types)
    plot_most_significant(idx, feature_types, positive_images[0])
