from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def evaluate_model(classifier, images, labels):
    """ Output the key metrics of a trained classifier """
    predictions = classifier.predict_all(images)
    recall = recall_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)

    print("Recall: {:.2f}".format(recall))
    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))


def cross_train_validate(classifier, images, labels, test_size):
    """ Split a data-set and train a classifier. See how well it performs """
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size)
    classifier.train(X_train, y_train)
    evaluate_model(classifier, images, labels)
