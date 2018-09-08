from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve


def evaluate_model(classifier, images, labels):
    """ Output the key metrics of a trained classifier """
    predictions = classifier.predict_all(images)
    recall = recall_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)

    print("Recall: {:.2f}".format(recall))
    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))


