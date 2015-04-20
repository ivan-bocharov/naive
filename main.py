# encoding=utf-8
#author: Bocharov Ivan

from argparse import ArgumentParser

from classifiers.naive import NaiveBayesClassifier
from sklearn.datasets import make_blobs
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_score, f1_score, recall_score

argument_parser = ArgumentParser(description="The demonstration of Naive Bayes classifier implemented as an assignment.")


def add_arguments(argument_parser):
    argument_parser.add_argument('-t')


def generate_dataset(n_classes=2, n_samples=300, n_features=5, center_box=(5.0, 10.0), cluster_std=1.0):
    return make_blobs(n_samples, n_features, n_classes, center_box=center_box, cluster_std=cluster_std)


def classifier_performance(classifier, dataset, n_folds=5, shuffle=True):
    data, target = dataset
    average_precision = 0.0
    average_recall = 0.0
    average_f1 = 0.0

    cv = StratifiedKFold(target, n_folds, shuffle)

    for fold in cv:
        precision, recall, f1 = performance_on_current_fold(classifier, dataset, fold)

        average_precision += precision/n_folds
        average_recall += recall/n_folds
        average_f1 += f1/n_folds

    return average_precision, average_recall, average_f1


def performance_on_current_fold(classifier, dataset, fold):
    data, target = dataset
    train_indices, test_indices = fold

    X_train, X_test = data[train_indices], data[test_indices]
    y_train, y_test = target[train_indices], target[test_indices]

    classifier.fit(X_train, y_train)
    y_predicted = classifier.predict(X_test)

    precision, recall, F1 = precision_score(y_test, y_predicted),\
        recall_score(y_test, y_predicted), f1_score(y_test, y_predicted)

    return precision, recall, F1




if __name__ == '__main__':
    add_arguments(argument_parser)
    #TODO:Add options and stuff
    dataset = generate_dataset()

    print classifier_performance(NaiveBayesClassifier(alpha=0.3), dataset)


