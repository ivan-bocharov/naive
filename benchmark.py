# encoding=utf-8
#author: Bocharov Ivan

import matplotlib.pyplot as plt
import numpy as np

from classifiers.naive import NaiveBayesClassifier

from sklearn.datasets import make_blobs
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_score, f1_score, recall_score
from collections import Counter, OrderedDict


class Benchmark(object):

    def __init__(self, plot=True, average='macro'):
        self.plot = plot
        self.average = average

    # def generate_dataset(self, n_classes=5, n_samples=300, n_features=5, center_box=(5.0, 10.0), cluster_std=1.0):
    #     return make_blobs(n_samples, n_features, n_classes, center_box=center_box, cluster_std=cluster_std)

    def classifier_performance(self, classifier, dataset, n_folds=10, shuffle=True):
        _, target = dataset
        average_precision = 0.0
        average_recall = 0.0
        average_f1 = 0.0

        cv = StratifiedKFold(target, n_folds, shuffle=shuffle)

        for fold in cv:
            precision, recall, f1 = self.performance_on_current_fold(classifier, dataset, fold)

            average_precision += precision/n_folds
            average_recall += recall/n_folds
            average_f1 += f1/n_folds

        return average_precision, average_recall, average_f1

    def performance_on_current_fold(self, classifier, dataset, fold):
        data, target = dataset
        train_indices, test_indices = fold

        X_train, X_test = data[train_indices], data[test_indices]
        y_train, y_test = target[train_indices], target[test_indices]

        classifier.fit(X_train, y_train)
        y_predicted = classifier.predict(X_test)

        n_classes = len(Counter(y_train))

        average = 'binary' if n_classes == 2 else self.average

        precision, recall, f1 = precision_score(y_test, y_predicted, average=average),\
            recall_score(y_test, y_predicted, average=average), f1_score(y_test, y_predicted, average=average)

        return precision, recall, f1

    def alpha_experiment(self, dataset):
        results = OrderedDict()
        for i in xrange(10):
            alpha = (i+1) * 0.1
            precision, recall, f1 = self.classifier_performance(NaiveBayesClassifier(alpha), dataset)
            results[alpha] = (precision, recall, f1)
        if self.plot:
            self.plot_results(results)
        return results

    def build_classifiers(self):
        classifiers = [NaiveBayesClassifier(alpha=0.3)]
        return classifiers

    def benchmark_experiment(self, dataset):
        classifiers = self.build_classifiers()

        data, target = dataset
        n_folds = 10
        cv = StratifiedKFold(target, n_folds)

        average_precision = 0.0
        average_recall = 0.0
        average_f1 = 0.0

        for classifier in classifiers:
            for fold in cv:
                precision, recall, f1 = self.performance_on_current_fold(classifier, dataset, fold)
                print precision, recall, f1
                average_precision += precision/n_folds
                average_recall += recall/n_folds
                average_f1 += f1/n_folds

        return average_precision, average_recall, average_f1
            #add to the plot

    def plot_results(self, results):

        ind = np.arange(len(results))  # the x locations for the groups
        width = 0.25       # the width of the bars

        fig, ax = plt.subplots()
        precisions = ax.bar(ind, [result[0] for result in results.itervalues()], width, color='r')
        recalls = ax.bar(ind+width, [result[1] for result in results.itervalues()], width, color='g')
        f1 = ax.bar(ind+2*width, [result[2] for result in results.itervalues()], width, color='b')

        ax.legend( (precisions, recalls, f1), ('Precision', 'Recall', 'F1') )
        plt.show()
