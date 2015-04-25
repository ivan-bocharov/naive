# encoding=utf-8
#author: Bocharov Ivan

import matplotlib.pyplot as plt
import numpy as np

from classifiers.naive import NaiveBayesClassifier

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model.logistic import LogisticRegression
from collections import Counter, OrderedDict


class Benchmark(object):

    def __init__(self, plot=True, average='macro', logging=True):
        self.plot = plot
        self.average = average
        self.logging = logging

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
        if self.logging:
                print "Performance of classifier on current fold: P={} R={} F1={}".format(precision, recall, f1)
        return precision, recall, f1

    def alpha_experiment(self, dataset):
        if self.logging:
            print "Running an experiment for Naive Bayes alpha parameter..."
            print "="*80
        results = OrderedDict()
        for i in xrange(10):
            alpha = (i+1) * 0.1

            precision, recall, f1 = self.classifier_performance(NaiveBayesClassifier(alpha), dataset)
            if self.logging:
                print "-"*80
                print "Performance of classifier when alpha={}: P={} R={} F1={}".format(alpha, precision, recall, f1)

            results[alpha] = (precision, recall, f1)
            print "-"*80
        if self.logging:
            print "Finished!"
            print "="*80

        if self.plot:
            self.plot_alpha_results(results)
        return results

    def build_classifiers(self):
        classifiers = [
            ("NaiveBayes", NaiveBayesClassifier(alpha=0.3)),
            ("kNN", KNeighborsClassifier()),
            ("Linear SVM", LinearSVC()),
            ("Logistic\nRegression", LogisticRegression())
        ]
        return classifiers

    def benchmark_experiment(self, dataset):
        if self.logging:
            print "Running classifier benchmark experiment..."

        classifiers = self.build_classifiers()

        data, target = dataset
        n_folds = 10
        cv = StratifiedKFold(target, n_folds)
        results = OrderedDict()
        for classifier_name, classifier in classifiers:
            average_precision, average_recall, average_f1 = 0.0, 0.0, 0.0
            if self.logging:
                print "="*80
                print "Running CV for classifier {}...".format(classifier_name)
                print "-"*80
            for fold in cv:
                precision, recall, f1 = self.performance_on_current_fold(classifier, dataset, fold)
                if self.logging:
                    print "Performance of classifier on current fold: P={} R={} F1={}".format(precision, recall, f1)
                average_precision += precision/n_folds
                average_recall += recall/n_folds
                average_f1 += f1/n_folds
            if self.logging:
                print "-"*80
                print "Average performance of classifier {}: P={} R={} F1={}".format(classifier_name,
                                                                                     average_precision,
                                                                                     average_recall,
                                                                                     average_f1)
                print "="*80
            results[classifier_name] = (average_precision, average_recall, average_f1)

        if self.plot:
            self.plot_benchmark_results(results)

    def plot_alpha_results(self, results):

        ind = np.arange(len(results))
        width = 0.25

        fig, ax = plt.subplots()
        precisions = ax.bar(ind, [result[0] for result in results.itervalues()], width, color='r')
        recalls = ax.bar(ind+width, [result[1] for result in results.itervalues()], width, color='g')
        f1 = ax.bar(ind+2*width, [result[2] for result in results.itervalues()], width, color='b')

        ax.legend((precisions, recalls, f1), ('Precision', 'Recall', 'F1'), loc='lower right')
        min_on_display = min(min(results.values()))
        plt.ylim([min_on_display-0.1, 1.0])
        ax.set_xticks(ind+(1.5*width))
        ax.set_xticklabels( [0.1*i for i in xrange(1, 11)] )
        plt.title("Naive Bayes classifier performance")
        plt.xlabel("Alpha (regularization parameter)")
        plt.show()

    def plot_benchmark_results(self, results):
        ind = np.arange(len(results))  # the x locations for the groups
        width = 0.25       # the width of the bars

        fig, ax = plt.subplots()

        precisions = ax.bar(ind, [result[0] for result in results.itervalues()], width, color='r')
        recalls = ax.bar(ind+width, [result[1] for result in results.itervalues()], width, color='g')
        f1 = ax.bar(ind+2*width, [result[2] for result in results.itervalues()], width, color='b')

        ax.legend((precisions, recalls, f1), ('Precision', 'Recall', 'F1'), loc='lower right')
        min_on_display = min(min(results.values()))
        plt.ylim([min_on_display-0.1, 1.0])
        ax.set_xticks(ind+(1.5*width))
        ax.set_xticklabels([c for c in results.iterkeys()])
        plt.title("Various classifiers' performance")
        plt.show()
