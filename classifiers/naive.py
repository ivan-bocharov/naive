# encoding=utf-8
#author: Bocharov Ivan
from __future__ import division

import numpy as np
import operator

from base import BaseClassifier
from math import log
from sklearn.datasets import make_blobs

from collections import defaultdict
from scipy.sparse import issparse
import matplotlib.pyplot as plt


class NaiveBayesClassifier(BaseClassifier):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def initiate_priors(self):
        self.labels = defaultdict(lambda: 0)
        self.labelsfeatures = defaultdict(lambda: 0)
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: 0))

    def fit(self, X_train, y_train):
        self.initiate_priors()
        for i in xrange(len(y_train)):
            label = y_train[i]
            self.labels[label] += 1
            features = X_train[i].indices if issparse(X_train) else X_train[i]
            features /= np.linalg.norm(features)
            for j in xrange(len(features)):
                self.feature_counts[label][j] += X_train[i, j]
            self.labelsfeatures[label] += len(X_train[i])
        self.vocabulary_size = X_train.shape[1]

    def predict(self, X):
        prediction = [self.predict_one(row) for row in X]
        return prediction

    def predict_one(self, obj):
        probabilities = {}
        obj /= np.linalg.norm(obj)
        for label in self.labels:
            probabilities[label] = self.object_probability(obj, label) + \
                self.label_probability(label)
        return max(probabilities.iteritems(), key=operator.itemgetter(1))[0]

    def feature_count(self, feature, label):
        if label in self.feature_counts:
            if feature in self.feature_counts[label]:
                return self.feature_counts[label][feature]
            else:
                return 0

    def total_label_count(self, label):
        return sum(self.feature_counts[label].values())

    def label_count(self, label):
        if label in self.labels:
            return self.labels[label]
        else:
            raise KeyError

    def get_probability(self, feature, label):
        key, value = feature
        if label in self.labels:
            return (self.feature_count(key, label)+self.alpha)/(self.labelsfeatures[label]+self.vocabulary_size*self.alpha)
        else:
            print "Given label doesn't exist"
            raise KeyError

    def object_probability(self, row_object, label):
        probability = 0.0
        for feature in xrange(len(row_object)):
            probability += row_object[feature]*log(self.get_probability((feature, row_object[feature]), label))
        return probability

    def label_probability(self, label):
        if label in self.labels:
            return log(self.labels[label]/sum(self.labels.values()))
        else:
            print "Given label doesn't exist"
            raise KeyError