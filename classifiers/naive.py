# encoding=utf-8
#author: Bocharov Ivan
from __future__ import division

import numpy as np
import operator

from base import BaseClassifier
from math import log
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict



class NaiveBayesClassifier(BaseClassifier):

    def __init__(self, alpha=1.0):
        self.alpha = alpha

        self.labels = defaultdict(lambda: 0)
        self.labelsfeatures = defaultdict(lambda: 0)
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: 0))

    def fit(self, X_train, y_train):
        for i in xrange(y_train.shape[0]):
            label = y_train[i]
            self.labels[label] += 1
            for index in X_train[i].indices:
                self.feature_counts[label][index] += X_train[i][0, index]
            self.labelsfeatures[label] += len(X_train[i].indices)
        print self.labels

    def predict(self, X):
        prediction = [self.predict_one(row) for row in X]
        return prediction

    def predict_one(self, obj):
        probabilities = {}
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

    def get_vocabulary_size(self):
        temp = defaultdict(lambda: 1)
        for label in self.feature_counts:
            for feature in self.feature_counts[label]:
                temp[feature];
        self.vocabulary_size = len(temp)

    def get_probability(self, feature, label):
        if label in self.labels:
            return (self.feature_count(feature, label)+self.alpha)/(self.labelsfeatures[label]+self.vocabulary_size*self.alpha)
        else:
            print "Given label doesn't exist"
            raise KeyError

    def object_probability(self, row_object, label):
        probability = 0.0
        for feature in row_object.indices:
            probability += log(self.get_probability(feature, label))
        return probability

    def label_probability(self, label):
        if label in self.labels:
            return log(self.labels[label]/sum(self.labels.values()))
        else:
            print "Given label doesn't exist"
            raise KeyError



if __name__ == '__main__':
    data_train = fetch_20newsgroups(subset='train',
                                shuffle=True, random_state=42,
                                )

    data_test = fetch_20newsgroups(subset='test',
                               shuffle=True, random_state=42,
                               )
    print('data loaded')

    categories = data_train.target_names

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    y_train, y_test = data_train.target, data_test.target

    X_train = vectorizer.fit_transform(data_train.data)

    X_test = vectorizer.transform(data_test.data)

    nb = NaiveBayesClassifier()
    print('Fitting...')
    nb.fit(X_train, y_train)
    print('Fitted!')
    nb.get_vocabulary_size()
    print nb.predict(X_test)
    print('Done!')