# encoding=utf-8
#author: Bocharov Ivan

import numpy as np

from base import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):

    def __init__(self, alpha=1):
        self.alpha = 0
        pass

    def fit(self, X_train, y_train):
        print X_train.shape()
        pass

    def predict(self, y_train):
        pass