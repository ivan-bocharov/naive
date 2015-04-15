# encoding=utf-8
#author: Bocharov Ivan


class BaseClassifier(object):

    def fit(self, X_train, y_train):
        raise NotImplementedError()

    def predict(self, X_test):
        raise NotImplementedError

