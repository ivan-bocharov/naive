# encoding=utf-8
#author: Bocharov Ivan

import matplotlib.pyplot as plt

from argparse import ArgumentParser
from sklearn.datasets import make_blobs

from benchmark import Benchmark

argument_parser = ArgumentParser(description="The demonstration of Naive Bayes classifier implemented as an assignment.")


def add_arguments(argument_parser):
    argument_parser.add_argument('-t')


def generate_dataset(n_classes=5, n_samples=300, n_features=100, center_box=(5.0, 10.0), cluster_std=3.0):
        print '''
Dataset parameters:
    Number of classes: {}
    Number of samples: {}
    Number of features: {}
    The box of centers of classes: {}
    Standard deviation of class elements: {}
'''.format(n_classes, n_samples, n_features, center_box, cluster_std)
        return make_blobs(n_samples, n_features, n_classes, center_box=center_box, cluster_std=cluster_std)


if __name__ == '__main__':
    benchmark = Benchmark()
    add_arguments(argument_parser)
    argument_parser.parse_args()
    #TODO:Add options and stuff
    benchmark.alpha_experiment(generate_dataset())
    benchmark.benchmark_experiment(generate_dataset())


