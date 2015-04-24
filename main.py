# encoding=utf-8
#author: Bocharov Ivan

import matplotlib.pyplot as plt

from argparse import ArgumentParser
from sklearn.datasets import make_blobs

from benchmark import Benchmark

argument_parser = ArgumentParser(description="The demonstration of Naive Bayes classifier implemented as an assignment.")


def add_arguments(argument_parser):
    argument_parser.add_argument('-c', dest='classes', help="Classes number", default=2, type=int)
    argument_parser.add_argument('-s', dest='samples', help="Samples number", default=300, type=int)
    argument_parser.add_argument('-f', dest='features', help="Features number", default=100, type=int)
    argument_parser.add_argument('-box', dest='box', action='store', nargs='*',  help="The box of centers of classes", default=(5.0, 10.0), type=str)
    argument_parser.add_argument('-std', dest='std', help="Standard deviation of class elements distribution", default=3.0, type=float)
    argument_parser.add_argument('-a', dest='average', help="Averaging method", default='macro', type=str)
    argument_parser.add_argument('--logging', dest='log', help="Logging enabling", nargs='?', const=1)
    argument_parser.add_argument('--plot', dest='plot', help="Plotting enabling", nargs='?', const=1)


def generate_dataset(n_classes=5, n_samples=300, n_features=100, center_box=(5.0, 10.0), cluster_std=3.0):
    print '''Dataset parameters:
    Number of classes: {}
    Number of samples: {}
    Number of features: {}
    The box of centers of classes: {}
    Standard deviation of class elements: {}
'''.format(n_classes, n_samples, n_features, center_box, cluster_std)
    return make_blobs(n_samples, n_features, n_classes, center_box=center_box, cluster_std=cluster_std)


if __name__ == '__main__':
    add_arguments(argument_parser)
    args = argument_parser.parse_args()
    benchmark = Benchmark(plot=args.plot, logging=args.log, average=args.average)
    n_classes, n_samples, n_features, cluster_std = args.classes,\
    args.samples, args.features, args.std
    try:
        center_box = tuple(float(coord) for coord in args.box)
    except ValueError:
        center_box = (5.0, 10.0)

    dataset = generate_dataset(n_classes, n_samples, n_features, center_box, cluster_std)
    benchmark.alpha_experiment(dataset)
    benchmark.benchmark_experiment(dataset)


