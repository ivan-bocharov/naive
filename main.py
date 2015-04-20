# encoding=utf-8
#author: Bocharov Ivan

import matplotlib.pyplot as plt

from argparse import ArgumentParser

from benchmark import Benchmark

argument_parser = ArgumentParser(description="The demonstration of Naive Bayes classifier implemented as an assignment.")


def add_arguments(argument_parser):
    argument_parser.add_argument('-t')


if __name__ == '__main__':
    benchmark = Benchmark()
    add_arguments(argument_parser)
    argument_parser.parse_args()
    #TODO:Add options and stuff
    print benchmark.benchmark_experiment()


