# encoding=utf-8
#author: Bocharov Ivan

from setuptools import setup

setup(name='Naive Bayes showcase',
      version='0.1',
      description='Some experiments with Naive Bayes and other classifiers',
      url='http://github.com/bocharov-ivan/naive',
      author='Bocharov Ivan',
      author_email='bocharovia@gmail.com',
      license='MIT',
      packages=['nb-experiments'],
      install_requires=[
          'scikit-learn',
      ],
      zip_safe=False)