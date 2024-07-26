from setuptools import setup, find_packages
from os.path import join, dirname

import fsnn_classifiers


setup(
    name='fsnn_classifiers',
    version=0.0,
    packages=find_packages(),
    #long_description=open(join(dirname(__file__), 'README.txt')).read(),
)