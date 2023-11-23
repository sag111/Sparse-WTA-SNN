from setuptools import setup, find_packages
from os.path import join, dirname

import fsnn_classifiers


setup(
    name='fsnn_classifiers',
    version=0.1,
    packages=find_packages(),
    include_package_data=True,
    long_description=open(join(dirname(__file__), 'README.md')).read(),
)