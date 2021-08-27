#-*- coding: utf-8 -*-

from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup
import sys
import os

def configuration(parent_package='', top_path=None):
    config = Configuration('hdflow', parent_package, top_path)
    config.make_config_py()
    return config

if __name__ == '__main__':
    setup(configuration=configuration)

# end of file
