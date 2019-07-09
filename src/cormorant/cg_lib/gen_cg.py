import torch
import torch.utils.cpp_extension as ext

import sys, warnings

import os
from pwd import getpwuid
from tempfile import gettempdir

build_directory = gettempdir() + '/' + getpwuid(os.getuid())[0] + '/cpp_extension/'

if not os.path.exists(build_directory):
    os.makedirs(build_directory)

print('C++ extension build directory:', build_directory)

if not sys.warnoptions:
	warnings.simplefilter("ignore")
	restore_warnings = True

source = __file__.replace('.py', '.cpp')
_GenCG = ext.load(name='GenCG', sources=[source], build_directory=build_directory)

if restore_warnings: warnings.simplefilter("default")
