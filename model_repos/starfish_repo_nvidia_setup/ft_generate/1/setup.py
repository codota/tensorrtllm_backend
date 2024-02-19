#  Copyright (c) Tabnine LTD  2023.
#  All Rights Reserved.

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("core/*.py"),
)
