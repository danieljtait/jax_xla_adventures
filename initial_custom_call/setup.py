from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("custom_call_for_test.pyx")
)