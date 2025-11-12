'''
  @author: Tyler Procko
  @date:   Fall 2022

  Cython setup.
'''

from distutils.core import setup
from Cython.Build import cythonize

setup(
   name='Hello',
   ext_modules = cythonize(['c_mandelbrot.pyx'])
)