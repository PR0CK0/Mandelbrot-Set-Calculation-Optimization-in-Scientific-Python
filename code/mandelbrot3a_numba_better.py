'''
  @author: Tyler Procko
  @date:   Fall 2022

  Calculates the Mandelbrot set using improved Numba. Even faster!
'''

import numpy as np
from numba import jit

@jit
def mandelbrot_numba_better(cReal, cImag, maxIterations):
  real = cReal                        # Maintain original 'z' value
  imag = cImag                        # Maintain original 'z' value
  for n in range(maxIterations):
    realNew = real * real             # Pre-calculation seemed to make it a bit faster
    imagNew = imag * imag             # Pre-calculation
    if realNew + imagNew > 4.0:       # Same as old numba
      return n
    imag = 2 * real * imag + cImag    # The square of a+bi is a new imaginary number a + bi...
    real = realNew - imagNew + cReal  # ... where the new a is a^2 - b^2 and the new b is 2ab
  return 0

@jit
def mandelbrot_set_numba_better(xMin, xMax, yMin, yMax, width, height, maxIterations):
  realNums = np.linspace(xMin, xMax, width)   # Declaring datatypes here breaks the code :)
  imagNums = np.linspace(yMin, yMax, height)  # ^^^
  complexNums = np.empty((width, height))
  for i in range(width):
    for j in range(height):
      # No built-in Python complex-type... just pass in every complex as two floats
      # We will use this exact thing in Cython to get around not being able to cdef Complex types
      complexNums[i, j] = mandelbrot_numba_better(realNums[i], imagNums[j], maxIterations)
  return (realNums, imagNums, complexNums)