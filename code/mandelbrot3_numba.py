'''
  @author: Tyler Procko
  @date:   Fall 2022

  Calculates the Mandelbrot set using Numba. Really fast!
'''

import numpy as np
from numba import jit

@jit
def mandelbrot_numba(c, maxIterations):
  z = 0                                          # z always starts at 0
  for n in range(maxIterations):
    if z.real * z.real + z.imag * z.imag > 4.0:  # Re-using the Pythagorean theorem trick
      return n
    z = z * z + c                                # Same as pure Python code 
  return 0                                       # TODO returning 0 plots correctly; returning iterations inverts colors... why?

@jit
def mandelbrot_set_numba(xMin, xMax, yMin, yMax, width, height, maxIterations):
  realNums = np.linspace(xMin, xMax, width)   # Declaring datatypes here breaks the code :)
  imagNums = np.linspace(yMin, yMax, height)  # ^^^
  escapeCounts = np.empty((width, height))
  for i in range(width):
    for j in range(height):
      # Same as complex(r, i) ... real + imaginary * 1j
      escapeCounts[i, j] = mandelbrot_numba(realNums[i] + imagNums[j] * 1j, maxIterations)
  return (realNums, imagNums, escapeCounts)