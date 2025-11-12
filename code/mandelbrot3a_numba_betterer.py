'''
  @author: Tyler Procko
  @date:   Fall 2022

  Calculates the Mandelbrot set using super-improved Numba. SO FASSSSSSST!!!!
'''

import numpy as np
from numba import jit, vectorize, guvectorize, complex64, int32

@jit(int32(complex64, int32))
def mandelbrot_numba_betterer3(c, maxIterations):
  realNew = 0
  real  = 0
  imag  = 0
  for n in range(maxIterations):
    realNew = real * real - imag * imag + c.real           # Uses the square of a complex number trick
    imag = 2 * real * imag + c.imag
    real = realNew
    if real * real + imag * imag > 4.0:     
      return n
  return 0

# The '(n),()->(n)' is an input template: means a 1D array and scalar are input, and a 1D array is output
# target is set to parallel to use multiple cores; default is 'cpu'
@guvectorize([(complex64[:], int32[:], int32[:])], '(n),()->(n)', target = 'parallel')
def mandelbrot_numba_betterer2(c, maxIterations, output):
  maxIterationsTemp = maxIterations[0]                        # Temp array is necessary because guvectorize expects arrays
  for i in range(c.shape[0]):
    output[i] = mandelbrot_numba_betterer3(c[i], maxIterationsTemp)
        
def mandelbrot_set_numba_betterer(xMin, xMax, yMin, yMax, width, height, maxIterations):
  realNums = np.linspace(xMin, xMax, width, dtype = np.float32)
  imagNums = np.linspace(yMin, yMax, height, dtype = np.float32)
  complexNums = realNums + imagNums[:, None] * 1j
  escapeCounts = mandelbrot_numba_betterer2(complexNums, maxIterations)
  return (realNums, imagNums, escapeCounts.T) 