'''
  @author: Tyler Procko
  @date:   Fall 2022

  Calculates the Mandelbrot set using Numpy. Pretty fast.
'''

import numpy as np

def mandelbrot_numpy(c, maxIterations):
  escapeCount = np.resize(np.array(0,), c.shape)     # Use np array to store output iterations count
  z = np.zeros(c.shape, np.complex64)           # Use np array for updating z, same shape as q but datatype complex64

  for iteration in range(maxIterations):
    z = z * z + c                               # Mandelbrot
    done   = np.greater(abs(z), 2.0)            # Faster than Python's greater-than; returns true/false
    c      = np.where(done, 0 + 0j, c)          # np.where() returns 0 if true, self if false
    z      = np.where(done, 0 + 0j, z)          # np.where() returns 0 if true, self if false
    escapeCount = np.where(done, iteration, escapeCount)  # np.where() returns iteration if true, output if false
  return escapeCount

def mandelbrot_set_numpy(xMin, xMax, yMin, yMax, width, height, maxIterations):
  realNums     = np.linspace(xMin, xMax, width, dtype = np.float32)   # If we do not declare the datatypes here, it is actually about 40% slower
  imagNums     = np.linspace(yMin, yMax, height, dtype = np.float32)  # ^^^
  complexNums  = np.ravel(realNums + imagNums[:,None] * 1j)           # Make a temporary 1D array of complex numbers (r + i * 1j)
  escapeCounts = mandelbrot_numpy(complexNums, maxIterations)
  escapeCounts = escapeCounts.reshape((width, height))                # Reshape output of Mandelbrot to be nice 2D array
  return (realNums, imagNums, escapeCounts.T)