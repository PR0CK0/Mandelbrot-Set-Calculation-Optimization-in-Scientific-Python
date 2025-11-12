'''
  @author: Tyler Procko
  @date:   Fall 2022

  Calculates the Mandelbrot set in pure Python. Slow...
'''

import numpy as np

def mandelbrot_purepython(c, maxIterations):
  z = 0                           # z always starts as 0
  for n in range(maxIterations):  # When testing a complex, c, if we hit the max number of iterations, kick out, otherwise we could be computing infinitely
    if abs(z) > 2:                # 2 is the radius of the Mandelbrot set circle (divergence point)
      return n                    # If we hit this limit, kick out
    z = z * z + c                 # Mandelbrot (z ** 2 is MUCH slower; about 40%)
  return maxIterations            # Otherwise, return max iterations

def mandelbrot_set_purepython(xMin, xMax, yMin, yMax, width, height, maxIterations):
  realNums = np.linspace(xMin, xMax, width)
  imagNums = np.linspace(yMin, yMax, height)
  escapeCounts = []
  for r in realNums:
    for i in imagNums:
      # Uses complex(), which is the same as real + imaginary * 1j
      escapeCounts.append(mandelbrot_purepython(complex(r, i), maxIterations))
  return (realNums, imagNums, escapeCounts)