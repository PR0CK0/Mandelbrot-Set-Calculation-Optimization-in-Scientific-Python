'''
  @author: Tyler Procko
  @date:   Fall 2022

  See c_mandelbrot.pyx
  Built with CythonBuilder in Windows

  Calculates the Mandelbrot set using Numba. Blazing fast!
'''

from c_mandelbrot import mandelbrot_set_cython_func

def mandelbrot_set_cython(xMin, xMax, yMin, yMax, width, height, maxIterations):
  return mandelbrot_set_cython_func(xMin, xMax, yMin, yMax, width, height, maxIterations)