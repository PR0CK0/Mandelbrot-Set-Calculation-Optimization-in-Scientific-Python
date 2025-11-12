'''
  @author: Tyler Procko
  @date:   Fall 2022

  Calculates the Mandelbrot set using multiprocessing. 

  NOTE: this code is not part of the regular testing suite, so its timings should be taken with a grain of salt.
  This is because there are no across-file calls like all the other optimization techniques, so it has a "head start".
  Here, we have our own main method to let the pools work without error, or repeatedly calling the other main.
  Still, the input values are the same as the others and we see that this is actually pretty fast.
'''

import time, numpy as np
from matplotlib      import pyplot
from multiprocessing import Pool

xMin, xMax    = -2.0, 0.5
yMin, yMax    = -1.2, 1.2
width, height =  1000,1000
maxIterations =  80         # map can only take one argument so we set this globally
loadIncrease  =  .2
loadTimes     =  20 

# Pretty much identical to the pure Python code
def mandelbrot(c): 
  z = 0
  for n in range(maxIterations):
    if abs(z) > 2:  # Pythagorean theorem trick has no effect here
      return n  
    z = z * z + c
  return maxIterations

if __name__ == '__main__':
  for i in range (loadTimes):
    print(width, 'x', height, ':', maxIterations, 'iterations')
    time1 = time.time()
    realNums = np.linspace(xMin, xMax, width)
    imagNums = np.linspace(yMin, yMax, height)

    p = Pool()
    complexNums = [complex(real, imag) for imag in imagNums for real in realNums] 
    escapeCounts = p.map(mandelbrot, complexNums)                                  
    escapeCounts = np.reshape(escapeCounts, (width, height))    
    print(str(time.time() - time1))

    #pyplot.imshow(escapeCounts) 
    #pyplot.show()

    width         += int(width         * loadIncrease)
    height        += int(height        * loadIncrease)
    maxIterations += int(maxIterations * loadIncrease)
