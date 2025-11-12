# Mandelbrot Set Optimization

**A comprehensive study of Python high-performance computing techniques through fractal generation**

## What is this?

This project demonstrates the dramatic performance improvements possible when optimizing computationally intensive Python code. By implementing the same mathematical problem (computing the Mandelbrot set) using 8 different optimization techniques, this work achieves speedups of up to **360x** over pure Python.

The Mandelbrot set creates stunning fractal images by testing which complex numbers remain bounded when repeatedly squared and added to themselves. With 1 million complex numbers each requiring up to 80 iterations to test, this is an *embarrassingly parallel* problem that's perfect for exploring optimization techniques.

**Final Project for MA553 - High Performance Scientific Computing (Fall 2022)**

## Performance Results

| Technique | Time (1000×1000, 80 iter) | Speedup vs Pure Python | % Improvement |
|-----------|---------------------------|------------------------|---------------|
| **Pure Python** | 3.6s | 1x | baseline |
| **Multiprocessing** | 1.34s | 2.69x | 62.78% |
| **NumPy** | 1.23s | 2.93x | 65.83% |
| **Better NumPy** | 0.85s | 4.24x | 76.39% |
| **Numexpr** | 0.45s | 8x | 87.5% |
| **Numba** | 0.24s | 15x | 93.33% |
| **Better Numba** | 0.14s | 25.71x | 96.11% |
| **Cython** | 0.06s | 60x | 98.33% |
| **Guvectorized Numba** | 0.01s | **360x** | **99.72%** |

### Key Discoveries

1. **Simple optimizations matter**: Replacing `z**2` with `z*z` gave a 40% speedup
2. **Data types are critical**: Using `np.float32` instead of default `np.float64` improved NumPy speed by 40%
3. **Mathematical tricks help**: Using the Pythagorean theorem (`x²+y² > 4`) instead of `abs(z) > 2` significantly reduced computation time
4. **Avoid Python classes**: Object-oriented code was measurably slower than imperative style for numerical work
5. **Numba is magical**: The guvectorized Numba implementation with parallel execution achieved near-optimal performance with minimal code changes

## Optimization Techniques Implemented

1. **Pure Python** (3.6s) - Baseline using nested loops and Python's complex type
2. **NumPy** (1.23s) - Vectorized operations with `np.where()` and array broadcasting
3. **Better NumPy** (0.85s) - Eliminated `ravel()`, used Pythagorean theorem for divergence check
4. **Numba** (0.24s) - JIT compilation with `@jit` decorator
5. **Better Numba** (0.14s) - Manual complex arithmetic, pre-computed squares
6. **Betterer Numba** (0.01s) - `@guvectorize` with parallel execution across multiple cores
7. **Numexpr** (0.45s) - String-based expression evaluation for complex NumPy operations
8. **Cython** (0.06s) - Compiled to C with static typing via `cdef`
9. **Multiprocessing** (1.34s) - Parallel execution using `Pool.map()`

## Features

- **Performance Benchmarking**: Runs each implementation 5 times and reports average execution time
- **Beautiful Visualizations**: Generate stunning Matplotlib renders of the Mandelbrot fractal with customizable color schemes
- **Scalability Testing**: Automatically increment problem size by 20% to analyze how optimizations scale (tested up to 32,000×32,000 pixels!)
- **Profiling Support**: Built-in `line_profiler` integration for bottleneck identification
- **Flexible Configuration**: Easily adjust resolution, iteration limits, and complex plane boundaries
- **Windows Compatible**: Successfully runs NumPy, Numexpr, Numba, Cython, and multiprocessing on Windows 10

## Usage

Run the main script:

```bash
python code/main.py
```

### Configuration

Edit the parameters in [main.py](code/main.py) to customize behavior:

- `_VISUALIZE` - Show visualizations (True) or just benchmark (False)
- `_SEE_ONE` - Quick visualization using fastest method
- `_INCREASE_LOAD` - Run scaling tests with increasing problem size
- `_PROFILING` - Enable detailed line-by-line profiling

### Mandelbrot Parameters

Adjust these in [main.py](code/main.py) to explore different regions:

```python
_XMIN, _XMAX = -2.0, 0.5      # Real axis bounds
_YMIN, _YMAX = -1.2, 1.2      # Imaginary axis bounds
_WIDTH, _HEIGHT = 1000, 1000  # Image resolution
_MAX_ITERATIONS = 80          # Divergence iteration limit
```

For extreme detail, try zooming in:
```python
_XMIN, _XMAX = -0.74877, -0.74872
_YMIN, _YMAX = 0.06505, 0.06510
_MAX_ITERATIONS = 2048
```

## Requirements

```bash
pip install numpy numba numexpr matplotlib
```

For Cython (requires C compiler):
```bash
pip install cython
python code/setup.py build_ext --inplace
```

Optional:
```bash
pip install line_profiler  # For profiling
pip install mpi4py         # For MPI (not fully implemented)
```

**Note**: This project was developed on Windows 10 with an Intel i9-9900k (8 cores) and 32GB RAM.

## Scalability Analysis

The project includes extensive scalability testing with progressively larger inputs (20% increments):

- **Pure Python** becomes exponentially slow, taking over 30 minutes for 8907×8907 images
- **NumPy variants** scale better but still show polynomial growth
- **Numba, Cython, and Better Numba** converge to nearly identical performance at large scales
- **Guvectorized Numba** maintains dominance: only 6 minutes for 32,000×32,000 images (over 1 billion pixels!)
- **Multiprocessing** starts slow but scales excellently, outpacing NumPy/Numexpr at higher resolutions

### Why Guvectorized Numba Wins

The `@guvectorize` decorator with `target='parallel'` combines:
1. **JIT compilation** to machine code
2. **Automatic parallelization** across all CPU cores
3. **Manual complex number arithmetic** avoiding Python object overhead
4. **Optimal memory access patterns** for cache efficiency

Result: Near-C performance with Python-like syntax.

## Implementation Highlights

### Pure Python → NumPy: Vectorization
```python
# Before: Nested loops
for r in realNums:
    for i in imagNums:
        escapeCounts.append(mandelbrot(complex(r, i)))

# After: NumPy arrays
complexNums = realNums + imagNums[:, None] * 1j
escapeCounts = mandelbrot_numpy(complexNums)
```

### NumPy → Better NumPy: Mathematical Optimization
```python
# Before: abs(z) > 2
done = np.greater(abs(z), 2.0)

# After: Pythagorean theorem (avoids expensive sqrt)
done = np.less(z.real * z.real + z.imag * z.imag, 4.0)
```

### Numba → Guvectorized Numba: Parallelization
```python
@guvectorize([(complex64[:], int32[:], int32[:])],
             '(n),()->(n)', target='parallel')
def mandelbrot_numba_betterer(c, maxIterations, output):
    # Numba handles parallel execution automatically!
```

## Project Structure

```
code/
├── main.py                          # Orchestrates all tests and visualizations
├── utils.py                         # Timer decorator and function registry
├── mandelbrot_visualizer.py         # Matplotlib rendering
├── mandelbrot1_purepython.py        # Baseline: nested loops
├── mandelbrot2_numpy.py             # First vectorization attempt
├── mandelbrot2a_numpy_better.py     # Pythagorean optimization
├── mandelbrot3_numba.py             # Basic JIT compilation
├── mandelbrot3a_numba_better.py     # Manual complex arithmetic
├── mandelbrot3a_numba_betterer.py   # Guvectorized parallel execution
├── mandelbrot4_numexpr.py           # String expression evaluation
├── mandelbrot5_cython.py            # Cython with static typing
├── mandelbrot6_multiprocessing.py   # Pool.map() parallelization
├── mandelbrot7_mpi.py               # MPI skeleton (incomplete)
├── c_mandelbrot.pyx                 # Cython source
└── setup.py                         # Cython build configuration
```

## Key Learnings

1. **Embarrassingly parallel problems** (like Mandelbrot) benefit most from multicore parallelization
2. **Numba's `@guvectorize`** with `target='parallel'` is the sweet spot for array operations
3. **Low-level optimizations compound**: Data types, operation order, and memory layout all matter
4. **Avoid premature abstraction**: Python classes add overhead for numerical code
5. **Profile before optimizing**: The `if abs(z) > 2` check consumed most execution time
6. **Different techniques scale differently**: Multiprocessing catches up to NumPy at large sizes

## Future Work

- Implement full MPI version for distributed computing
- Explore GPU acceleration with CUDA/OpenCL
- Optimize visualization with GPU rendering
- Leverage Mandelbrot symmetry to halve computations
- Test alternative Python interpreters (PyPy, Pyston)
- Compare against Julia and Fortran implementations

## Documentation

For detailed analysis, mathematical explanations, profiling results, and scalability charts, see:
- **[Project Report PDF](PROCKOT_MA553_FINALPROJECT_REPORT.pdf)** - Complete 30-page analysis
- **[Presentation](PROCKOT_MA553_FINALPROJECT_PRESENTATION.pdf)** - Visual summary

## Author

**Tyler T. Procko**

MA553 - High Performance Scientific Computing
Fall 2022, Dr. Khanal

All code, research, analysis, benchmarking, visualizations, and documentation were completed by Tyler T. Procko as part of the MA553 final project.

### README Attribution

This README was written by Claude (Anthropic) in 2025 to document Tyler's original 2022 project, based on the comprehensive 30-page project report.

## License

MIT License - see [LICENSE](LICENSE) file for details

---

*"From 3.6 seconds to 0.01 seconds—that's the power of optimization."*
