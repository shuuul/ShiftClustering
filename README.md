# ShiftClustering

Fast clustering algorithms with OpenMP parallel support for Python.

## Features

- **MeanShiftPP**: An optimized mean shift clustering algorithm
- **GridShift**: A grid-based clustering algorithm  
- **Parallel Processing**: Both algorithms support OpenMP-based parallelization via the `n_jobs` parameter
- **Scikit-learn Compatible**: Similar API to scikit-learn clustering algorithms
- **High Performance**: C++ implementations with Cython wrappers

## Installation

```bash
# Basic installation
uv sync

# Install with benchmark dependencies (includes scikit-learn)
uv sync --all-extras

# Alternative: specific extra
uv sync --extra benchmark
```

## Requirements

- Python ≥ 3.12
- NumPy ≥ 2.2.6
- C++ compiler with C++17 support
- OpenMP (optional, for parallel processing)

### Optional Dependencies

- `scikit-learn>=1.7.0` (for benchmarks)

## Usage

### Basic Usage

```python
import numpy as np
from shiftclustering import MeanShiftPP, GridShift

# Create sample data
X = np.random.randn(1000, 2).astype(np.float32)

# MeanShiftPP clustering
ms = MeanShiftPP(bandwidth=1.0, max_iter=300, n_jobs=-1)
labels = ms.fit_predict(X)

# GridShift clustering  
gs = GridShift(bandwidth=1.0, max_iters=300, n_jobs=-1)
labels = gs.fit_predict(X)
```

### Parallel Processing with n_jobs

The `n_jobs` parameter controls the number of OpenMP threads used for parallel computation:

- `n_jobs=1`: Single-threaded (default)
- `n_jobs=2, 4, etc.`: Use specified number of threads
- `n_jobs=-1`: Use all available CPU cores

```python
# Single-threaded
clusterer = MeanShiftPP(bandwidth=1.0, n_jobs=1)

# Use 4 threads
clusterer = MeanShiftPP(bandwidth=1.0, n_jobs=4)

# Use all available cores
clusterer = MeanShiftPP(bandwidth=1.0, n_jobs=-1)
```

## Benchmarks

The package includes comprehensive benchmarks to evaluate performance:

```bash
# Install with benchmark dependencies
uv sync --all-extras

# Run algorithm comparison benchmark
cd benchmark
uv run python benchmark_sklearn.py

# Run parallel processing benchmark
uv run python benchmark_njobs.py
```

See the [benchmark README](benchmark/README.md) for detailed information.

## API Reference

### MeanShiftPP

```python
MeanShiftPP(bandwidth, threshold=1e-4, max_iter=300, n_jobs=1)
```

**Parameters:**
- `bandwidth`: Radius for binning points
- `threshold`: Stop when L2 norm between iterations < threshold  
- `max_iter`: Maximum number of iterations
- `n_jobs`: Number of parallel threads (-1 for all cores)

### GridShift

```python
GridShift(bandwidth, threshold=1e-4, max_iters=300, n_jobs=1)
```

**Parameters:**
- `bandwidth`: Radius for binning points
- `threshold`: Stop when L2 norm between iterations < threshold
- `max_iters`: Maximum number of iterations  
- `n_jobs`: Number of parallel threads (-1 for all cores)

## Project Structure

```
shiftclustering/
├── include/          # C++ header files
│   ├── gridshift.h   # GridShift algorithm
│   ├── meanshiftpp.h # MeanShiftPP algorithm  
│   └── utils.h       # Utility functions
├── src/              # Cython source files
│   ├── _gridshift.pyx
│   └── _meanshiftpp.pyx
└── __init__.py       # Main module

benchmark/            # Performance benchmarks
├── benchmark_sklearn.py  # Algorithm comparison
├── benchmark_njobs.py    # Parallel processing tests
└── README.md            # Benchmark documentation
```

## Performance Notes

- OpenMP parallelization provides the best speedup for larger datasets (>10k samples)
- For small datasets, single-threaded performance may be comparable due to overhead
- The algorithms use critical sections for thread-safe map operations
- Performance depends on your system's CPU cores and OpenMP implementation

## Building from Source

The package uses scikit-build-core for building C++ extensions:

```bash
# Development installation
uv sync

# Reinstall with rebuild
uv sync --reinstall

# Install with all extras (benchmarks, etc.)
uv sync --all-extras
```

**Build Requirements:**
- CMake ≥ 3.15
- C++ compiler with C++17 support
- Cython ≥ 3.1.2
- NumPy ≥ 1.20.0 
- OpenMP (optional but recommended)

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests and run benchmarks
5. Submit a pull request

## Changelog

### v0.1.0
- Initial release with MeanShiftPP and GridShift algorithms
- Added OpenMP parallel processing support via `n_jobs` parameter
- Reorganized project structure with separate include/ and src/ directories
- Scikit-build-core integration for robust C++ extension building
- Comprehensive benchmark suite for performance evaluation

## Acknowledgements

This repo is based on the following repository

- [meanshiftpp](https://github.com/jenniferjang/meanshiftpp)
- [GridShift](https://github.com/abhisheka456/GridShift)

Please cite their papers if you use this package.