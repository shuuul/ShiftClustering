# ShiftClustering

Fast clustering algorithms for Python.

## Features

- **MeanShiftPP**: An optimized mean shift clustering algorithm
- **GridShift**: A grid-based clustering algorithm  
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
ms = MeanShiftPP(bandwidth=1.0, max_iter=300)
labels = ms.fit_predict(X)

# GridShift clustering  
gs = GridShift(bandwidth=1.0, max_iters=300)
labels = gs.fit_predict(X)
```

## Benchmarks

The package includes benchmarks to evaluate performance:

```bash
# Install with benchmark dependencies
uv sync --all-extras

# Run algorithm comparison benchmark
cd benchmark
uv run python benchmark_sklearn.py
```

See the [benchmark README](benchmark/README.md) for detailed information.

## API Reference

### MeanShiftPP

```python
MeanShiftPP(bandwidth, threshold=1e-4, max_iter=300)
```

**Parameters:**
- `bandwidth`: Radius for binning points
- `threshold`: Stop when L2 norm between iterations < threshold  
- `max_iter`: Maximum number of iterations

### GridShift

```python
GridShift(bandwidth, threshold=1e-4, max_iters=300)
```

**Parameters:**
- `bandwidth`: Radius for binning points
- `threshold`: Stop when L2 norm between iterations < threshold
- `max_iters`: Maximum number of iterations  

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
└── README.md            # Benchmark documentation
```

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
- Reorganized project structure with separate include/ and src/ directories
- Scikit-build-core integration for robust C++ extension building
- Comprehensive benchmark suite for performance evaluation

## Acknowledgements

This repo is based on the following repository

- [meanshiftpp](https://github.com/jenniferjang/meanshiftpp)
- [GridShift](https://github.com/abhisheka456/GridShift)

Please cite their papers if you use this package.