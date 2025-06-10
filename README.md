# ShiftClustering

Fast clustering algorithms for Python with C++/Cython implementation and OpenMP parallel support.

## Features

- **MeanShiftPP**: Optimized mean shift clustering algorithm
- **LocalShift**: Local shift algorithm for 3D point cloud optimization in cryo-EM contexts
- **GridShift**: Grid-based clustering algorithm  
- **High Performance**: C++ implementations with Cython wrappers and parallel processing
- **Scikit-learn Compatible**: Familiar API with both class and functional interfaces

## Installation

```bash
# Basic installation
uv sync

# Install with benchmark dependencies
uv sync --all-extras
```

## Requirements

- Python ≥ 3.10
- NumPy ≥ 2.0
- C++ compiler with C++17 support
- OpenMP support

## Quick Start

### Basic Usage

```python
import numpy as np
from shiftclustering import meanshiftpp, localshift, gridshift

# Generate sample data
X = np.random.randn(1000, 2).astype(np.float32)

# MeanShift++ clustering
labels = meanshiftpp(X, bandwidth=1.0)

# GridShift clustering  
labels = gridshift(X, bandwidth=1.0)

# For 3D point cloud optimization (cryo-EM)
atoms = np.random.rand(100, 3) * 50
density_map = np.random.rand(50, 50, 50)
optimized_atoms = localshift(atoms, density_map, fmaxd=5.0, fsiv=0.1)
```

### Class-based Interface (Scikit-learn compatible)

```python
from shiftclustering import MeanShiftPP, LocalShift, GridShift

# MeanShiftPP
ms = MeanShiftPP(bandwidth=1.0, max_iter=300)
labels = ms.fit_predict(X)

# LocalShift for cryo-EM optimization
ls = LocalShift(fmaxd=5.0, fsiv=0.1, n_steps=100)
optimized_atoms = ls.fit_predict(atoms, density_map)
```

## Available Algorithms

### MeanShiftPP
Fast implementation of mean shift clustering with optimized binning strategy.

### GridShift  
Grid-based clustering algorithm for large datasets.

### LocalShift
3D point cloud optimization algorithm for cryo-EM structure refinement. Iteratively shifts points towards local maxima in density maps using Gaussian kernel weighting.

**Citation Required**: If you use LocalShift in your research, please cite:

> Terashi, Genki, and Daisuke Kihara. "De novo main-chain modeling for EM maps using MAINMAST." *Nature Communications* 9, no. 1 (2018): 1618.
> 
> https://www.nature.com/articles/s41467-018-04053-7

## Performance

The package provides significant speedups over pure Python implementations:
- **LocalShift**: ~10x faster than Numba implementation
- **MeanShiftPP**: ~11x faster than sklearn.cluster.MeanShift  
- **GridShift**: ~80x faster than sklearn.cluster.MeanShift

See [benchmark/](benchmark/) for detailed performance comparisons.

## Project Structure

```
shiftclustering/
├── include/          # C++ header files
├── src/              # Cython implementation files  
└── __init__.py       # Main module

benchmark/            # Performance benchmarks
```

## Building from Source

```bash
# Development installation
uv sync --all-extras

# Reinstall with rebuild
uv sync --all-extras --reinstall
```

## License

GPL-3.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This project builds upon:
- [meanshiftpp](https://github.com/jenniferjang/meanshiftpp)
- [GridShift](https://github.com/abhisheka456/GridShift)

Please cite the original papers when using these algorithms.