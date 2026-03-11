# ShiftClustering Benchmarks

Performance benchmarks for ShiftClustering algorithms.

## Available Benchmarks

| Script | Compares |
|---|---|
| `benchmark_sklearn.py` | MeanShiftPP & GridShift vs sklearn MeanShift |
| `benchmark_mspp_numba.py` | MeanShiftPP (C++/Cython) vs Numba |
| `benchmark_mspp_torch.py` | MeanShiftPP (C++/Cython) vs PyTorch (CPU) |
| `benchmark_localshift_numba.py` | LocalShift (C++/Cython + OpenMP) vs Numba |

## Quick Usage

```bash
uv sync --extra benchmark

python benchmark/benchmark_sklearn.py
python benchmark/benchmark_mspp_numba.py
python benchmark/benchmark_mspp_torch.py
python benchmark/benchmark_localshift_numba.py
```

## Results

> Collected on Apple M4 Max, Python 3.12, numpy 2.4.3.
> Each timing is the mean of 3 runs.

### sklearn Comparison (`benchmark_sklearn.py`)

3 000 samples, 3 features, 20 blob centres, bandwidth = 1.5.

| Algorithm | Time (s) | Speedup | Clusters | Silhouette |
|---|---|---|---|---|
| sklearn MeanShift | 0.7288 | 1.0x | 38 | 0.3164 |
| **MeanShiftPP** | 0.0368 | **19.8x** | 16 | **0.4352** |
| **GridShift** | 0.0069 | **104.9x** | 16 | -0.1140 |

### Numba Comparison (`benchmark_mspp_numba.py`)

| Dataset | Numba (s) | C++/Cython (s) | Speedup | ARI between |
|---|---|---|---|---|
| 1 000 × 2D, 5 centres | 0.920* | 0.003 | 354x | 0.997 |
| 3 000 × 3D, 10 centres | 0.054 | 0.034 | 1.6x | 0.996 |
| 5 000 × 4D, 15 centres | 0.170 | 0.160 | 1.1x | 0.998 |

*\*First run includes Numba JIT compilation; subsequent runs are ~0.003 s.*

### PyTorch Comparison (`benchmark_mspp_torch.py`)

CPU-only PyTorch (no GPU).

| Dataset | PyTorch (s) | C++/Cython (s) | Speedup | ARI between |
|---|---|---|---|---|
| 1 000 × 2D, 5 centres | 0.083 | 0.003 | 28x | 0.782 |
| 3 000 × 3D, 10 centres | 8.261 | 0.033 | 248x | 0.786 |
| 5 000 × 4D, 15 centres | 4.777 | 0.158 | 30x | 0.763 |

### LocalShift Comparison (`benchmark_localshift_numba.py`)

1 000 atoms in a 600^3 density map, fmaxd = 5.0, fsiv = 0.1, 50 steps.

| Implementation | Time (s) | Speedup |
|---|---|---|
| Numba | 0.225 | 1.0x |
| **Cython (class)** | 0.023 | **9.7x** |
| **Cython (functional)** | 0.024 | **9.3x** |

Numerical difference between Numba and Cython: mean 0.00025, max 0.0014 (float32 precision).

## Summary

| Algorithm | vs Baseline | Notes |
|---|---|---|
| **GridShift** | ~105x vs sklearn MeanShift | Fastest; may sacrifice clustering quality |
| **MeanShiftPP** | ~20x vs sklearn MeanShift | Best quality-speed trade-off |
| **MeanShiftPP** | ~1-30x vs PyTorch CPU | PyTorch overhead dominates on CPU |
| **MeanShiftPP** | ~1-1.6x vs Numba (warm) | Comparable after JIT; C++ wins on small data |
| **LocalShift** | ~10x vs Numba | OpenMP parallelism over per-point kernels |

## Configuration

Edit benchmark parameters in the script files:

```python
# Example: benchmark_sklearn.py
n_samples = 3000
n_features = 3
n_centers = 20
bandwidth = 1.5
```

## Requirements

```bash
uv sync --extra benchmark
```

Includes: `numpy`, `scikit-learn`, `numba`, `torch`
