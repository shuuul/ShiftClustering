# ShiftClustering Benchmarks

Performance benchmarks for ShiftClustering algorithms.

## Available Benchmarks

### Algorithm Comparison (`benchmark_sklearn.py`)

Compares clustering algorithms against sklearn baselines:
- **sklearn.cluster.MeanShift** (baseline)
- **shiftclustering.MeanShiftPP** (optimized implementation)  
- **shiftclustering.GridShift** (grid-based implementation)

### LocalShift Comparison (`benchmark_localshift_numba.py`)

Compares LocalShift implementations:
- **Numba implementation** (baseline)
- **Cython implementation** (class interface)
- **Cython implementation** (functional interface)

## Quick Usage

```bash
cd benchmark

# Compare clustering algorithms
python benchmark_sklearn.py

# Compare LocalShift implementations  
python benchmark_localshift_numba.py
```

## Example Results

### Algorithm Performance
- **GridShift**: ~80x faster than sklearn MeanShift
- **MeanShiftPP**: ~11x faster than sklearn MeanShift  
- **LocalShift**: ~10x faster than Numba implementation

### Quality vs Speed Trade-offs
- **MeanShiftPP**: Best clustering quality with good speed
- **GridShift**: Fastest but may sacrifice some quality
- **LocalShift**: High accuracy for cryo-EM optimization tasks

## Configuration

Edit benchmark parameters in the script files:

```python
# Example configuration
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