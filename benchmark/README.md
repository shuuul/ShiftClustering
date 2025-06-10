# ShiftClustering Benchmarks

This directory contains benchmarks for evaluating the performance of ShiftClustering algorithms.

## Available Benchmarks

### 1. Algorithm Comparison (`benchmark_sklearn.py`)

Compares the performance of three clustering algorithms:
- **sklearn.cluster.MeanShift** (baseline)
- **shiftclustering.MeanShiftPP** (our optimized implementation)  
- **shiftclustering.GridShift** (our grid-based implementation)

**Features:**
- Multiple runs for statistical stability
- Quality metrics (silhouette score)
- Speedup analysis relative to sklearn
- Moderate dataset size (sklearn MeanShift is slow)

**Usage:**
```bash
cd benchmark
python benchmark_sklearn.py
```

**Example Output:**
```
================================================================================
CLUSTERING ALGORITHM BENCHMARK RESULTS
================================================================================
Algorithm       Time (s)     ±Std     Clusters   Silhouette  
--------------------------------------------------------------------------------
MeanShift       0.5564       ±0.0086  38         0.3164      
MeanShiftPP     0.0515       ±0.0224  16         0.4352      
GridShift       0.0070       ±0.0001  16         -0.1140     
```

### 2. Parallel Processing (`benchmark_njobs.py`)

Tests the scalability of our algorithms with different numbers of parallel jobs (`n_jobs`).

**Features:**
- Large datasets to see parallel benefits
- Tests multiple n_jobs values (1, 2, 4, max_cores, -1)
- Speedup and efficiency analysis
- Only tests our algorithms (MeanShiftPP and GridShift)
- Recommendations for optimal n_jobs

**Usage:**
```bash
cd benchmark
python benchmark_njobs.py
```

**Example Output:**
```
==========================================================================================
MEANSHIFTPP PARALLEL PERFORMANCE RESULTS
==========================================================================================
n_jobs   Time (s)     ±Std     Clusters   Speedup    Efficiency  
------------------------------------------------------------------------------------------
1        4.1473       ±0.0134  31         1.00       1.000       
2        2.8321       ±0.0350  58         1.46       0.732       
4        1.4051       ±0.0530  120        2.95       0.738       
12       1.2380       ±0.0120  375        3.35       0.279       
```

## Key Metrics Explained

### Performance Metrics
- **Time (s)**: Average execution time
- **±Std**: Standard deviation across multiple runs
- **Clusters**: Number of clusters found
- **Speedup**: Performance improvement relative to single-threaded (n_jobs=1)
- **Efficiency**: Speedup divided by number of cores (ideal = 1.0)

### Quality Metrics
- **Silhouette Score**: Clustering quality measure (-1 to 1, higher is better)

## Results Summary

Based on typical benchmark results:

### Algorithm Performance (Single-threaded)
1. **GridShift**: Fastest (~80x faster than sklearn)
2. **MeanShiftPP**: Fast (~11x faster than sklearn)  
3. **sklearn MeanShift**: Slowest (baseline)

### Parallel Scaling
- **Optimal n_jobs**: Usually 2-4 cores for best efficiency
- **Maximum speedup**: ~3-4x on modern multi-core systems
- **Diminishing returns**: Beyond 4-8 cores due to overhead

### Quality Trade-offs
- **MeanShiftPP**: Best clustering quality (highest silhouette scores)
- **GridShift**: Fastest but may sacrifice some quality for speed
- **sklearn MeanShift**: Good quality but very slow

## Configuration

You can modify the benchmark parameters by editing the configuration sections in each script:

```python
# Algorithm comparison benchmark
n_samples = 3000    # Keep moderate for sklearn MeanShift
n_features = 3
n_centers = 20
bandwidth = 1.5

# Parallel processing benchmark  
n_samples = 50000   # Large dataset for parallel benefits
n_features = 4
n_centers = 50
bandwidth = 2.0
```

## Requirements
`uv sync --extra benchmark`
- `numpy`
- `scikit-learn`
- `shiftclustering` (this package)

## Notes

- The parallel benchmark uses large datasets (50K+ samples) to see meaningful parallel speedup
- sklearn MeanShift doesn't support efficient parallelization, so it's excluded from n_jobs testing
- Results may vary based on your hardware and dataset characteristics
- For production use, test with your specific data to find optimal parameters 