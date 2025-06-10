# ShiftClustering Benchmarks

This directory contains benchmarks for evaluating the performance of ShiftClustering algorithms.

## Available Benchmarks

### Algorithm Comparison (`benchmark_sklearn.py`)

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

## Key Metrics Explained

### Performance Metrics
- **Time (s)**: Average execution time
- **±Std**: Standard deviation across multiple runs
- **Clusters**: Number of clusters found

### Quality Metrics
- **Silhouette Score**: Clustering quality measure (-1 to 1, higher is better)

## Results Summary

Based on typical benchmark results:

### Algorithm Performance
1. **GridShift**: Fastest (~80x faster than sklearn)
2. **MeanShiftPP**: Fast (~11x faster than sklearn)  
3. **sklearn MeanShift**: Slowest (baseline)

### Quality Trade-offs
- **MeanShiftPP**: Best clustering quality (highest silhouette scores)
- **GridShift**: Fastest but may sacrifice some quality for speed
- **sklearn MeanShift**: Good quality but very slow

## Configuration

You can modify the benchmark parameters by editing the configuration sections in the script:

```python
# Algorithm comparison benchmark
n_samples = 3000    # Keep moderate for sklearn MeanShift
n_features = 3
n_centers = 20
bandwidth = 1.5
```

## Requirements

```bash
uv sync --extra benchmark
```

- `numpy`
- `scikit-learn`
- `shiftclustering` (this package)

## Notes

- Results may vary based on your hardware and dataset characteristics
- For production use, test with your specific data to find optimal parameters 